import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from src.frag_adder.adder_random import initialize_random_adder
from src.frag_adder.configs.config import get_config
from src.frag_adder.ligand_node import LigandNode, LigandNode_List
from src.utils.struc_tools import read_mae, write_mae


def parse_csv_paths(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def select_open_bond_across_pockets(open_bond_payload, step_idx):
    """
    TODO: Implement multi-pocket open-bond selection criterion.

    Expected return:
      {
        "atom_from": int,
        "atom_h": int
      }
    """
    raise NotImplementedError("Implement select_open_bond_across_pockets(...)")


def select_fragment_across_pockets(fragment_payload, step_idx):
    """
    TODO: Implement multi-pocket fragment selection criterion.

    Expected return:
      {
        "indices_by_pocket": {
          "<pocket_id>": int
        }
      }
    """
    raise NotImplementedError("Implement select_fragment_across_pockets(...)")


def dump_json(path, obj):
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2)


def dump_fragment_mae(path, candidates):
    candidate_nodes = LigandNode_List()
    for node in candidates:
        candidate_nodes.append(node)

    def title_format_func(node):
        s = node.score if node.score is not None else 0.0
        return f"d{node.depth}_{node.fragname}_{s:.3f}"

    candidate_nodes.write_to_file(path, title_format_func)


def resolve_open_bond_by_atoms(adder, parent_node, atom_from, atom_h):
    for open_bond in adder.get_open_bonds(parent_node.ligand):
        if open_bond[0].index == atom_from and open_bond[1].index == atom_h:
            return open_bond
    return None


def score_open_bonds_for_pocket(args):
    pocket_id, adder, parent_node = args
    open_bonds = adder.sample_open_bonds(parent_node)
    rows = []
    for idx, ob in enumerate(open_bonds):
        rows.append(
            dict(
                idx=idx,
                score=float(adder.open_bond_scorefxn(ob, adder.goal)),
                atom_from=int(ob[0].index),
                atom_h=int(ob[1].index),
                depth=int(parent_node.depth + 1),
            )
        )
    return pocket_id, rows


def score_fragments_for_pocket(args):
    pocket_id, adder, parent_node, selected_open = args
    open_bond = resolve_open_bond_by_atoms(
        adder, parent_node, selected_open["atom_from"], selected_open["atom_h"]
    )
    if open_bond is None:
        return pocket_id, [], []

    raw_fragments = adder.sample_fragments(parent_node, open_bond, adder.fragname_list)
    raw_fragments = [n for n in raw_fragments if adder.reaction_filter(n)]
    raw_fragments = adder.filter_top(
        raw_fragments, adder.max_fragments, adder.fragment_scorefxn
    )

    fragment_candidates = []
    for new_node in raw_fragments:
        candidate_dihedrals = adder.sample_dihedrals(new_node, parent_node)
        candidate_dihedrals = adder.filter_top(
            candidate_dihedrals, adder.max_dihedrals, adder.dihedral_scorefxn
        )
        fragment_candidates.extend(candidate_dihedrals)

    if len(fragment_candidates) == 0:
        return pocket_id, [], []

    scores = adder.heuristic_timed(fragment_candidates, adder.goal)
    rows = []
    for idx, (node, score) in enumerate(zip(fragment_candidates, scores)):
        node.score = float(score)
        rows.append(
            dict(
                idx=idx,
                score=float(score),
                fragname=node.fragname,
                attachment_atom=node.get_attachment_atom_original_id(),
                depth=int(node.depth),
            )
        )
    return pocket_id, rows, fragment_candidates


def initialize_adder(config, e3nn_env_path):
    adder_type = config["adder_type"]
    if adder_type == "ML_2model":
        sys.path.insert(0, e3nn_env_path)
        from src.frag_adder.adder_2model import initialize_2model_adder

        return initialize_2model_adder(config)
    if adder_type == "random":
        return initialize_random_adder(config)
    raise ValueError(f"Unsupported adder_type: {adder_type}")


def run_multi_pocket(args):
    seed_paths = parse_csv_paths(args.seed_ligand_paths)
    pocket_paths = parse_csv_paths(args.protein_pocket_paths)
    if len(seed_paths) != len(pocket_paths):
        raise ValueError("seed_ligand_paths and protein_pocket_paths must have equal lengths")

    config = get_config(args.config_name)
    config["max_depth"] = args.max_steps
    config["goal_type"] = "number_steps"
    config["advanced_config"]["save_candidate_scores_json"] = True

    output_root = args.output_folder_path
    os.makedirs(output_root, exist_ok=True)

    states = {}
    for i, (seed_path, pocket_path) in enumerate(zip(seed_paths, pocket_paths)):
        pocket_id = f"pk{i}"
        seed = read_mae(seed_path)[0]
        pocket = read_mae(pocket_path)[0]
        adder = initialize_adder(config.copy(), args.e3nn_env_path)
        adder.goal = {"type": "depth", "value": args.max_steps}
        adder.debug_config["debug_output_root"] = os.path.join(output_root, pocket_id) + "/"
        os.makedirs(adder.debug_config["debug_output_root"], exist_ok=True)
        states[pocket_id] = dict(adder=adder, node=LigandNode(pocket, seed))

    for step in range(1, args.max_steps + 1):
        with ThreadPoolExecutor(max_workers=len(states)) as executor:
            jobs = [
                (pid, s["adder"], s["node"])
                for pid, s in states.items()
            ]
            open_payload = dict(executor.map(score_open_bonds_for_pocket, jobs))

        dump_json(os.path.join(output_root, f"d{step}_open_bonds_all_pockets.json"), open_payload)
        selected_open = select_open_bond_across_pockets(open_payload, step)

        with ThreadPoolExecutor(max_workers=len(states)) as executor:
            jobs = [
                (pid, s["adder"], s["node"], selected_open)
                for pid, s in states.items()
            ]
            fragment_results = list(executor.map(score_fragments_for_pocket, jobs))

        fragment_payload = {}
        fragment_nodes = {}
        for pocket_id, rows, candidates in fragment_results:
            fragment_payload[pocket_id] = rows
            fragment_nodes[pocket_id] = candidates
            if len(candidates):
                mae_path = os.path.join(output_root, f"d{step}_{pocket_id}_fragment_candidates.mae")
                dump_fragment_mae(mae_path, candidates)

        dump_json(os.path.join(output_root, f"d{step}_fragment_candidates_all_pockets.json"), fragment_payload)
        selected_frag = select_fragment_across_pockets(fragment_payload, step)

        for pocket_id, sel_idx in selected_frag["indices_by_pocket"].items():
            candidates = fragment_nodes[pocket_id]
            if sel_idx < 0 or sel_idx >= len(candidates):
                raise ValueError(f"Invalid fragment index {sel_idx} for {pocket_id}")
            states[pocket_id]["node"] = candidates[sel_idx]

    # Save final grown ligands for each pocket.
    for pocket_id, state in states.items():
        out_file = os.path.join(output_root, f"{pocket_id}_final.mae")
        write_mae(out_file, [state["node"].ligand])


def get_args():
    parser = argparse.ArgumentParser(description="Multi-pocket FRAME growth")
    parser.add_argument("--config_name", choices=["config_random", "config_ML"], default="config_ML")
    parser.add_argument("--seed_ligand_paths", type=str, required=True,
                        help="Comma-separated list of seed ligand MAE paths (one per pocket).")
    parser.add_argument("--protein_pocket_paths", type=str, required=True,
                        help="Comma-separated list of pocket MAE paths.")
    parser.add_argument("--output_folder_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--e3nn_env_path", type=str,
                        default="/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/software/anaconda3/envs/e3nn/lib/python3.8/site-packages")
    return parser.parse_args()


def main():
    args = get_args()
    run_multi_pocket(args)


if __name__ == "__main__":
    main()
