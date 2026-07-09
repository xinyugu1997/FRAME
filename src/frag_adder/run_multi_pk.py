import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from src.frag_adder.adder_random import initialize_random_adder
from src.frag_adder.configs.config import get_config
from src.frag_adder.ligand_node import LigandNode, LigandNode_List
from src.utils.struc_tools import read_mae, write_mae


class BranchExhaustedError(RuntimeError):
    """Raised when the current branch has no valid growth moves left."""


def parse_csv_paths(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_csv_floats(text):
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _bonded_atom_indices(atom):
    return sorted(neighbor.index for neighbor in atom.bonded_atoms)


def _atom_identity(atom):
    return dict(
        index=int(atom.index),
        element=atom.element,
        atomic_number=int(atom.atomic_number),
        bonded_atom_indices=_bonded_atom_indices(atom),
    )


def validate_ligand_atom_index_consistency(states, context):
    """
    Validate that each pocket's ligand uses the same atom-index identity map.

    Multi-pocket selection matches open bonds across pockets by atom index, so
    every pocket must keep the same atom index -> chemical atom mapping during
    the whole growth trajectory. Coordinates are intentionally ignored because
    each pocket can have a different 3D pose; atom count, element/atomic number,
    and bonded-neighbor indices must remain identical.
    """
    if not states:
        raise ValueError(f"{context}: no pocket states were provided")

    pocket_ids = list(states.keys())
    reference_pocket_id = pocket_ids[0]
    reference_ligand = states[reference_pocket_id]["node"].ligand
    reference_signature = [_atom_identity(atom) for atom in reference_ligand.atom]

    for pocket_id in pocket_ids[1:]:
        ligand = states[pocket_id]["node"].ligand
        if ligand.atom_total != reference_ligand.atom_total:
            raise ValueError(
                f"{context}: ligand atom count mismatch between "
                f"{reference_pocket_id} ({reference_ligand.atom_total}) and "
                f"{pocket_id} ({ligand.atom_total})"
            )

        signature = [_atom_identity(atom) for atom in ligand.atom]
        for reference_atom, atom in zip(reference_signature, signature):
            if atom != reference_atom:
                raise ValueError(
                    f"{context}: ligand atom-index identity mismatch at atom "
                    f"{reference_atom['index']} between {reference_pocket_id} "
                    f"and {pocket_id}. Expected {reference_atom}, got {atom}."
                )


def select_open_bond_across_pockets(
        open_bond_payload, step_idx, pocket_weights=None,
        num_open_bonds_to_sample=1):
    """
    Select open bonds by weighted multi-pocket scores and softmax sampling.

    Open bonds are matched by their ligand atom ids, (atom_from, atom_h). Only
    bonds available in every pocket are considered, because selected bonds must
    be usable for fragment scoring in all pockets. For each common bond, compute
    the weighted score sum across pockets. Since lower scores are better in
    FRAME, sampling probabilities are computed with softmax over the negated
    weighted scores. Sampling is performed with replacement, then duplicate bond
    selections are removed while preserving sampling order.

    Returns:
      {
        "open_bonds": [
          {
            "atom_from": int,
            "atom_h": int,
            "score": float,
            "probability": float,
            "scores_by_pocket": {"<pocket_id>": float}
          }
        ]
      }
    """
    if not open_bond_payload:
        raise ValueError("No open-bond payloads were provided")
    if num_open_bonds_to_sample < 1:
        raise ValueError("num_open_bonds_to_sample must be >= 1")

    pocket_ids = list(open_bond_payload.keys())
    if pocket_weights is None or len(pocket_weights) == 0:
        pocket_weights = [1.0 for _ in pocket_ids]
    if len(pocket_weights) != len(pocket_ids):
        raise ValueError(
            f"Expected {len(pocket_ids)} pocket weights, got {len(pocket_weights)}")

    # Open-bond score summaries should only include positively weighted pockets.
    # Treat zero/negative pocket weights as disabled for this selection stage.
    weights_by_pocket = {
        pocket_id: max(0.0, float(weight))
        for pocket_id, weight in zip(pocket_ids, pocket_weights)
    }
    weighted_scores_by_bond = {}
    raw_scores_by_bond = {}
    expected_pocket_count = len(open_bond_payload)

    for pocket_id, rows in open_bond_payload.items():
        for row in rows:
            bond_key = (row["atom_from"], row["atom_h"])
            weighted_scores_by_bond.setdefault(bond_key, 0.0)
            raw_scores_by_bond.setdefault(bond_key, {})
            score = float(row["score"])
            weighted_scores_by_bond[bond_key] += weights_by_pocket[pocket_id] * score
            raw_scores_by_bond[bond_key][pocket_id] = score

    common_bonds = [
        bond_key
        for bond_key, pocket_scores in raw_scores_by_bond.items()
        if len(pocket_scores) == expected_pocket_count
    ]
    if not common_bonds:
        raise BranchExhaustedError("No open bond is available in every pocket")

    # Lower weighted score should have higher sampling probability. Subtract the
    # max logit for numerical stability.
    logits = [-weighted_scores_by_bond[bond_key] for bond_key in common_bonds]
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    normalizer = sum(exp_logits)
    probabilities = [value / normalizer for value in exp_logits]

    sampled_bonds = random.choices(
        common_bonds, weights=probabilities, k=num_open_bonds_to_sample)

    selected_bonds = []
    seen = set()
    probability_by_bond = dict(zip(common_bonds, probabilities))
    for bond_key in sampled_bonds:
        if bond_key in seen:
            continue
        seen.add(bond_key)
        atom_from, atom_h = bond_key
        selected_bonds.append(dict(
            atom_from=atom_from,
            atom_h=atom_h,
            score=weighted_scores_by_bond[bond_key],
            probability=probability_by_bond[bond_key],
            scores_by_pocket=raw_scores_by_bond[bond_key],
        ))

    return dict(open_bonds=selected_bonds)


#def select_fragment_across_pockets(
#        fragment_payload, step_idx, pocket_weights=None,
#        num_fragments_to_sample=1):
#    """
#    Select fragments by weighted best-dihedral scores across pockets.
#
#    Fragment candidates are grouped by (open_atom_from, open_atom_h, fragname).
#    Within each pocket and fragment group, only the candidate dihedral with the
#    lowest score is kept. The kept scores are combined with pocket weights,
#    converted into sampling probabilities with softmax over negated weighted
#    scores, sampled with replacement, and de-duplicated while preserving sample
#    order.
#
#    Returns:
#      {
#        "fragments": [
#          {
#            "fragname": str,
#            "open_atom_from": int,
#            "open_atom_h": int,
#            "score": float,
#            "probability": float,
#            "scores_by_pocket": {"<pocket_id>": float},
#            "indices_by_pocket": {"<pocket_id>": int}
#          }
#        ]
#      }
#    """
#    if not fragment_payload:
#        raise ValueError("No fragment payloads were provided")
#    if num_fragments_to_sample < 1:
#        raise ValueError("num_fragments_to_sample must be >= 1")
#
#    pocket_ids = list(fragment_payload.keys())
#    if pocket_weights is None or len(pocket_weights) == 0:
#        pocket_weights = [1.0 for _ in pocket_ids]
#    if len(pocket_weights) != len(pocket_ids):
#        raise ValueError(
#            f"Expected {len(pocket_ids)} pocket weights, got {len(pocket_weights)}")
#
#    weights_by_pocket = dict(zip(pocket_ids, pocket_weights))
#    best_by_fragment = {}
#    expected_pocket_count = len(fragment_payload)
#
#    for pocket_id, rows in fragment_payload.items():
#        for row in rows:
#            fragment_key = (
#                row.get("open_atom_from"),
#                row.get("open_atom_h"),
#                row["fragname"],
#            )
#            score = float(row["score"])
#            current_best = best_by_fragment.setdefault(fragment_key, {}).get(pocket_id)
#            if current_best is None or score < current_best["score"]:
#                best_by_fragment[fragment_key][pocket_id] = dict(
#                    score=score,
#                    idx=int(row["idx"]),
#                )
#
#    common_fragments = [
#        fragment_key
#        for fragment_key, pocket_rows in best_by_fragment.items()
#        if len(pocket_rows) == expected_pocket_count
#    ]
#    if not common_fragments:
#        raise ValueError("No fragment candidate is available in every pocket")
#
#    weighted_scores_by_fragment = {}
#    scores_by_fragment = {}
#    indices_by_fragment = {}
#    for fragment_key in common_fragments:
#        weighted_score = 0.0
#        scores_by_fragment[fragment_key] = {}
#        indices_by_fragment[fragment_key] = {}
#        for pocket_id, best_row in best_by_fragment[fragment_key].items():
#            weighted_score += weights_by_pocket[pocket_id] * best_row["score"]
#            scores_by_fragment[fragment_key][pocket_id] = best_row["score"]
#            indices_by_fragment[fragment_key][pocket_id] = best_row["idx"]
#        weighted_scores_by_fragment[fragment_key] = weighted_score
#
#    logits = [-weighted_scores_by_fragment[key] for key in common_fragments]
#    max_logit = max(logits)
#    exp_logits = [math.exp(logit - max_logit) for logit in logits]
#    normalizer = sum(exp_logits)
#    probabilities = [value / normalizer for value in exp_logits]
#
#    sampled_fragments = random.choices(
#        common_fragments, weights=probabilities, k=num_fragments_to_sample)
#
#    selected_fragments = []
#    seen = set()
#    probability_by_fragment = dict(zip(common_fragments, probabilities))
#    for fragment_key in sampled_fragments:
#        if fragment_key in seen:
#            continue
#        seen.add(fragment_key)
#        open_atom_from, open_atom_h, fragname = fragment_key
#        selected_fragments.append(dict(
#            fragname=fragname,
#            open_atom_from=open_atom_from,
#            open_atom_h=open_atom_h,
#            score=weighted_scores_by_fragment[fragment_key],
#            probability=probability_by_fragment[fragment_key],
#            scores_by_pocket=scores_by_fragment[fragment_key],
#            indices_by_pocket=indices_by_fragment[fragment_key],
#        ))
#
#    return dict(fragments=selected_fragments)

# lighter-version of select_fragment_across_pockets
def select_fragment_across_pockets(
    fragment_payload,
    step_idx,
    pocket_weights=None,
    num_fragments_to_sample=1,
    cutoff_ratio=0.2,
):
    """
    Memory-friendlier version:
    - uses integer fragment IDs internally
    - uses pocket indices internally
    - reconstructs tuple/string fields only for selected fragments
    - pre-filters with (1 - cutoff_ratio) * lowest + cutoff_ratio * highest
    """
    if not fragment_payload:
        raise ValueError("No fragment payloads were provided")
    if num_fragments_to_sample < 1:
        raise ValueError("num_fragments_to_sample must be >= 1")
    cutoff_ratio = float(cutoff_ratio)

    pocket_ids = list(fragment_payload.keys())
    num_pockets = len(pocket_ids)

    if pocket_weights is None or len(pocket_weights) == 0:
        pocket_weights = [1.0] * num_pockets
    if len(pocket_weights) != num_pockets:
        raise ValueError(
            f"Expected {num_pockets} pocket weights, got {len(pocket_weights)}"
        )

    pocket_to_idx = {pocket_id: i for i, pocket_id in enumerate(pocket_ids)}
    pocket_weights = [float(w) for w in pocket_weights]

    # fragment_key -> frag_id
    frag_to_id = {}
    # frag_id -> (open_atom_from, open_atom_h, fragname)
    frag_meta = []

    # Compact per-fragment storage
    best_scores = []  # frag_id -> [best score per pocket]
    best_indices = []  # frag_id -> [best idx per pocket]
    seen_pocket_count = []  # frag_id -> number of pockets observed

    # Pass 1: keep only the best dihedral per fragment-pocket pair
    for pocket_id, rows in fragment_payload.items():
        pocket_idx = pocket_to_idx[pocket_id]

        for row in rows:
            fragment_key = (
                row.get("open_atom_from"),
                row.get("open_atom_h"),
                row["fragname"],
            )

            frag_id = frag_to_id.get(fragment_key)
            if frag_id is None:
                frag_id = len(frag_meta)
                frag_to_id[fragment_key] = frag_id
                frag_meta.append(fragment_key)
                best_scores.append([math.inf] * num_pockets)
                best_indices.append([-1] * num_pockets)
                seen_pocket_count.append(0)

            score = float(row["score"])
            if score < best_scores[frag_id][pocket_idx]:
                if best_scores[frag_id][pocket_idx] == math.inf:
                    seen_pocket_count[frag_id] += 1
                best_scores[frag_id][pocket_idx] = score
                best_indices[frag_id][pocket_idx] = int(row["idx"])

    common_frag_ids = [
        frag_id
        for frag_id, count in enumerate(seen_pocket_count)
        if count == num_pockets
    ]
    if not common_frag_ids:
        raise BranchExhaustedError(
            "No fragment candidate is available in every pocket")

    # Step 1: pre-filter fragments using a positive-weight-only score summary.
    # Pockets with zero/negative weights are ignored for this thresholding step.
    positive_pocket_weights = [max(0.0, weight) for weight in pocket_weights]
    positive_weighted_scores = []
    for frag_id in common_frag_ids:
        score_vec = best_scores[frag_id]
        positive_weighted_score = 0.0
        for pocket_idx, weight in enumerate(positive_pocket_weights):
            positive_weighted_score += weight * score_vec[pocket_idx]
        positive_weighted_scores.append(positive_weighted_score)

    lowest_fragment_score = min(positive_weighted_scores)
    highest_fragment_score = max(positive_weighted_scores)
    fragment_score_cutoff = (
        (1.0 - cutoff_ratio) * lowest_fragment_score
        + cutoff_ratio * highest_fragment_score
    )
    common_frag_ids = [
        frag_id
        for frag_id, positive_weighted_score in zip(
            common_frag_ids, positive_weighted_scores
        )
        if positive_weighted_score <= fragment_score_cutoff
    ]
    # Step 2: weighted score for each fragment that passed the pre-filter.
    weighted_scores = []
    for frag_id in common_frag_ids:
        score_vec = best_scores[frag_id]
        weighted_score = 0.0
        for pocket_idx, weight in enumerate(pocket_weights):
            weighted_score += weight * score_vec[pocket_idx]
        weighted_scores.append(weighted_score)

    # Softmax over negated weighted scores
    logits = [-score for score in weighted_scores]
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    normalizer = sum(exp_logits)
    probabilities = [value / normalizer for value in exp_logits]

    # Sample with replacement using compact IDs
    sampled_frag_ids = random.choices(
        common_frag_ids,
        weights=probabilities,
        k=num_fragments_to_sample,
    )

    # Fast lookup from frag_id -> position in common_frag_ids
    pos_by_frag_id = {
        frag_id: pos
        for pos, frag_id in enumerate(common_frag_ids)
    }

    selected_fragments = []
    seen = set()

    for frag_id in sampled_frag_ids:
        if frag_id in seen:
            continue
        seen.add(frag_id)

        pos = pos_by_frag_id[frag_id]
        open_atom_from, open_atom_h, fragname = frag_meta[frag_id]

        selected_fragments.append(
            dict(
                fragname=fragname,
                open_atom_from=open_atom_from,
                open_atom_h=open_atom_h,
                score=weighted_scores[pos],
                probability=probabilities[pos],
                scores_by_pocket={
                    pocket_ids[i]: best_scores[frag_id][i]
                    for i in range(num_pockets)
                },
                indices_by_pocket={
                    pocket_ids[i]: best_indices[frag_id][i]
                    for i in range(num_pockets)
                },
            )
        )

    return dict(fragments=selected_fragments)

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


def save_branch_ligands(output_root, branch, status_suffix="final"):
    """Save the current ligand for every pocket in a branch."""
    for pocket_id, state in branch["states"].items():
        out_file = os.path.join(
            output_root, f"{branch['branch_id']}_{pocket_id}_{status_suffix}.mae")
        write_mae(out_file, [state["node"].ligand])


def save_stopped_branch(output_root, branch, step, reason):
    """
    Save a branch that cannot be expanded further and record why it stopped.

    The branch contains ligands after the previous completed growth step.  For
    example, if expansion fails while attempting step 5, the saved ligands are
    the depth-4 solution for that branch.
    """
    stopped_depth = max(step - 1, 0)
    status_suffix = f"stopped_d{stopped_depth}_final"
    save_branch_ligands(output_root, branch, status_suffix=status_suffix)
    dump_json(
        os.path.join(
            output_root,
            f"{branch['branch_id']}_stopped_d{stopped_depth}_reason.json"),
        dict(
            branch_id=branch["branch_id"],
            stopped_at_attempted_step=int(step),
            saved_depth=int(stopped_depth),
            reason=str(reason),
        ),
    )


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
    selected_open_bonds = selected_open.get("open_bonds", [selected_open])

    fragment_candidates = []
    candidate_open_bonds = []
    for selected_bond in selected_open_bonds:
        open_bond = resolve_open_bond_by_atoms(
            adder, parent_node, selected_bond["atom_from"], selected_bond["atom_h"]
        )
        if open_bond is None:
            continue

        raw_fragments = adder.sample_fragments(
            parent_node, open_bond, adder.fragname_list)
        raw_fragments = [n for n in raw_fragments if adder.reaction_filter(n)]
        raw_fragments = adder.filter_top(
            raw_fragments, adder.max_fragments, adder.fragment_scorefxn
        )

        for new_node in raw_fragments:
            candidate_dihedrals = adder.sample_dihedrals(new_node, parent_node)
            candidate_dihedrals = adder.filter_top(
                candidate_dihedrals, adder.max_dihedrals, adder.dihedral_scorefxn
            )
            fragment_candidates.extend(candidate_dihedrals)
            candidate_open_bonds.extend([selected_bond for _ in candidate_dihedrals])

    if len(fragment_candidates) == 0:
        return pocket_id, [], []

    scores = adder.heuristic_timed(fragment_candidates, adder.goal)
    rows = []
    for idx, (node, score, selected_bond) in enumerate(
            zip(fragment_candidates, scores, candidate_open_bonds)):
        node.score = float(score)
        node.parent = parent_node
        rows.append(
            dict(
                idx=idx,
                score=float(score),
                fragname=node.fragname,
                attachment_atom=node.get_attachment_atom_original_id(),
                open_atom_from=selected_bond["atom_from"],
                open_atom_h=selected_bond["atom_h"],
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
    # Multi-pocket runs emit stage-specific JSON/MAE outputs for each pocket,
    # so do not create the default adder-level debug.log file.
    config["advanced_config"]["log_to_file"] = False

    output_root = args.output_folder_path
    os.makedirs(output_root, exist_ok=True)

    pocket_weights = parse_csv_floats(args.pocket_weights)
    random.seed(args.random_seed)

    initial_states = {}
    for i, (seed_path, pocket_path) in enumerate(zip(seed_paths, pocket_paths)):
        pocket_id = f"pk{i}"
        seed = read_mae(seed_path)[0]
        pocket = read_mae(pocket_path)[0]
        adder = initialize_adder(config.copy(), args.e3nn_env_path)
        adder.goal = {"type": "depth", "value": args.max_steps}
        adder.debug_config["debug_output_root"] = os.path.join(output_root, pocket_id) + "/"
        os.makedirs(adder.debug_config["debug_output_root"], exist_ok=True)
        initial_states[pocket_id] = dict(adder=adder, node=LigandNode(pocket, seed))

    validate_ligand_atom_index_consistency(
        initial_states,
        context="initial multi-pocket seed ligands",
    )

    branches = [dict(branch_id="b0", states=initial_states)]
    for step in range(1, args.max_steps + 1):
        next_branches = []
        for branch in branches:
            branch_id = branch["branch_id"]
            states = branch["states"]
            with ThreadPoolExecutor(max_workers=len(states)) as executor:
                jobs = [
                    (pid, s["adder"], s["node"])
                    for pid, s in states.items()
                ]
                open_payload = dict(executor.map(score_open_bonds_for_pocket, jobs))

            dump_json(
                os.path.join(output_root, f"d{step}_{branch_id}_open_bonds_all_pockets.json"),
                open_payload)
            try:
                selected_open = select_open_bond_across_pockets(
                    open_payload, step, pocket_weights, args.num_open_bonds_to_sample)
            except BranchExhaustedError as error:
                save_stopped_branch(output_root, branch, step, error)
                continue
            dump_json(
                os.path.join(output_root, f"d{step}_{branch_id}_selected_open_bonds.json"),
                selected_open)

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
                    mae_path = os.path.join(
                        output_root, f"d{step}_{branch_id}_{pocket_id}_fragment_candidates.mae")
                    dump_fragment_mae(mae_path, candidates)

            dump_json(
                os.path.join(output_root, f"d{step}_{branch_id}_fragment_candidates_all_pockets.json"),
                fragment_payload)
            try:
                selected_frag = select_fragment_across_pockets(
                    fragment_payload, step, pocket_weights, args.num_fragments_to_sample)
            except BranchExhaustedError as error:
                save_stopped_branch(output_root, branch, step, error)
                continue
            dump_json(
                os.path.join(output_root, f"d{step}_{branch_id}_selected_fragments.json"),
                selected_frag)

            for selection_idx, selected in enumerate(selected_frag["fragments"]):
                next_states = {}
                next_branch_id = f"{branch_id}_s{step}_{selection_idx}"
                for pocket_id, sel_idx in selected["indices_by_pocket"].items():
                    candidates = fragment_nodes[pocket_id]
                    if sel_idx < 0 or sel_idx >= len(candidates):
                        raise ValueError(f"Invalid fragment index {sel_idx} for {pocket_id}")
                    out_file = os.path.join(
                        output_root,
                        f"d{step}_{next_branch_id}_{pocket_id}_selected_fragment.mae")
                    write_mae(out_file, [candidates[sel_idx].ligand])
                    next_states[pocket_id] = dict(
                        adder=states[pocket_id]["adder"],
                        node=candidates[sel_idx],
                    )
                validate_ligand_atom_index_consistency(
                    next_states,
                    context=f"step {step} branch {next_branch_id}",
                )
                next_branches.append(dict(branch_id=next_branch_id, states=next_states))
        branches = next_branches
        if not branches:
            break

    # Save final grown ligands for each pocket.
    for branch in branches:
        save_branch_ligands(output_root, branch)


def get_args():
    parser = argparse.ArgumentParser(description="Run multi-pocket FRAME growth")
    parser.add_argument("--config_name", choices=["config_random", "config_ML"], default="config_ML")
    parser.add_argument("--seed_ligand_paths", type=str, required=True,
                        help="Comma-separated list of seed ligand MAE paths (one per pocket).")
    parser.add_argument("--protein_pocket_paths", type=str, required=True,
                        help="Comma-separated list of pocket MAE paths.")
    parser.add_argument("--output_folder_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--pocket_weights", type=str, default="",
                        help="Comma-separated pocket weights. Defaults to equal weights.")
    parser.add_argument("--num_open_bonds_to_sample", type=int, default=1,
                        help="Number of open-bond samples to draw from the "
                             "weighted-score softmax before de-duplication.")
    parser.add_argument("--num_fragments_to_sample", type=int, default=1,
                        help="Number of fragment samples to draw from the "
                             "weighted-score softmax before de-duplication.")
    parser.add_argument("--random_seed", type=int, default=10,
                        help="Random seed for softmax open-bond sampling.")
    parser.add_argument("--e3nn_env_path", type=str,
                        default="/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/software/anaconda3/envs/e3nn/lib/python3.8/site-packages")
    return parser.parse_args()

'''
$SCHRODINGER/run python3 -m src.frag_adder.run_multi_pk --config_name config_ML --output_folder_path ./test_outputs --seed_ligand_path ./data/test_inputs/3C49_seed_ligand.mae --protein_pocket_path ./data/test_inputs/3C49_pocket.mae --end_point number_steps --max_steps 5
'''


def main():
    args = get_args()
    run_multi_pocket(args)


if __name__ == "__main__":
    main()
