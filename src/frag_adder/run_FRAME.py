import sys
import os
import argparse

from src.frag_adder.adder_random import initialize_random_adder
from src.frag_adder.configs.config import get_config

from src.utils.config_tools import write_config
from src.utils.struc_tools import write_mae, read_mae


def run_FRAME(args, config):

    #set up output folder
    output_folder = args.output_folder_path
    config["output_root_folder"] = output_folder
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, 'steps.mae')
    if os.path.exists(output_filename):
        print('Output file already exists, terminating')
        return

    #Setup logging
    config['advanced_config']['debug_output_root'] = output_folder
    config['advanced_config']['log_file'] = os.path.join(output_folder, 'experiment.log')
    config['advanced_config']['save_scored_candidates'] = args.save_scored_candidates
    config['advanced_config']['save_open_bond_candidates'] = args.save_open_bond_candidates
    config['advanced_config']['save_candidate_scores_json'] = args.save_candidate_scores_json
    config['stop_after_open_bond_stage'] = args.stop_after_open_bond_stage
    config['stop_after_fragment_stage'] = args.stop_after_fragment_stage
    config['selected_fragment_index'] = args.selected_fragment_index
    config['selected_open_atom_from'] = args.selected_open_atom_from
    config['selected_open_atom_h'] = args.selected_open_atom_h
    config['fragment_candidates_mae_path'] = args.fragment_candidates_mae_path

    #Setup other configuration options
    adder_type = config["adder_type"]
    config["goal_type"] = args.end_point
    config['max_depth'] = args.max_steps
    config['beam_width'] = args.beam_width
    config['search_strategy'] = args.search_strategy

    # Write the fully resolved runtime config after all CLI overrides have been applied.
    write_config(config, os.path.join(output_folder, 'experiment.log'))

    #Load input files
    try:
        seed_path = args.resume_ligand_path if args.resume_ligand_path else args.seed_ligand_path
        seed = read_mae(seed_path)[0]
        pocket = read_mae(args.protein_pocket_path)[0]
    except:
        print("Problem loading input files, terminating")
        return

    if args.end_point in ['ref_heavy', 'ref_mw']:
        end_point_ligand = read_mae(args.endpoint_ligand_path)[0]
    else:
        end_point_ligand = None

    if adder_type == "ML_2model":
        # need to access e3nn and torch libraries from SCHRODINGER python, this is a simple way to do it os.getenv('E3NN_PATH')
        sys.path.insert(0, args.e3nn_env_path)
        from src.frag_adder.adder_2model import initialize_2model_adder
        adder = initialize_2model_adder(config)
    if adder_type == 'random':
        adder = initialize_random_adder(config)

    solution = adder.run(seed, pocket, output_filename, endpoint_struc=end_point_ligand, goal=config["goal_type"])
    adder.logger.handlers.clear()

def get_args():
    parser = argparse.ArgumentParser(description='Job Runner')

    parser.add_argument('--config_name', choices=['config_random', 'config_ML'], type=str, default='config_random', help='Most of options for FRAME are specified in configs, see src/frag_adder/configs')
    parser.add_argument('--output_folder_path', type=str, help='Folder to output results, will create folder if it does not exist')
    parser.add_argument('--seed_ligand_path', type=str, help='The starting ligand .mae file, must be aligned with pocket')
    parser.add_argument('--resume_ligand_path', type=str, default='',
                        help='Optional restart ligand path selected by human from previous candidates. If provided, this overrides --seed_ligand_path.')
    parser.add_argument('--protein_pocket_path', type=str, help='The protein pocket .mae file, recommended to select ~5-7 A around ligand')

    parser.add_argument('--end_point', choices=['number_steps', 'ref_heavy', 'ref_mw'], default='number_steps',
                        help='Options for when to terminate adding fragments, ref_heavy and ref_mw use provided reference ligand (--endpoint_ligand_path) to determine maximum number of heavy atoms or molecular weight')
    parser.add_argument('--max_steps', type=int, default=5, help='If end point is number_steps, maximum number of fragments to add')
    parser.add_argument('--search_strategy', choices=['greedy', 'beam'], default='greedy',
                        help='Search strategy across depths. greedy keeps one branch; beam keeps top-k branches.')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='Beam size for beam search. Ignored when search_strategy=greedy.')
    parser.add_argument('--endpoint_ligand_path', type=str, default='', help='If end point is ref_heavy or ref_mw, path to reference .mae file for determine number of fragments to add')
    parser.add_argument('--save_scored_candidates', action='store_true',
                        help='Save scored final candidates at each depth as MAE.')
    parser.add_argument('--save_open_bond_candidates', action='store_true',
                        help='Save open-bond candidate structures at each depth as MAE.')
    parser.add_argument('--save_candidate_scores_json', action='store_true',
                        help='Save scored final candidates at each depth as JSON for human-in-the-loop selection.')
    parser.add_argument('--stop_after_open_bond_stage', action='store_true',
                        help='Stop after writing open-bond candidates/scores for the current ligand.')
    parser.add_argument('--stop_after_fragment_stage', action='store_true',
                        help='Stop after writing fragment candidates/scores for a selected open bond.')
    parser.add_argument('--selected_fragment_index', type=int, default=None,
                        help='Index of fragment candidate to continue from (after fragment stage output).')
    parser.add_argument('--fragment_candidates_mae_path', type=str, default='',
                        help='Optional MAE file generated in stage-2 (e.g. d1_fragment_candidates.mae). '
                             'If provided with --selected_fragment_index, stage-3 selects directly from this file without re-sampling fragments.')
    parser.add_argument('--selected_open_atom_from', type=int, default=None,
                        help='Optional atom index (heavy atom) to identify selected open bond directly during restart.')
    parser.add_argument('--selected_open_atom_h', type=int, default=None,
                        help='Optional atom index (hydrogen atom) to identify selected open bond directly during restart.')

    parser.add_argument('--e3nn_env_path', type=str, default='/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/software/anaconda3/envs/e3nn/lib/python3.8/site-packages', help='To allow using e3nn and custom torch with schrodinger python environment')
    args = parser.parse_args()
    return args

'''
$SCHRODINGER/run python3 -m src.frag_adder.run_FRAME --config_name config_ML --output_folder_path ./test_outputs --seed_ligand_path ./data/test_inputs/3C49_seed_ligand.mae --protein_pocket_path ./data/test_inputs/3C49_pocket.mae --end_point number_steps --max_steps 5
'''
def main():
    command_line_args = get_args()
    config = get_config(command_line_args.config_name)
    run_FRAME(command_line_args, config)

if __name__ == "__main__":
    main()
