#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --job-name=ago_rar

set -euo pipefail
module load schrodinger/2024
export PYTHONPATH="/public/home/xinyu/FRAME:/public/home/xinyu/.conda/envs/frame/lib/python3.8/site-packages:${PYTHONPATH:-}"

OUTPUT_DIR="${SLURM_SUBMIT_DIR:-$PWD}/outputs"
cd /public/home/xinyu/FRAME

/public/software/schrodinger2024-1/run python3 -m src.frag_adder.run_multi_pk \
  --config_name config_ML \
  --output_folder_path "$OUTPUT_DIR" \
  --seed_ligand_paths /public/home/xinyu/RAR/input/3KMZ_ligand.mae,/public/home/xinyu/RAR/input/3KMR_ligand.mae \
  --protein_pocket_paths /public/home/xinyu/RAR/input/3KMZ_pocket.mae,/public/home/xinyu/RAR/input/3KMR_pocket.mae \
  --pocket_weights=-1,1 \
  --max_steps 5 \
  --num_open_bonds_to_sample 2 \
  --num_fragments_to_sample 20 \
  --random_seed 10 \
  --e3nn_env_path /public/home/xinyu/.conda/envs/frame/lib/python3.8/site-packages
