# FRAME examples

Run commands from the FRAME repository root so relative model and fragment-library
paths resolve. Install the dependencies and download model weights as described
in the [main README](../README.md). ML configurations expect the two checkpoint
files named in `src/frag_adder/configs/config.py` under `data/source/`.

## HPC RAR jobs

These examples are adapted from all four `basic.slurm` files read from
`/public/home/xinyu/RAR/lib/{ago,anta,multi_ago,multi_anta}` on the HPC.
The labels follow those jobs; they describe the intended scoring preference,
not experimentally established activity.

| Example | Pocket order | Weights | Purpose |
| --- | --- | --- | --- |
| [ago](rar/ago/basic.slurm) | 3KMR | `1` | Grow against 3KMR alone. |
| [anta](rar/anta/basic.slurm) | 3KMZ | `1` | Grow against 3KMZ alone. |
| [multi_ago](rar/multi_ago/basic.slurm) | 3KMZ, 3KMR | `-1,1` | Favor lower scores in 3KMR and higher scores in 3KMZ. |
| [multi_anta](rar/multi_anta/basic.slurm) | 3KMZ, 3KMR | `1,-1` | Favor lower scores in 3KMZ and higher scores in 3KMR. |

All four use `run_multi_pk`, including the one-pocket jobs. They retain the
original five growth steps, two open-bond draws, twenty fragment draws, seed 10,
48-hour limit, and Schrödinger 2024 module. The examples quote paths, stop on
command errors, use the dependency environment's `site-packages` for
`--e3nn_env_path`, and request two CPUs for the two-pocket jobs rather than one.
They do not request a GPU; add your cluster's GPU/partition directives if needed.
The runner reports its chosen device for each pocket.

The scripts use these existing HPC paths; edit them for another installation:

- FRAME checkout: `/public/home/xinyu/FRAME`.
- Dependencies: `/public/home/xinyu/.conda/envs/frame/lib/python3.8/site-packages`.
- Schrödinger launcher: `/public/software/schrodinger2024-1/run`.
- Input structures: `/public/home/xinyu/RAR/input/{3KMR,3KMZ}_{ligand,pocket}.mae`.

The RAR structures are not bundled here. The paths above were checked on the
HPC. Update that machine's FRAME checkout to this fork's current code before
running the new examples; the local checkout and HPC checkout are separate.

Submit from the example's directory, so each job gets its own `outputs/` folder:

```bash
cd /public/home/xinyu/FRAME/examples/rar/multi_anta
sbatch basic.slurm
```

Use a fresh submission directory for another run to avoid overwriting outputs.
Submitting a job can consume cluster resources; these example files have not
been submitted as part of preparing this documentation.

## How multi-pocket selection works

Provide equally long, ordered, comma-separated lists of seed ligand and pocket
MAE files. `pk0` is the first pair, `pk1` the second, and so on. Each seed ligand
must already be aligned to its pocket. Ligands must have the same atom indices,
elements, and bonded-neighbor identities across pockets; their coordinates can
differ. The runner validates this at startup, after growth, and on restart.

FRAME treats lower scores as better. At each step:

1. Find attachment bonds available in every pocket. Sum their scores using only
   positive pocket weights, then sample with probabilities proportional to
   `exp(-weighted_score)`.
2. Generate and score fragment conformations for the selected bonds in each
   pocket. Match candidates by heavy-atom index, hydrogen index, and fragment
   name, retaining the lowest-scoring dihedral for each fragment in each pocket.
   A fragment must be available in every pocket, even one with zero weight.
3. Prefilter using positive weights: keep fragments with scores at or below
   `0.8 * lowest_score + 0.2 * highest_score`. This is a score-range threshold,
   not the best 20% of candidates. The function accepts `cutoff_ratio`; there is
   currently no CLI flag for it.
4. Apply the full signed pocket weights to the surviving fragment scores and
   sample with the same negated-score softmax. Each distinct sampled fragment
   becomes a new branch, with its own best dihedral in each pocket.

Both sampling counts are draws **with replacement**, followed by deduplication;
they do not guarantee that many distinct results. Multi-pocket branching has
no global beam-width limit, so twenty fragment draws per branch can produce
many trajectories over five steps. Negative weights influence final fragment
sampling; they do not penalize open-bond selection or the prefilter. Weight
magnitudes also affect how concentrated the softmax probabilities are.

## Beam search with the bundled single-pocket input

In a shell where Schrödinger and the model dependencies are available:

```bash
export SCHRODINGER=/public/software/schrodinger2024-1
export E3NN_SITE=/public/home/xinyu/.conda/envs/frame/lib/python3.8/site-packages
export PYTHONPATH="$PWD:$E3NN_SITE:${PYTHONPATH:-}"

"$SCHRODINGER/run" python3 -m src.frag_adder.run_FRAME \
  --config_name config_ML \
  --seed_ligand_path data/test_inputs/3C49_seed_ligand.mae \
  --protein_pocket_path data/test_inputs/3C49_pocket.mae \
  --output_folder_path runs/beam/ \
  --end_point number_steps --max_steps 5 \
  --search_strategy beam --beam_width 3 \
  --save_scored_candidates --save_candidate_scores_json \
  --e3nn_env_path "$E3NN_SITE"
```

Beam search keeps the best three candidates across expansions at each depth.
It uses the current candidate's heuristic score, not a cumulative path score.
`steps.mae` includes the protein, seed, and retained nodes from each depth;
it can contain multiple branches. Greedy search keeps one path and remains the
default. Use a trailing `/` for single-pocket output folders because candidate
filenames are built by concatenating the folder and filename.

## Manual bond and fragment selection

Use the same shell environment as above. Each command intentionally performs
one stage. Start with a fresh directory for each stage.

```bash
COMMON=(--config_name config_ML
  --seed_ligand_path data/test_inputs/3C49_seed_ligand.mae
  --protein_pocket_path data/test_inputs/3C49_pocket.mae
  --end_point number_steps --max_steps 1
  --save_candidate_scores_json --e3nn_env_path "$E3NN_SITE")

# Stage 1: score attachment bonds and stop.
"$SCHRODINGER/run" python3 -m src.frag_adder.run_FRAME "${COMMON[@]}" \
  --output_folder_path runs/manual/bonds/ \
  --stop_after_open_bond_stage
```

Read `runs/manual/bonds/d1_open_bonds.json` and choose its `atom_from` and
`atom_h` values. Set `ATOM_FROM` and `ATOM_H` to those actual integer values,
then run:

```bash
# Stage 2: score fragment/conformation candidates for your chosen bond.
"$SCHRODINGER/run" python3 -m src.frag_adder.run_FRAME "${COMMON[@]}" \
  --output_folder_path runs/manual/fragments/ \
  --selected_open_atom_from "$ATOM_FROM" --selected_open_atom_h "$ATOM_H" \
  --stop_after_fragment_stage

# Stage 3: choose zero-based candidate 0 from the saved stage-2 MAE file.
"$SCHRODINGER/run" python3 -m src.frag_adder.run_FRAME "${COMMON[@]}" \
  --output_folder_path runs/manual/selected/ \
  --fragment_candidates_mae_path runs/manual/fragments/d1_fragment_candidates.mae \
  --selected_fragment_index 0
```

Choose an `idx` from `d1_fragment_candidates.json` and inspect the corresponding
structure in `d1_fragment_candidates.mae`; replace `0` with that index.
Stage 3 reads that saved structure directly, avoiding fragment resampling.
`--save_candidate_scores_json` is needed to export the JSON and stage-2 MAE
files used here. Stages 1 and 2 normally return only the seed and do not write
`steps.mae`; stage 3 writes the protein, seed, and selected candidate there.
To start further growth, export the selected ligand alone to a MAE file and
pass it through `--resume_ligand_path`; this replaces the seed input and starts
a new run. Do not pass a protein-containing `steps.mae` as the restart ligand.

## Checkpoint restart

For a completed depth in a multi-pocket run, keep its JSON manifest and all
referenced checkpoint MAE files together. For example, extend `multi_anta` to
a total of eight steps, using its original pocket order and weights:

```bash
"$SCHRODINGER/run" python3 -m src.frag_adder.run_multi_pk \
  --config_name config_ML \
  --seed_ligand_paths /public/home/xinyu/RAR/input/3KMZ_ligand.mae,/public/home/xinyu/RAR/input/3KMR_ligand.mae \
  --protein_pocket_paths /public/home/xinyu/RAR/input/3KMZ_pocket.mae,/public/home/xinyu/RAR/input/3KMR_pocket.mae \
  --pocket_weights=1,-1 \
  --num_open_bonds_to_sample 2 --num_fragments_to_sample 20 \
  --random_seed 10 --max_steps 8 \
  --resume_checkpoint examples/rar/multi_anta/outputs/checkpoint_latest.json \
  --output_folder_path runs/multi_anta_resumed/ \
  --e3nn_env_path "$E3NN_SITE"
```

`--max_steps` is the total target depth, not an additional step count.
The seed paths remain required by the CLI; checkpoint ligand files supply the
actual restart structures. Reuse the original pocket order, weights, model
configuration, and sampling options: the manifest does not store or restore a
complete run configuration. Python's saved random state is restored after all
adders are initialized; this does not guarantee bitwise reproducibility of
threaded or GPU calculations.

## Outputs and diagnostics

| Runner | Files |
| --- | --- |
| Single-pocket | `steps.mae` and `experiment.log`; the log starts with the configuration after CLI overrides. Greedy search also writes `intermediate.mae`. |
| Single-pocket candidate export | `--save_scored_candidates` exports scored MAE files; `--save_open_bond_candidates` exports open-bond structures; `--save_candidate_scores_json` exports JSON scores and manual-stage candidate MAE files. |
| Multi-pocket final ligands | `<branch_id>_<pocket_id>_final.mae`, one current ligand for each surviving branch/pocket. |
| Multi-pocket checkpoints | `checkpoint_d<N>.json`, corresponding `checkpoint_d<N>_<branch_id>_<pocket_id>.mae` files, and `checkpoint_latest.json`. Initial depth 0 is also saved. |
| Exhausted branches | `<branch_id>_<pocket_id>_stopped_d<N>_final.mae` and `<branch_id>_stopped_d<N>_reason.json`; N is the last completed depth. |
| Multi-pocket debug | Add `--debug_outputs` for intermediate JSON/MAE candidates, selections, and per-pocket `debug.log` files. These exports are off by default; final ligands and checkpoints remain enabled. |

Checkpoints cover completed depths only. If a job stops during expansion, resume
from its last completed checkpoint. There is no automatic restart of a partially
written checkpoint. The examples were checked for shell syntax and CLI argument
compatibility; full molecular generation still needs the HPC runtime and model
files.
