# Long-Range Generalization Evaluation

Standalone evaluation of trained checkpoints on sequence lengths **1 to 1000**
(the paper's long-range generalization figure; default task:
`missing_duplicate_string`, MD). It is independent of the training pipeline
and of the standard end-of-training evaluation over lengths 1-100 — it only
requires finished run directories with saved checkpoints.

## Usage

1. Train with checkpointing enabled (`--save_param 1`), so each run stores its
   final parameters in `<run_dir>/params/param_s<training_steps>.pt`:

   ```bash
   cd ../run_experiments
   python run_adam.py --cuda 0 --tasks missing_duplicate_string --save_param 1
   python run_bgf.py  --cuda 0 --tasks missing_duplicate_string --save_param 1
   ```

2. Evaluate the checkpoints:

   ```bash
   cd ../run_long_eval
   python long_eval.py --save_dir ../run_experiments/results --cuda 0   # all MD runs
   python long_eval.py --save_dir ../run_experiments/results --dry_run  # list only
   python long_eval.py --run_dir <path to one seedN_... run directory>
   ```

Each evaluated run gets a `long_range_eval_s<step>.json` in its run directory
with `accuracy_per_length` (index `l-1` = accuracy at length `l`), evaluated on
`--total_batch_size` (default 512) fresh samples per length with a fixed
evaluation seed — the same protocol as the standard range evaluation, extended
to `--max_length` (default 1000).

Useful options: `--tasks/--architectures/--folders` to filter runs,
`--start/--end` to split the work across GPUs, `--overwrite` to re-evaluate,
`--checkpoint_step` if runs were trained with a different `--training_steps`.

Note: 1000 lengths means 1000 separate jit compilations per run — a full run
takes a while; use `--start/--end` to parallelize across GPUs.
