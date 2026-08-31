# Toy Experiments: Spurious Correlations

This directory contains the two toy experiments of the paper, which probe
shortcut learning under controlled spurious correlations of strength
`alpha` (`--spu_prob`).

## The two experiments

### 1. Label concatenation (`spu_*`)
A **cue channel** is appended to the input features of every training sample
(`spurious_corr.add_spruious`): with probability `alpha` it is the one-hot
**true label**, otherwise a random label. Copying the cue therefore yields
training accuracy `~alpha` without learning the task rule.

* The per-step validation batch (`v_a` in the logs) and the final length-range
  evaluation instead carry an **uncorrelated random cue**
  (`spurious_corr.add_random`), so they measure true-rule generalization.
* For multi-output tasks the cue spans the last `output_length` time steps of
  the sequence (cue placed at the **end** of the input for recurrent models).

### 2. Input-length confounding (`spu_len_*`)
Training batches are **rejection-sampled** (`spu_len_input.add_spruious`) so
that, with probability `alpha`, the label equals a length-dependent
pseudo-label (the input length `1..40` binned into `output_size` classes).
Inputs are unmodified; only the length-label joint distribution is skewed.
Validation and the final range evaluation use the standard (uncorrelated)
samplers. Because the pseudo-label is a single class, this experiment is only
defined for **single-output** tasks (`task.output_length == 1`).

## Supported task-setting combinations

Only the combinations that were actually run for the paper are enabled
(enforced by the parsers):

| | Label concatenation | Length confounding |
|---|---|---|
| Tasks | 14 main tasks (all except `binary_multiplication`) | 7 single-output tasks: `modular_arithmetic`, `parity_check`, `even_pairs`, `cycle_navigation`, `modular_arithmetic_brackets`, `missing_duplicate_string`, `solve_equation` |
| Architectures | `rnn`, `lstm`, `stack_rnn`, `tape_rnn` | same |
| Methods | Adam (`none`), BGF (`ours_balance`), ADD (`ours_add`) | same |

## Files

| File | Role |
|---|---|
| `spurious_corr.py` | cue-channel helpers for label concatenation |
| `spu_len_input.py` | rejection sampler for length confounding |
| `spu_training.py` / `spu_len_training.py` | training loops (Adam / BGF / ADD) |
| `spu_range_evaluation.py` | length-range evaluation with a random cue channel |
| `spu_main.py` / `spu_len_main.py` | experiment setup (model, task, logging) |
| `spu_parser.py` / `spu_len_parser.py` | CLI for a **single** run |
| `paper_toy_configs.json` | every configuration actually run for the paper |
| `run_toy.py` | replays `paper_toy_configs.json` (the full paper grid) |

## Reproducing the paper results

`paper_toy_configs.json` lists, for each method x task x architecture, the
learning rate, BGF/ADD weights and the spurious strengths used in the paper
(the best hyperparameters from the main experiments). `run_toy.py` replays it:

```bash
# list all runs (with indices) without executing anything
python run_toy.py --experiment label --dry_run

# label-concatenation grid, 3 seeds each (4176 runs) on GPU 0
python run_toy.py --experiment label --cuda 0

# length-confounding grid (2520 runs), split over two GPUs
python run_toy.py --experiment length --cuda 0 --start 1    --end 1261
python run_toy.py --experiment length --cuda 1 --start 1261 --end 2521

# subsets
python run_toy.py --experiment label --methods bgf add --tasks parity_check bucket_sort
python run_toy.py --experiment label --spu_probs 0.9 0.99 0.995
```

A single custom run (any supported combination, also outside the paper grid
of learning rates / weights / strengths):

```bash
python spu_parser.py --cuda 0 --task parity_check --architecture lstm \
    --optim ours_balance --weight_a 0.7 --weight_b 0.3 --queue_size 100 \
    --lr 5e-4 --spu_prob 0.95 --training_steps 100000 \
    --folder_name ours_balance-0.0005-0.7-0.3-p0.95-100000

python spu_len_parser.py --cuda 0 --task parity_check --architecture rnn \
    --optim none --lr 5e-4 --spu_prob 0.9 --training_steps 100000 \
    --folder_name baseline-0.0005-p0.9-100000
```

## Output format

Each run writes
`<save_dir>/<folder_name>/<architecture>/<task>/seed<N>_<timestamp>/logs_spu`
(or `logs_spu_len`), a JSON file with:

* `step_log`: per-step `{t_l, t_a, v_l, v_a}` = train loss/accuracy (spurious
  data; can be inflated by cue copying) and validation loss/accuracy
  (uncorrelated data; measures the true rule),
* `range_eval`: final length-range evaluation `{length: {final_acc}}`
  for lengths `1..100`,
* `setting`: all hyperparameters of the run.

Note: the spurious strength is embedded in the folder name (`-p<alpha>-`), so
sweeps over `alpha` stay separated on disk.
