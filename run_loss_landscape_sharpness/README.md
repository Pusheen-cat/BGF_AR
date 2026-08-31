# Loss-Landscape Sharpness Analysis

Computes the paper's three sharpness measures on trained Adam and BGF
checkpoints and summarises them by Chomsky hierarchy level with paired
t-tests.  The analysis covers the four recurrent models (`rnn`, `lstm`,
`stack_rnn`, `tape_rnn`) and all 15 tasks.

| Measure | File | Definition |
| :--- | :--- | :--- |
| Low-pass-filter-based | `flat_measures.low_pass` | Monte-Carlo estimate of the Gaussian-smoothed loss E[L(θ+ε)], ε ~ N(0, σ²I) (σ = 0.01, 100 samples) |
| FIM-based | `flat_measures.fim` | θᵀHθ at the trained parameters, with the loss Hessian applied via a Hessian-vector product |
| Shannon-based | `flat_measures.shannon_entropy` | Mean Shannon entropy −E[Σ p log p] of the model's output distribution |

## 1. Prerequisite: trained checkpoints

The measures are evaluated on final checkpoints, so the runs must be trained
with `--save_param 1`:

```bash
cd ../run_experiments
python run_adam.py --cuda 0 --architectures rnn lstm stack_rnn tape_rnn --save_param 1
python run_bgf.py  --cuda 0 --architectures rnn lstm stack_rnn tape_rnn --save_param 1
```

## 2. Sharpness evaluation (`sharp_eval.py`)

```bash
python sharp_eval.py --save_dir ../run_experiments/results --cuda 0
python sharp_eval.py --save_dir ../run_experiments/results --dry_run     # list runs
python sharp_eval.py --save_dir ../run_experiments/results --start 1 --end 91 --cuda 0
```

Every run with a `params/param_s<checkpoint_step>.pt` checkpoint gets a
`sharpness.json` with the three measures, each evaluated on the training
distribution (`train`) and on the `--val_len` sequence lengths (default 100).
Monte-Carlo settings (`--sharp_batch 128`, `--sharp_steps 100`,
`--mcmc_itr 100`, `--sigma 0.01`) follow the paper's evaluation.  The
low-pass measure dominates the runtime (`mcmc_itr` × `sharp_steps` loss
evaluations per distribution); reduce `--mcmc_itr`/`--sharp_steps` or the
`--val_len` list for a quicker pass.

## 3. Summary + paired t-tests (`analyze_sharpness.py`)

```bash
python analyze_sharpness.py --save_dir ../run_experiments/results
```

For each measure the script pairs the Adam (`baseline-*`) and BGF
(`ours_balance-*`) results on (task, architecture, seed), **excludes a pair
whenever either member is NaN**, and then — per Chomsky hierarchy level,
averaging over that level's tasks and the four models — reports the mean
Adam value, the mean BGF value and the paired t-test p-value
(`scipy.stats.ttest_rel`) for:

* Adam-Regular vs. BGF-Regular — `modular_arithmetic`, `parity_check`, `even_pairs`, `cycle_navigation`
* Adam-Context-Free vs. BGF-Context-Free — `modular_arithmetic_brackets`, `reverse_string`, `solve_equation`, `stack_manipulation`
* Adam-Context-Sensitive vs. BGF-Context-Sensitive — `binary_addition`, `binary_multiplication`, `bucket_sort`, `compute_sqrt`, `duplicate_string`, `missing_duplicate_string`, `odds_first`

The tables are printed and written to `<save_dir>/sharpness_summary.csv`.
The analysed value defaults to the training-distribution sharpness
(`--phase train`); use `--phase val --length 100` for a fixed evaluation
length.  If several folders of one method contain the same
(task, architecture, seed) — e.g. several learning rates or BGF weight
pairs — select the intended ones with `--adam_folders`/`--bgf_folders`.
