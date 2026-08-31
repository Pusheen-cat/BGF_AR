# 1M-Step Momentum Experiment (MD task)

Long-horizon comparison of **Adam** and **BGF** under three values of Adam's
momentum (beta1) — since BGF is built on top of Adam, the momentum affects
both methods.  Setting:

| | |
| :--- | :--- |
| Task | `missing_duplicate_string` (MD) |
| Architectures | `rnn`, `lstm`, `stack_rnn`, `tape_rnn` |
| Methods | Adam (`--optim none`), BGF (`--optim ours_balance`) |
| Momentum (beta1) | 0.8, 0.9, 0.95 |
| BGF weights | (a, b) = (0.95, 0.05), fixed |
| Training steps | 1,000,000 |
| Learning rate | 5e-4 |
| Seeds | 0, 1, 2 |

Grid: 2 methods x 3 momenta x 4 architectures x 3 seeds = **72 runs**.

## 1. Training (`run_1M.py`)

```bash
python run_1M.py --dry_run                    # list the 72 runs
python run_1M.py --cuda 0 --start 1 --end 37  # split across GPUs
python run_1M.py --cuda 1 --start 37
python run_1M.py --cuda 0 --methods bgf --momentums 0.9   # subsets
```

Result folders embed the momentum as an `-m{momentum}` suffix, e.g.
`baseline-0.0005-1000000-m0.8/` and
`ours_balance-0.0005-0.95-0.05-1000000-m0.95/`.

## 2. Summary table (`summarize_results.py`)

```bash
python summarize_results.py                     # reads ./results_1M
python summarize_results.py --metric range_score
```

For every setting (momentum x architecture x method) the script identifies
the **best generalization result across the three seeds** and prints one
table per momentum (also written to `summary_1M.csv`; the per-run metrics go
to `runs_1M.csv`).  Metrics per run:

* `final_valid` (default) — mean validation accuracy (length 100) over the
  last 50 training steps: the end-of-training generalization accuracy.
* `max_valid` — best 50-step moving average of the validation accuracy
  reached at any point during training.
* `range_score` — mean end-of-training accuracy over the unseen lengths
  41–100 from the range evaluation.

"Best" is the maximum of the chosen metric across the seeds whose
end-of-training accuracy on the training lengths (1–40) is at least 0.98
(`--min_train_acc`) — a run that never fitted the training distribution has
no meaningful generalization score.  If no seed passes the guard, the best
over all seeds is shown marked with `*`.  The table additionally reports
mean ± std over all seeds.
