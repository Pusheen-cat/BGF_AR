# Fine-Grained Adam Momentum Analysis (MD task)

Sweeps Adam's momentum (beta1) over 15 values

```
0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99
```

on the `missing_duplicate_string` (MD) task with the **RNN** and **Tape-RNN**
architectures (plain Adam, `--optim none`; learning rate 5e-4; seeds 0, 1, 2;
1,000,000 training steps).  Grid: 2 x 15 x 3 = **90 runs**.

## 1. Training (`run_momentum_sweep.py`)

```bash
python run_momentum_sweep.py --dry_run                     # list the 90 runs
python run_momentum_sweep.py --cuda 0 --start 1 --end 46   # split across GPUs
python run_momentum_sweep.py --cuda 1 --start 46
```

Result folders embed the momentum (`baseline-0.0005-1000000-m0.2/`, ...).

## 2. Report (`summarize_momentum.py`)

```bash
python summarize_momentum.py         # reads ./results_momentum
```

Per architecture and momentum, averaged over the seeds (all accuracies use a
50-step moving average, `--ma_window`):

* **best test accuracy** — maximum of the moving-averaged per-step validation
  accuracy (sequence length 100) reached at any point during training;
* **steps to 90% training accuracy** — the step at which the trailing
  moving average of the training accuracy first reaches 90% (seeds that never
  reach it are excluded; the reached/found count is reported);
* **steps to 95% training accuracy** — the same for 95%.

Tables are printed and written to `summary_momentum.csv`; the per-run values
go to `runs_momentum.csv`.
