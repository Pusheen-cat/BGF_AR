# Gradient-Frequency Visualization

Reproduces the paper's gradient-spectrum analysis: the raw gradient of an
**Adam** run is compared against the filtered (actually applied) gradient of a
**BGF** run in the frequency domain, during the **training** and the
**generalization** phase.  Because saving the full gradient tree at every one
of the 100k steps would require excessive storage, only **500 sampled gradient
coordinates** are recorded per step (at least 5 per bias / 20 per weight
tensor, the rest proportional to tensor size; fixed sampling seed, so both
runs record the *same* coordinates).  The experiment is set up for the
**Solve Equation** task with the **LSTM** model.

## 1. Training stage (`run_grad_training.py`)

Launches the two runs (Adam baseline + BGF with weights a=0.9, b=0.1;
lr 5e-4, seed 2, 100k steps) with `--save_grad_signal 1`, which makes the
training loop write `<run_dir>/grad_signal.pkl` — the per-step sampled
gradient signals (`raw` for both runs, plus `filtered` for the BGF run,
starting at the filter warm-up step 100):

```bash
python run_grad_training.py --cuda 0                 # both runs sequentially
python run_grad_training.py --cuda 0 --methods adam  # or one per GPU
python run_grad_training.py --cuda 1 --methods bgf
```

## 2. Figures (`plot_grad_fft.py`)

```bash
python plot_grad_fft.py            # reads ./results_grad, writes ./figures
```

Seven figures are produced; the numbers in the file names are the
analysis-window start steps.

Four **Adam-vs-BGF** figures (two types x two phases; Adam start first, BGF
start second):

| Figure | Content |
| :--- | :--- |
| `FFT_<s>_<s>.pdf` | Training phase — per-coordinate FFT line plots, Baseline vs BGF |
| `FFT_<sb>_<so>.pdf` | Generalization phase — the same |
| `boxFFTnorm_<s>_<s>.pdf` | Training phase — box plots of the per-coordinate band power fractions (Low / High / Very-high), each band normalised by its Adam median |
| `boxFFTnorm_<sb>_<so>.pdf` | Generalization phase — the same |

Three **training-vs-generalization** figures comparing the two phases WITHIN
the Adam run alone (its raw gradient signal; BGF is not involved — training
start first, generalization start second):

| Figure | Content |
| :--- | :--- |
| `js_overlay_<tb>_<tg>.pdf` | Per-coordinate \|FFT\| overlays (red = training, blue = generalization) for five sampled coordinates plus an all-coordinate aggregate panel |
| `js_box_norm_<tb>_<tg>.pdf` | Box plots of the per-coordinate band power fractions, each band normalised by its Training median |
| `js_thresholds_simple_<tb>_<tg>.pdf` | Mean power-fraction spectra of the two phases with the per-log-bin gen/train power ratio drawn as gray bars behind them (bar baseline = 1 aligned with the level where the two spectra cross) |

**Phases.**  The two phases use different window starts:

* *Training phase*: the 10,000-step window starting at step **100**
  (`--train_start`).
* *Generalization phase*: the 10,000-step window starting at the first step
  at which the trailing **50-step moving average of the training accuracy
  reaches 95%** (`--ma_window`, `--acc_threshold`).  The onset is computed
  separately for each run from its own training log, so the Adam and BGF
  windows generally start at different steps.

**Analysis.**  For the Adam-vs-BGF figures, signals are L2-normalised per step
across the 500 coordinates (removes the decaying gradient-magnitude envelope)
and z-scored per analysis window (equal total spectral power, by Parseval, so
the comparison is purely about how energy is distributed over frequency).  The
box plots partition each spectrum's power into three bands — Low < 10^-1.7,
High 10^-1.7..10^-1.1, Very-high >= 10^-1.1 Hz (`--t1`, `--t2`) — and report
the paired Wilcoxon significance across the coordinates.  The line plots show
the 10-tap-smoothed FFT magnitudes of the 12 default coordinates (`--coords`).
The single-run `js_*` figures use per-window z-scoring only (the L2 step exists
to make two different runs comparable, which is unnecessary within one run)
and the same bands and statistics.

Useful options: `--window` (analysis-window length), `--gen_start_adam` /
`--gen_start_bgf` (manual onset overrides), `--adam_run` / `--bgf_run`
(explicit run directories when several runs exist under `--save_dir`).
