# Balancing Gradient Frequencies Facilitates Inductive Inference in Algorithmic Reasoning

> **Acknowledgment:** This repository utilizes code from Google DeepMind's research on the Neural Networks Chomsky Hierarchy.
> * **Paper:** [Neural Networks and the Chomsky Hierarchy](https://arxiv.org/abs/2207.02098)
> * **Original Repository:** [google-deepmind/neural_networks_chomsky_hierarchy](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy)

This repository contains code to run and manage experiments across various model architectures and tasks.

---

## 🔧 Installation

Python 3.11; all dependencies are pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The pinned `jax==0.4.20` is the CPU wheel — for GPU training install the
matching CUDA build instead (see the note inside `requirements.txt`; the
paper's experiments used `jaxlib 0.4.20+cuda11.cudnn86`).  All scripts use
relative default paths (results under the respective experiment directory or
`./results`); every path can be overridden with `--save_dir`.

## 🗺️ Reproduction Map

Each experiment of the paper has a self-contained directory with its own
runner scripts, analysis/plotting code and README:

| Paper result | Directory | Scripts |
| :--- | :--- | :--- |
| Main generalization tables (Adam / BGF / BGF-EMA / ADD, 15 tasks × 8 architectures) | `run_experiments/` | `run_adam.py`, `run_bgf.py`, `run_bgf_ema.py`, `run_add.py` |
| AdamW variants of the main tables | `run_experiments/` | `run_adamW.py`, `run_bgf_adamW.py`, `run_bgf_ema_adamW.py` |
| Long-range generalization (lengths 1–1000, MD) | `run_long_eval/` | `long_eval.py` |
| Spurious-correlation toy experiments | `run_toy_experiment/` | `run_toy.py` (replays `paper_toy_configs.json`) |
| Gradient-frequency (FFT) analysis figures | `run_grad_visualization/` | `run_grad_training.py`, `plot_grad_fft.py` |
| Loss-landscape sharpness (3 metrics, paired t-tests by Chomsky level) | `run_loss_landscape_sharpness/` | `sharp_eval.py`, `analyze_sharpness.py` |
| 1M-step momentum experiment (MD, Adam vs BGF) | `run_1M_training/` | `run_1M.py`, `summarize_results.py` |
| Fine-grained Adam momentum sweep (MD, RNN/Tape-RNN) | `run_adam_momentum_analysis/` | `run_momentum_sweep.py`, `summarize_momentum.py` |

## 📊 Reproducing the Paper Results (`run_experiments/`)

Four sweep scripts generate the complete main results (15 tasks × 8 architectures,
learning rates {1e-4, 3e-4, 5e-4}, seeds {0, 1, 2}, 100k training steps):

| Script | Method (`--optim`) | Method-specific grid | Runs |
| :--- | :--- | :--- | :--- |
| `run_adam.py` | Adam (`none`) | – | 1080 |
| `run_bgf.py` | queue-based BGF (`ours_balance`) | weights (a,b) ∈ {(0.7,0.3), (0.8,0.2), (0.9,0.1), (0.95,0.05)}; queue size λ via `--queue_size` (default 100) | 4320 |
| `run_bgf_ema.py` | EMA-based BGF (`ours_ema`) | same weight pairs; smoothing via `--ema_sm` (default 0.98) | 4320 |
| `run_add.py` | ADD (`ours_add`) | g = α·g + β·g_low with α = 1, β ∈ {0.5, 1.0, 2.0, 3.0} (no balancing/normalization) | 4320 |

Each script enumerates its grid with 1-based indices and executes the slice
`[--start, --end)` on the GPU given by `--cuda`; use `--dry_run` to list the
grid, and `--tasks/--architectures/--lrs/--seeds` to restrict it.

```bash
cd run_experiments
python run_adam.py --dry_run                        # list the grid
python run_adam.py --cuda 0                          # full Adam baseline
python run_bgf.py  --cuda 0 --start 1 --end 2161     # first half of the BGF grid
python run_bgf.py  --cuda 1 --start 2161             # second half on another GPU
python run_bgf.py  --cuda 0 --queue_size 300         # queue-size (λ) ablation
python run_add.py  --cuda 0 --betas 2.0              # single ADD β
```

**AdamW experiments are also included.**  The same grids can be run with
**AdamW** as the underlying optimizer (`--base_optim adamw`; decoupled weight
decay 1e-4 by default, configurable with `--weight_decay`).  Since BGF is
built on top of the base optimizer, the weight decay applies to the BGF
variants as well.  These runs are written to `./results_adamw` by default,
with an `_adamw` folder prefix, so they stay separate from the Adam results:

| Script | Method (`--optim`, on AdamW) | Method-specific grid | Runs |
| :--- | :--- | :--- | :--- |
| `run_adamW.py` | AdamW baseline (`none`) | – | 1080 |
| `run_bgf_adamW.py` | queue-based BGF (`ours_balance`) | same weight pairs; `--queue_size` (default 100) | 4320 |
| `run_bgf_ema_adamW.py` | EMA-based BGF (`ours_ema`) | same weight pairs; `--ema_sm` (default 0.98) | 4320 |

```bash
python run_adamW.py         --cuda 0                 # AdamW baseline
python run_bgf_adamW.py     --cuda 0                 # BGF on AdamW
python run_bgf_ema_adamW.py --cuda 0                 # EMA-BGF on AdamW
```

## 📏 Long-Range Generalization (`run_long_eval/`)

The paper's long-range figure evaluates trained models on sequence lengths
1–1000 (default task: `missing_duplicate_string`, MD). The standalone script
`run_long_eval/long_eval.py` is independent of the training pipeline and of
the standard end-of-training evaluation over lengths 1–100; see the README in
`run_long_eval/`. In short:

```bash
# 1. train with checkpointing enabled (writes params/param_s<steps>.pt)
cd run_experiments
python run_adam.py --cuda 0 --tasks missing_duplicate_string --save_param 1
python run_bgf.py  --cuda 0 --tasks missing_duplicate_string --save_param 1

# 2. evaluate the checkpoints on lengths 1..1000
cd ../run_long_eval
python long_eval.py --save_dir ../run_experiments/results --cuda 0
```

Results are written as `long_range_eval_s<step>.json` (`accuracy_per_length`)
into each run directory.

## 🎚️ Fine-Grained Adam Momentum Analysis (`run_adam_momentum_analysis/`)

Sweeps Adam's momentum (beta1) over 15 values from 0.01 to 0.99 on the MD
task with the RNN and Tape-RNN models (seeds {0, 1, 2}, lr 5e-4, 1M steps;
90 runs).  The summary reports, per momentum: best test accuracy and the
steps needed to reach 90% / 95% training accuracy.  See the README there.

```bash
cd run_adam_momentum_analysis
python run_momentum_sweep.py --cuda 0    # train (use --start/--end to split)
python summarize_momentum.py             # report tables + CSVs
```

## 🏔️ Loss-Landscape Sharpness (`run_loss_landscape_sharpness/`)

The paper's three sharpness measures — low-pass-filter-based, FIM-based and
Shannon-based — evaluated on trained Adam/BGF checkpoints of the four
recurrent models across all 15 tasks, then summarised per Chomsky hierarchy
level with paired t-tests (NaN results excluded pairwise).  Requires runs
trained with `--save_param 1`; see the README there.

```bash
cd run_loss_landscape_sharpness
python sharp_eval.py --save_dir ../run_experiments/results --cuda 0  # per-run sharpness.json
python analyze_sharpness.py --save_dir ../run_experiments/results    # tables + t-tests
```

## ⏱️ 1M-Step Momentum Experiment (`run_1M_training/`)

Long-horizon (1,000,000-step) comparison of Adam and BGF on the MD task
(`missing_duplicate_string`) with the four recurrent architectures, sweeping
Adam's momentum (beta1) over {0.8, 0.9, 0.95} — BGF sits on top of Adam, so
the momentum affects both methods.  BGF weights fixed to (0.95, 0.05); seeds
{0, 1, 2}; 72 runs total.  See the README there.

```bash
cd run_1M_training
python run_1M.py --cuda 0                 # train (use --start/--end to split)
python summarize_results.py               # best-generalization summary table
```

## 📈 Gradient-Frequency Visualization (`run_grad_visualization/`)

The paper's gradient-spectrum analysis lives in `run_grad_visualization/`;
see the README there.  Gradients are recorded at every training step as 500
sampled coordinates (`--save_grad_signal 1`), for the Solve Equation task with
the LSTM model.  Seven figures are produced: four comparing Adam vs BGF
(`FFT_*`, `boxFFTnorm_*`, each for the training and the generalization phase)
and three comparing the training vs the generalization phase within the Adam
run alone (`js_overlay_*`, `js_box_norm_*`, `js_thresholds_simple_*`):

```bash
cd run_grad_visualization
python run_grad_training.py --cuda 0   # Adam + BGF runs, writes grad_signal.pkl
python plot_grad_fft.py                # writes the 7 figures into ./figures
```

## 🧪 Toy Experiments (`run_toy_experiment/`)

The spurious-correlation toy experiments (label concatenation and input-length
confounding) live in `run_toy_experiment/`; see the README there. The full
paper grid is replayed from `paper_toy_configs.json` with:

```bash
cd run_toy_experiment
python run_toy.py --experiment label  --cuda 0
python run_toy.py --experiment length --cuda 0
```

---

## 🚀 Running a Single Experiment (`one_sample.py`)

`one_sample.py` is a helper script designed to run a single instance of `basic_parser.py` without complex loops, allowing you to easily specify the task and architecture you want to test.

### 1. Basic Execution
Run the experiment with default values (e.g., `modular_arithmetic`, `rnn`, `seed=3`).

```bash
python one_sample.py
```

### 2. Custom Execution
You can manually specify the task, model architecture, seed, learning rate, and other parameters.

```bash
python one_sample.py \
    --task parity_check \
    --architecture lstm \
    --seed 0 \
    --model_init_seed 0 \
    --valid_seed 0 \
    --lr 1e-3 \
    --training_steps 1000000 \
    --cuda 1
```

---

## ⚙️ `basic_parser.py` Arguments

Below is the description of the `argparse` arguments used in the main `basic_parser.py` script.

| Argument | Type |   Default   | Description                                                                                                                                                    |
| :--- | :---: |:-----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **General & Environment** | |             |                                                                                                                                                                |
| `--cuda` | `int` |     `0`     | CUDA device ID to use (e.g., 0, 1, 2)                                                                                                                          |
| `--save_dir` | `str` | `Required` | Root directory path to save experiment results and model checkpoints                                                                                           |
| `--folder_name` | `str` |  `'debug'`  | Sub-folder name to store results for individual experiments                                                                                                    |
| **Seed Settings** | |             |                                                                                                                                                                |
| `--seed` | `int` |     `0`     | Global seed for data generation and random number control                                                                                                      |
| `--model_init_seed` | `int` |     `0`     | Seed for initializing model parameters                                                                                                                         |
| `--valid_seed` | `int` |     `0`     | Random seed for validation data generation                                                                                                                     |
| **Experiment & Task** | |             |                                                                                                                                                                |
| `--task` | `str` | `Required`  | Name of the task to run (e.g., `modular_arithmetic`, `parity_check`) - see **tasks**                                                                           |
| `--architecture` | `str` | `Required`  | Model architecture to use (e.g., `RNN`, `LSTM`, `Transformer_none`) - see **models**                                                                           |
| **Model Hyperparameters** | |             |                                                                                                                                                                |
| `--hidden_size` | `int` |    `256`    | Hidden dimension size of the model                                                                                                                             |
| `--memory_cell_size` | `int` |     `8`     | Number of memory stacks or tapes (`Stack-RNN` and `Tape-RNN`)                                                                                                  |
| `--memory_size` | `int` |    `40`     | Number of memory cells in each stack or tape (`Stack-RNN` and `Tape-RNN`)                                                                                      |
| `--is_autoregressive` | `bool` |   `False`   | Whether to use an autoregressive modeling approach                                                                                                             |
| **Training Hyperparameters**| |             |                                                                                                                                                                |
| `--training_steps` | `int` | `1,000,000` | Total number of training steps                                                                                                                                 |
| `--batch_size` | `int` |    `128`    | Batch size per training step                                                                                                                                   |
| `--sequence_length` | `int` |    `40`     | Length of the input sequence                                                                                                                                   |
| `--lr` | `float` |   `5e-4`    | Learning rate                                                                                                                                                  |
| `--optim` | `str` |  `'none'`   | Type of optimizer to use (`none`: default Adam, `ours_balance`: BGF with sliding window, `ours_ema`: BGF with EMA, `ours_add`: Non-balanced gradient addition) |
| `--momentum` | `float` |    `0.9`    | Momentum (beta1) value for the Adam optimizer                                                                                                                  |
| `--base_optim` | `str` |  `'adam'`   | Underlying optimizer of the baseline and every BGF variant (`adam` or `adamw`)                                                                                 |
| `--weight_decay` | `float` |   `1e-4`    | Decoupled weight decay (used when `--base_optim adamw`)                                                                                                        |
| `--ema_sm` | `float` |   `0.98`    | Exponential Moving Average (EMA) smoothing parameter of `ours_ema`                                                                                             |
| `--queue_size` | `int` |    `100`    | Queue size (λ) of the gradient queue used by the queue-based BGF (`ours_balance`) and `ours_add`                                                               |
| `--save_param` | `int` |     `0`     | `0`: no checkpoints; `1`: save parameters at steps [0, 10, 100, ..., final] plus the final optimizer state (required for `long_eval.py`); `2`: dense schedule  |
| `--save_grad_signal` | `int` |     `0`     | `1`: record 500 sampled gradient coordinates at every step into `grad_signal.pkl` (used by `run_grad_visualization/`)                                           |
| `--weight_a` | `float` |    `1.0`    | Original gradient (high-frequency) proportion                                                                                                                  |
| `--weight_b` | `float` |    `1.0`    | Low-frequency gradient proportion                                                                                                                              |
| **Validation & Testing** | |             |                                                                                                                                                                |
| `--valid_length` | `int` |    `100`    | Sequence length for validation data                                                                                                                            |
| `--max_range_test_length` | `int` |    `100`    | Maximum length for length generalization tests                                                                                                                 |
| `--range_test_sub_batch_size`| `int` |    `128`    | Sub-batch size used during range testing                                                                                                                       |