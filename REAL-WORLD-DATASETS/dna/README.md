# DNA / DeepSEA experiments (RNN, LSTM, Stack-RNN, Tape-RNN)

Compares the **BGF** optimizer against the **AdamW** baseline on the DeepSEA
chromatin-profile task, for four from-scratch sequence models. Every input is a
1,000-bp DNA window encoded as `[4, 1000]`; every model outputs raw logits over
the **919 binary chromatin labels**. Metrics: AUROC and AUPRC averaged over the
919 labels (mean and median), on the test split.

| Model | `model.type` | file | notes |
|---|---|---|---|
| Vanilla RNN | `rnn` | `src/models/rnn.py` | 2-layer bidirectional |
| LSTM | `lstm` | `src/models/lstm.py` | 2-layer bidirectional, AMP |
| Stack-RNN | `stack_rnn` | `src/models/stack_rnn.py` | 1-layer unidirectional, differentiable stack |
| Tape-RNN | `tape_rnn` | `src/models/tape_rnn.py` | 1-layer unidirectional, differentiable tape |

**Optimizer.** `training.optimizer` is `adamw` or `bgf`. BGF fixes
**`bgf_alpha = 0.95`** (used without hyperparameter search) and `bgf_lambda = 0.01`
(see `../README.md` for the BGF definition and `src/training/optim.py` for the
implementation).

**Gradient clipping and NaN-skipping.** All runs apply gradient clipping every
step. The optimizer skips any step with a non-finite gradient (NaN-skipping). Both
are applied to the **Stack-RNN and Tape-RNN** runs on both the BGF and the AdamW
(NaN-guarded `SafeAdamW`) arms; for those two models the AdamW baseline is
`SafeAdamW`, which is identical to `torch.optim.AdamW` whenever gradients are finite.

## Layout

```
configs/          {rnn,lstm,stack_rnn,tape_rnn}_{adamw,bgf}.yaml   (BGF configs fix alpha=0.95)
scripts/
  download_deepsea.py            fetch the DeepSEA .mat bundle
  prepare_hdf5.py                build the processed HDF5 (incl. the 10% subset)
  train.py                       train one model (one config, one seed)
  evaluate.py                    evaluate a checkpoint on a split
  reproduce_dna.py               run all archs x {adamw,bgf} x seeds  (resumable, multi-GPU)
  reproduce_lambda_rnn_lstm.py   BGF bgf_lambda sweep for RNN & LSTM only
  compare_results.py             AdamW-vs-BGF tables (--mode compare) and lambda tables (--mode lambda)
src/
  data/     dna_encoding.py, deepsea_dataset.py, mat_io.py
  models/   rnn.py, lstm.py, stack_rnn.py, tape_rnn.py, __init__.py (build_model)
  training/ trainer.py, optim.py (BGF + SafeAdamW), losses.py, metrics.py, utils.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # install a CUDA build of torch first if needed
```

## 1. Data

```bash
python scripts/download_deepsea.py --raw_dir data/raw/deepsea
# Reported experiments use a 10% training subset (test split = 45,502 windows):
python scripts/prepare_hdf5.py --raw_dir data/raw/deepsea --out_dir data/processed \
    --only subset --frac 0.1 --out_name deepsea_sub10.h5
```

This writes `data/processed/deepsea_sub10.h5` with `train` / `valid` / `test`
splits. (Use `--only full` for the complete dataset.)

## 2. Reproduce AdamW vs BGF (four architectures)

```bash
# reported seed counts: RNN/LSTM = 20, Stack-RNN/Tape-RNN = 5
python scripts/reproduce_dna.py --data data/processed/deepsea_sub10.h5 --gpus 0 1 2 3 4 5 6 7
python scripts/compare_results.py            # -> results_dna_compare.{md,csv}
```

`reproduce_dna.py` trains + evaluates `outputs/<arch>_<optimizer>/seed<k>/`; it is
resumable and schedules one run per GPU. `compare_results.py` reports, per
architecture and metric, mean±std over seeds, Δ = BGF − AdamW, a paired t-test and
a Wilcoxon test (using only seed-pairs finite on both arms).

## 3. Lambda comparison (RNN & LSTM only)

Sweeps `bgf_lambda ∈ {0.001, 0.003, 0.01, 0.03, 0.1}` with `bgf_alpha` fixed at
0.95, for RNN and LSTM only (as requested; not run for the other settings):

```bash
python scripts/reproduce_lambda_rnn_lstm.py --data data/processed/deepsea_sub10.h5 \
    --seeds 20 --gpus 0 1 2 3 4 5 6 7
python scripts/compare_results.py --mode lambda    # -> results_dna_lambda.{md,csv}
```

## Single run

```bash
PYTHONPATH=src python scripts/train.py --config configs/lstm_bgf.yaml \
    --data data/processed/deepsea_sub10.h5 --outdir outputs/lstm_bgf/seed0 --seed 0
PYTHONPATH=src python scripts/evaluate.py --checkpoint outputs/lstm_bgf/seed0/checkpoints/run.pt \
    --data data/processed/deepsea_sub10.h5 --split test --outdir outputs/lstm_bgf/seed0 --name run
```
