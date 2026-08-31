# Code release — BGF vs. AdamW on DNA and SMILES tasks

Public code accompanying the revised manuscript. It contains everything needed to
reproduce the reported experiments comparing the **BGF** optimizer against the
**AdamW** baseline on two problem families:

- **DNA / DeepSEA** — four from-scratch sequence models: RNN, LSTM, Stack-RNN, Tape-RNN.
- **SMILES / ChemBERTa** — fine-tuning three pretrained ChemBERTa encoders (settings 1, 2, 3).

Each comparison is **paired**: the two optimizers use the same seeds and identical
hyperparameters — only the optimizer differs.

```
code_release/
  README.md              this file
  dna/                   DeepSEA experiments (RNN, LSTM, Stack-RNN, Tape-RNN)
  smiles/                ChemBERTa experiments (settings 1, 2, 3)
```

See `dna/README.md` and `smiles/README.md` for per-pipeline details. The two use
separate Python environments (`dna/requirements.txt`, `smiles/requirements.txt`).

---

## The BGF optimizer

BGF (Balanced Gradient Filter) pre-conditions each parameter's gradient before the
AdamW update. For every parameter tensor it keeps an exponential moving average
(EMA) of past gradients — the low-frequency gradient component — and blends it,
norm-matched, back into the raw gradient:

```
que <- (1 - lambda) * que + lambda * g            # EMA of gradients (seeded with the first g)
g'  <- alpha * g + (1 - alpha) * (||g|| / ||que||) * que
```

`g'` is then passed to a standard AdamW step. The norms are per-tensor L2
(Frobenius) norms, so the added term has the same norm as the raw gradient.

Hyperparameters:

- **`bgf_alpha` — fixed to 0.95 in this release.** It weights the raw gradient
  (`1 - alpha` weights the norm-matched EMA term). **α = 0.95 was used throughout
  without any hyperparameter search.**
- **`bgf_lambda` = 0.01** — the EMA smoothing factor (smaller = slower-moving EMA).
  It is the default for every experiment; the DNA RNN/LSTM lambda-comparison script
  is the only place it is varied.

Reference implementations: `dna/src/training/optim.py` and
`smiles/src/smiles_pp/optim.py` (identical BGF math). The **DNA** `optimizer: adamw`
baseline is a NaN-guarded AdamW (see "NaN-skipping" below); the **SMILES** AdamW
baseline is standard `torch.optim.AdamW`.

### Gradient clipping and NaN-skipping (DNA)

All DNA runs apply gradient clipping every step. The optimizer also applies
**NaN-skipping**: any step whose gradients contain a non-finite value is skipped
(parameters and the EMA buffer keep their last finite values), so one overflowing
batch cannot corrupt training. **Gradient clipping and NaN-skipping are applied to
the Stack-RNN and Tape-RNN runs** (both arms), whose 1-layer unidirectional cores
can otherwise produce non-finite gradients. For those two models this guard is on
for both the BGF and the AdamW (SafeAdamW) arm, so the comparison stays paired.

---

## ChemBERTa settings 1, 2, 3

Three pretrained encoders from DeepChem (HuggingFace), fine-tuned on each dataset:

| Setting | Checkpoint | Params | Pretraining |
|---|---|---|---|
| **ChemBERTa-1** | `DeepChem/ChemBERTa-77M-MTR` | 77M | Multi-Task Regression (MTR): pretrained to predict a set of computed molecular properties |
| **ChemBERTa-2** | `DeepChem/ChemBERTa-77M-MLM` | 77M | Masked Language Modeling (MLM): pretrained to predict masked SMILES tokens |
| **ChemBERTa-3** | `DeepChem/ChemBERTa-10M-MTR` | 10M | Multi-Task Regression (MTR) |

Settings 1 and 3 share the MTR pretraining objective and differ in size
(77M vs 10M); settings 1 and 2 share the size (77M) and differ in the pretraining
objective (MTR vs MLM).

---

## Datasets

### DNA — DeepSEA

Chromatin-profile prediction from DNA sequence. Each input is a **1,000-bp** window
one-hot encoded as `[4, 1000]` (A/C/G/T); the target is **919 binary labels** —
690 transcription-factor binding profiles, 125 DNase I hypersensitivity profiles,
and 104 histone-mark profiles. It is a multi-label classification task; performance
is reported as **AUROC and AUPRC averaged over the 919 labels** (mean and median).
The reported experiments use a **10% subset** of the DeepSEA training data (test
split: 45,502 windows); see `dna/README.md` for how to build it.

### SMILES — MoleculeNet (scaffold split)

Molecular property prediction from SMILES strings, using the standard MoleculeNet
datasets under a **scaffold split** (train/valid/test by molecular scaffold).

| Dataset | Task | Type | Molecules (approx.) | Metrics |
|---|---|---|---|---|
| **ESOL** | aqueous solubility (log) | regression | ~1,100 | RMSE, MAE (lower is better) |
| **BBBP** | blood-brain-barrier penetration | binary classification | ~2,000 | ROC-AUC, PRC-AUC |
| **HIV** | inhibition of HIV replication | binary classification | ~41,000 | ROC-AUC, PRC-AUC |
| **Tox21** | 12 toxicity assays | multi-task binary classification | ~7,800 | ROC-AUC, PRC-AUC (averaged over tasks) |

Seeds: ESOL and BBBP use 10 seeds; HIV and Tox21 use 5 seeds.

---

## Reproducing the experiments

Each pipeline is self-contained. In short:

```bash
# DNA (see dna/README.md for data download/prep)
cd dna
python scripts/reproduce_dna.py            --data data/processed/deepsea_sub10.h5 --gpus 0 1 2 3
python scripts/reproduce_lambda_rnn_lstm.py --data data/processed/deepsea_sub10.h5 --gpus 0 1 2 3
python scripts/compare_results.py                 # AdamW vs BGF tables
python scripts/compare_results.py --mode lambda   # RNN/LSTM lambda comparison

# SMILES (see smiles/README.md for install/data)
cd ../smiles
python scripts/reproduce_smiles.py --gpus 0 1 2 3
python scripts/compare_results.py                 # AdamW vs BGF tables
```

All reproduction scripts are resumable (completed runs are skipped) and schedule
one run per GPU across the GPUs passed via `--gpus`. BGF runs always use
`bgf_alpha = 0.95`.
