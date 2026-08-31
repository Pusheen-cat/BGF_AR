# SMILES / ChemBERTa experiments (AdamW vs. BGF, settings 1/2/3)

Fine-tunes three pretrained ChemBERTa encoders on four MoleculeNet datasets under a
scaffold split, comparing the **BGF** optimizer against the **AdamW** baseline. Each
comparison is paired (same seeds, same config, only the optimizer differs). BGF
fixes **`bgf_alpha = 0.95`** (used without hyperparameter search) and
`bgf_lambda = 0.01`.

**ChemBERTa settings** (selected by `model.pretrained` in the config):

| Setting | `model` | Checkpoint | Params | Pretraining |
|---|---|---|---|---|
| ChemBERTa-1 | `chemberta`  | `DeepChem/ChemBERTa-77M-MTR` | 77M | Multi-Task Regression |
| ChemBERTa-2 | `chemberta2` | `DeepChem/ChemBERTa-77M-MLM` | 77M | Masked Language Modeling |
| ChemBERTa-3 | `chemberta3` | `DeepChem/ChemBERTa-10M-MTR` | 10M | Multi-Task Regression |

**Datasets** (MoleculeNet, scaffold split): ESOL (regression; RMSE, MAE), BBBP, HIV,
Tox21 (classification; ROC-AUC, PRC-AUC). ESOL/BBBP use 10 seeds; HIV/Tox21 use 5.
See `../README.md` for dataset characteristics and the BGF definition.

## Layout

```
configs/    {esol,bbbp,hiv,tox21}_chemberta{,2,3}.yaml   (task/model/train settings)
scripts/
  reproduce_smiles.py   run all datasets x settings x {adamw,bgf} x seeds (resumable, multi-GPU)
  compare_results.py    AdamW-vs-BGF paired tables
src/smiles_pp/          the fine-tuning library (chemberta model, data, training, optim)
```

The optimizer and BGF hyperparameters are passed on the command line by
`reproduce_smiles.py` (`--optimizer`, `--bgf-alpha 0.95`, `--bgf-lambda 0.01`); the
YAML configs carry the task, model checkpoint, and training schedule.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
# install a CUDA build of torch first if needed, then:
pip install -e .            # installs smiles_pp + the `smiles-pp` command
```

`transformers` downloads the ChemBERTa checkpoints from HuggingFace on first use.

## 1. Data (optional pre-caching)

Datasets are downloaded and scaffold-split via DeepChem and cached to
`data/<name>_scaffold.csv` **automatically on first use**. To pre-cache all four
once (recommended before launching many parallel runs, to avoid a first-run race):

```bash
for d in esol bbbp hiv tox21; do smiles-pp prepare --dataset $d; done
```

## 2. Reproduce AdamW vs BGF

```bash
python scripts/reproduce_smiles.py --gpus 0 1 2 3 4 5 6 7
python scripts/compare_results.py            # -> results_smiles_compare.{md,csv}
```

`reproduce_smiles.py` writes `outputs/<task>/<variant>/<optimizer>/seed<k>/final.json`
(the test metric at the best-validation epoch); it is resumable and schedules one
run per GPU. `compare_results.py` reports, per dataset / setting / metric, mean±std
over seeds, Δ = BGF − AdamW, the better optimizer, BGF win-rate, and a paired t-test.

If `smiles_pp` is not pip-installed, set `SMILES_PY` to a Python interpreter that can
import it (the scripts already add `src/` to `PYTHONPATH`):

```bash
SMILES_PY=python python scripts/reproduce_smiles.py --gpus 0
```

## Single run

```bash
PYTHONPATH=src python -m smiles_pp.study --task esol --model chemberta \
    --optimizer bgf --bgf-alpha 0.95 --bgf-lambda 0.01 --seed 0 \
    --outdir outputs/esol/chemberta/bgf/seed0 --device cuda
```
