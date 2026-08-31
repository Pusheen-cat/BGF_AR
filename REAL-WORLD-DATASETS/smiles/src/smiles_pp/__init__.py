"""Reproducible molecular property prediction from SMILES.

A small, config-driven pipeline for fine-tuning pretrained ChemBERTa encoders on
MoleculeNet tasks, used here to compare the AdamW and BGF optimizers. Three
pretrained checkpoints are supported via ``model.pretrained`` (settings 1/2/3):
ChemBERTa-77M-MTR, ChemBERTa-77M-MLM, and ChemBERTa-10M-MTR.
"""

__version__ = "0.1.0"