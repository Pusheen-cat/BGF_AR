"""RDKit-based SMILES canonicalization and validity filtering."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

# Silence RDKit's very chatty C++ logger; we report drops ourselves.
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover - rdkit always present in the env
    pass


def canonical_smiles(smiles: str, *, isomeric: bool = True) -> str | None:
    """Return the RDKit canonical SMILES, or ``None`` if it cannot be parsed."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def is_valid(smiles: str) -> bool:
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles) is not None


def clean_dataset(
    smiles: Sequence[str],
    labels: Iterable[float],
    *,
    isomeric: bool = True,
    deduplicate: bool = False,
) -> tuple[list[str], np.ndarray, int]:
    """Canonicalize a (smiles, labels) pair and drop unparseable molecules.

    Returns ``(canonical_smiles, labels, n_dropped)``. When ``deduplicate`` is
    set, repeated canonical SMILES keep only their first occurrence (off by
    default so that the MoleculeNet scaffold splits are preserved exactly).
    """
    labels = np.asarray(list(labels), dtype=np.float64)
    out_smiles: list[str] = []
    out_labels: list[float] = []
    seen: set[str] = set()
    n_dropped = 0
    for smi, lab in zip(smiles, labels):
        canon = canonical_smiles(smi, isomeric=isomeric)
        if canon is None:
            n_dropped += 1
            continue
        if deduplicate and canon in seen:
            continue
        seen.add(canon)
        out_smiles.append(canon)
        out_labels.append(lab)
    if n_dropped:
        logger.warning("Dropped %d invalid SMILES during canonicalization", n_dropped)
    return out_smiles, np.asarray(out_labels, dtype=np.float64), n_dropped
