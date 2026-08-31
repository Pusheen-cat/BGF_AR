#!/usr/bin/env python
"""Convert DeepSEA ``.mat`` files into tidy HDF5 tensors.

    python scripts/prepare_hdf5.py --raw_dir data/raw/deepsea

Produces::

    data/processed/deepsea.h5          (full)
    data/processed/deepsea_debug.h5    (10k train / 2k valid / 2k test)

Each file has groups ``train/valid/test`` each with::

    X  float32 [N, 4, 1000]
    y  float32 [N, 919]

Design notes
------------
* The train split has ~4.4M rows -> it is streamed in chunks from the source
  ``.mat`` (never fully materialised) so RAM stays bounded.  valid/test are
  small enough to load whole.
* Axis order is detected, not assumed (see ``src/data/mat_io``): a ``[1000,4,N]``
  (h5py-transposed) or ``[N,4,1000]`` (scipy) source both normalise to
  ``[N,4,1000]``.
* If a sequence array turns out to be integer-encoded (values not just 0/1 over
  a size-4 axis) we still one-hot it; DeepSEA ships true one-hot so this is a
  safety net only.
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.mat_io import (  # noqa: E402
    MatFile,
    find_roles,
    label_axis_map,
    n_samples,
    seq_axis_map,
    to_ncl,
    to_nl,
)

# how source .mat files map onto splits; the first existing filename wins
SPLIT_FILES = {
    "train": ["train.mat"],
    "valid": ["valid.mat", "validation.mat"],
    "test": ["test.mat"],
}
DEBUG_SIZES = {"train": 10_000, "valid": 2_000, "test": 2_000}


def _find_split_file(raw_dir: str, split: str) -> str | None:
    for fn in SPLIT_FILES[split]:
        p = os.path.join(raw_dir, fn)
        if os.path.exists(p):
            return p
    return None


def _write_split(
    dst: h5py.File,
    split: str,
    src_path: str,
    *,
    limit: int | None,
    chunk: int,
    compression: str | None,
) -> int:
    """Stream one split from ``src_path`` into ``dst[<split>]``. Returns N."""
    with MatFile(src_path) as mat:
        roles = find_roles(mat)
        seq_key = next((k for k, r in roles.items() if r == "sequence"), None)
        lbl_key = next((k for k, r in roles.items() if r == "labels"), None)
        if seq_key is None or lbl_key is None:
            raise RuntimeError(
                f"could not identify sequence/label arrays in {src_path}; "
                f"keys={mat.keys()}"
            )
        xarr = mat.array(seq_key)
        yarr = mat.array(lbl_key)
        n_ax, c_ax, l_ax = seq_axis_map(xarr.shape)
        yn_ax, ylab_ax = label_axis_map(yarr.shape)

        n_total = n_samples(xarr.shape, n_ax)
        n_y = n_samples(yarr.shape, yn_ax)
        if n_total != n_y:
            raise RuntimeError(
                f"{src_path}: X has {n_total} rows but y has {n_y}"
            )
        n = n_total if limit is None else min(limit, n_total)
        print(f"  [{split}] {src_path}  X{xarr.shape} y{yarr.shape}  -> N={n}"
              f" ({'full' if limit is None else 'debug'})")

        grp = dst.create_group(split)
        xds = grp.create_dataset(
            "X", shape=(n, 4, 1000), dtype="float32",
            chunks=(min(chunk, n) or 1, 4, 1000), compression=compression,
        )
        yds = grp.create_dataset(
            "y", shape=(n, 919), dtype="float32",
            chunks=(min(chunk, n) or 1, 919), compression=compression,
        )

        # Build slicers that index only the N axis of the original array.
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            xsl = _axis_slice(xarr.shape, n_ax, start, stop)
            ysl = _axis_slice(yarr.shape, yn_ax, start, stop)
            xchunk = to_ncl(xarr.read(xsl), n_ax, c_ax, l_ax)
            ychunk = to_nl(yarr.read(ysl), yn_ax, ylab_ax)
            # safety: clip to {0,1} float; if source was raw counts this is a no-op
            xds[start:stop] = xchunk
            yds[start:stop] = ychunk
            print(f"\r     wrote {stop}/{n}", end="", flush=True)
        print()
    return n


def _axis_slice(shape, axis: int, start: int, stop: int):
    """A tuple index selecting ``start:stop`` along ``axis`` (`:` elsewhere)."""
    idx = [slice(None)] * len(shape)
    idx[axis] = slice(start, stop)
    return tuple(idx)


def _split_n(raw_dir: str, split: str) -> int | None:
    """Cheaply read the number of samples in a split (metadata only)."""
    src = _find_split_file(raw_dir, split)
    if src is None:
        return None
    with MatFile(src) as mat:
        roles = find_roles(mat)
        seq_key = next((k for k, r in roles.items() if r == "sequence"), None)
        if seq_key is None:
            return None
        arr = mat.array(seq_key)
        n_ax, _c, _l = seq_axis_map(arr.shape)
        return n_samples(arr.shape, n_ax)


def build_file(out_path: str, raw_dir: str, *, limits: dict[str, int | None],
               chunk: int, compression: str | None) -> None:
    print(f"\n==> building {out_path}")
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as dst:
        dst.attrs["source"] = os.path.abspath(raw_dir)
        dst.attrs["limits"] = repr(limits)
        for split in ("train", "valid", "test"):
            src = _find_split_file(raw_dir, split)
            if src is None:
                print(f"  [{split}] WARNING: no source .mat found "
                      f"(looked for {SPLIT_FILES[split]}) -- skipping")
                continue
            _write_split(dst, split, src, limit=limits.get(split), chunk=chunk,
                         compression=compression)
    os.replace(tmp, out_path)
    print(f"  wrote {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw/deepsea")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--chunk", type=int, default=4096,
                    help="rows streamed/written at a time")
    ap.add_argument("--compression", default=None,
                    choices=[None, "gzip", "lzf"],
                    help="HDF5 compression (default none = fastest/largest)")
    ap.add_argument("--only", choices=["full", "debug", "subset", "both"],
                    default="both",
                    help="'both'=debug+full; 'subset' uses --frac/--*_limit")
    ap.add_argument("--frac", type=float, default=None,
                    help="subset mode: fraction of train & test to keep "
                         "(valid kept full unless --valid_limit given)")
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--valid_limit", type=int, default=None)
    ap.add_argument("--test_limit", type=int, default=None)
    ap.add_argument("--out_name", default="deepsea_subset.h5",
                    help="output filename for subset mode")
    args = ap.parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"raw_dir not found: {args.raw_dir}", file=sys.stderr)
        return 1
    mats = [f for f in os.listdir(args.raw_dir) if f.endswith(".mat")]
    if not mats:
        print(f"No .mat files in {args.raw_dir}. Run download_deepsea.py first.",
              file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)

    if args.only in ("debug", "both"):
        build_file(os.path.join(args.out_dir, "deepsea_debug.h5"),
                   args.raw_dir, limits=dict(DEBUG_SIZES), chunk=args.chunk,
                   compression=args.compression)
    if args.only in ("full", "both"):
        build_file(os.path.join(args.out_dir, "deepsea.h5"),
                   args.raw_dir, limits={"train": None, "valid": None,
                                         "test": None},
                   chunk=args.chunk, compression=args.compression)
    if args.only == "subset":
        limits: dict[str, int | None] = {"train": None, "valid": None,
                                         "test": None}
        if args.frac is not None:
            if not (0 < args.frac <= 1):
                print("--frac must be in (0, 1]", file=sys.stderr)
                return 1
            for split in ("train", "test"):  # valid kept full by default
                n = _split_n(args.raw_dir, split)
                if n is not None:
                    limits[split] = max(1, int(round(args.frac * n)))
        # explicit per-split limits override --frac
        if args.train_limit is not None:
            limits["train"] = args.train_limit
        if args.valid_limit is not None:
            limits["valid"] = args.valid_limit
        if args.test_limit is not None:
            limits["test"] = args.test_limit
        print(f"subset limits: {limits}")
        build_file(os.path.join(args.out_dir, args.out_name), args.raw_dir,
                   limits=limits, chunk=args.chunk,
                   compression=args.compression)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
