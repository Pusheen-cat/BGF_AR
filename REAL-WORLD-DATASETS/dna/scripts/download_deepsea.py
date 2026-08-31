#!/usr/bin/env python
"""Download and extract the DeepSEA training bundle.

    python scripts/download_deepsea.py [--raw_dir data/raw/deepsea]

The canonical archive is::

    http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz

which extracts ``train.mat``, ``valid.mat``, ``test.mat`` (plus some extras).
The Princeton host has been unreliable for years; we try a few known mirrors and,
if *all* fail, print clear instructions for placing the ``.mat`` files manually.
Selene SDK does NOT ship this data -- use this bundle directly.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile

# Candidate URLs, tried in order.  The first is the original; the others are
# community mirrors that have hosted the identical bundle.
URLS = [
    "http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz",
    "https://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz",
    "https://zenodo.org/record/5750592/files/deepsea_train_bundle.v0.9.tar.gz",
    "https://huggingface.co/datasets/dtch1997/deepsea/resolve/main/deepsea_train_bundle.v0.9.tar.gz",
]

MANUAL_MSG = """\
================================================================================
Automatic download FAILED for every mirror.

Please obtain `deepsea_train_bundle.v0.9.tar.gz` manually (search for the file
name; it is ~3-4 GB) and either:

  (a) drop the archive at:   {archive}
      then re-run:           python scripts/download_deepsea.py
      (this script will extract an already-present archive), OR

  (b) extract it yourself and place these files into:
          {raw}/
      The bundle extracts a folder containing at least:
          train.mat   valid.mat   test.mat
      Move (or symlink) those three .mat files directly into {raw}/.

Then continue with:
    python scripts/inspect_deepsea_mat.py --raw_dir {raw}
    python scripts/prepare_hdf5.py       --raw_dir {raw}
================================================================================
"""


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def try_download(url: str, dest: str) -> bool:
    """Stream ``url`` to ``dest``; return True on success."""
    try:
        import requests
    except ImportError:
        print("  ! `requests` not installed (pip install requests)", file=sys.stderr)
        return False
    try:
        print(f"  -> trying {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(f"\r     {pct:5.1f}%  {_human(done)}/{_human(total)}",
                              end="", flush=True)
            print()
        if os.path.getsize(dest) < 1 << 20:  # < 1 MB is surely an error page
            print("  ! downloaded file suspiciously small, discarding")
            os.remove(dest)
            return False
        return True
    except Exception as e:  # noqa: BLE001 - report and move to next mirror
        print(f"\n  ! failed: {e}")
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        return False


def extract(archive: str, raw_dir: str) -> None:
    print(f"Extracting {archive} -> {raw_dir}")
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        tar.extractall(raw_dir)  # noqa: S202 - trusted archive
    # Flatten: move any *.mat found in subfolders up to raw_dir for convenience.
    for root, _dirs, files in os.walk(raw_dir):
        if os.path.abspath(root) == os.path.abspath(raw_dir):
            continue
        for fn in files:
            if fn.endswith(".mat"):
                src = os.path.join(root, fn)
                dst = os.path.join(raw_dir, fn)
                if not os.path.exists(dst):
                    os.replace(src, dst)
    print(f"  extracted {len(members)} members")


def report_mats(raw_dir: str) -> list[str]:
    mats = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".mat")
    )
    print("\nDetected .mat files:")
    if not mats:
        print("  (none)")
    for m in mats:
        print(f"  {m}   {_human(os.path.getsize(m))}")
    return mats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw/deepsea")
    ap.add_argument("--archive", default=None,
                    help="path for the downloaded/extracted .tar.gz "
                         "(default: <raw_dir>/deepsea_train_bundle.v0.9.tar.gz)")
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    archive = args.archive or os.path.join(
        args.raw_dir, "deepsea_train_bundle.v0.9.tar.gz"
    )

    # If the .mat files already exist, we are done.
    existing = [f for f in os.listdir(args.raw_dir) if f.endswith(".mat")]
    if existing:
        print("Found existing .mat files -- skipping download.")
        report_mats(args.raw_dir)
        return 0

    # If the archive is already present, just extract it.
    if os.path.exists(archive) and os.path.getsize(archive) > (1 << 20):
        print(f"Found existing archive {archive}")
    else:
        ok = False
        for url in URLS:
            if try_download(url, archive):
                ok = True
                break
        if not ok:
            print(MANUAL_MSG.format(archive=archive, raw=args.raw_dir))
            return 1

    extract(archive, args.raw_dir)
    mats = report_mats(args.raw_dir)
    if not mats:
        print("\n! Extraction produced no .mat files.")
        print(MANUAL_MSG.format(archive=archive, raw=args.raw_dir))
        return 1
    print("\nDone. Next:  python scripts/inspect_deepsea_mat.py --raw_dir", args.raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
