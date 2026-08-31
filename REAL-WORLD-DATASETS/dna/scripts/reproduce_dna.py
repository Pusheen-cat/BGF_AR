#!/usr/bin/env python
"""Reproduce all reported DNA (DeepSEA) experiments.

For each architecture {rnn, lstm, stack_rnn, tape_rnn} and each optimizer
{adamw, bgf} it trains + evaluates one run per seed. BGF is always run with
**bgf_alpha = 0.95** (fixed; used without hyperparameter search) and
bgf_lambda = 0.01. AdamW is the paired baseline (same seeds, same config, only
the optimizer differs).

Stack-RNN and Tape-RNN additionally use gradient clipping (0.5) and the
optimizer's NaN-skipping guard (a step with any non-finite gradient is skipped);
both are already set in their configs / the optimizer.

Runs are scheduled across the given GPUs (one process per GPU) and are
**resumable**: a run whose test-metrics JSON already exists is skipped.

    # reproduce everything on 8 GPUs (reported seed counts: RNN/LSTM=20, Stack/Tape=5)
    python scripts/reproduce_dna.py --data data/processed/deepsea.h5 --gpus 0 1 2 3 4 5 6 7

    # a quick check: 2 seeds for every architecture on 2 GPUs
    python scripts/reproduce_dna.py --data ... --seeds 2 --gpus 0 1

Outputs:  outputs/<arch>_<optimizer>/seed<k>/metrics/run_test_metrics.json
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                      # code_release/dna
PY = sys.executable

BGF_ALPHA = 0.95                        # fixed for every BGF run (no hyperparameter search)
BGF_LAMBDA = 0.01

# architecture -> reported number of seeds + rough per-run cost (hours) for scheduling
ARCHS = {
    "rnn":       {"seeds": 20, "cost": 0.4},
    "lstm":      {"seeds": 20, "cost": 1.1},
    "stack_rnn": {"seeds": 5,  "cost": 19.0},
    "tape_rnn":  {"seeds": 5,  "cost": 15.0},
}
OPTIMIZERS = ["adamw", "bgf"]


def build_jobs(data, archs, optimizers, seed_override, force, outroot):
    jobs = []
    for arch in archs:
        n_seeds = seed_override if seed_override else ARCHS[arch]["seeds"]
        for opt, seed in itertools.product(optimizers, range(n_seeds)):
            cfg = REPO / "configs" / f"{arch}_{opt}.yaml"
            rundir = outroot / f"{arch}_{opt}" / f"seed{seed}"
            ckpt = rundir / "checkpoints" / "run.pt"
            marker = rundir / "metrics" / "run_test_metrics.json"
            train = [PY, str(HERE / "train.py"), "--config", str(cfg),
                     "--data", data, "--name", "run", "--outdir", str(rundir),
                     "--device", "cuda", "--seed", str(seed)]
            if opt == "bgf":                     # make alpha=0.95 explicit on the command line
                train += ["--bgf_alpha", str(BGF_ALPHA), "--bgf_lambda", str(BGF_LAMBDA)]
            ev = [PY, str(HERE / "evaluate.py"), "--checkpoint", str(ckpt),
                  "--data", data, "--split", "test", "--outdir", str(rundir),
                  "--device", "cuda", "--name", "run", "--batch_size", "256"]
            shell = subprocess.list2cmdline(train) + " && " + subprocess.list2cmdline(ev)
            jobs.append({"name": f"{arch}_{opt}.s{seed}", "shell": shell,
                         "marker": marker, "cost": ARCHS[arch]["cost"]})
    jobs.sort(key=lambda j: j["cost"], reverse=True)          # big jobs first
    todo = [j for j in jobs if force or not j["marker"].exists()]
    return jobs, todo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="processed DeepSEA .h5 (see prepare_hdf5.py)")
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=list(ARCHS))
    ap.add_argument("--optimizers", nargs="+", default=OPTIMIZERS, choices=OPTIMIZERS)
    ap.add_argument("--seeds", type=int, default=None,
                    help="seed count for ALL archs (default: reported per-arch counts)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--outroot", default=str(REPO / "outputs"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    outroot = Path(args.outroot)
    logdir = outroot / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    jobs, todo = build_jobs(args.data, args.archs, args.optimizers,
                            args.seeds, args.force, outroot)
    print(f"total={len(jobs)}  to run={len(todo)}  gpus={args.gpus}  "
          f"(BGF alpha={BGF_ALPHA}, lambda={BGF_LAMBDA})")
    if args.dry_run:
        for j in todo:
            print(f"  {j['name']:22s} ~{j['cost']:.1f}h")
        return 0
    if not todo:
        print("nothing to do (all runs complete).")
        return 0

    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    free = list(args.gpus)
    running, queue = {}, list(todo)
    n_ok = n_fail = 0
    t0 = time.time()
    while queue or running:
        while queue and free:
            gpu = free.pop()
            job = queue.pop(0)
            lf = open(logdir / f"{job['name']}.log", "w")
            e = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            job["proc"], job["lf"] = subprocess.Popen(
                ["bash", "-lc", job["shell"]], cwd=str(REPO),
                stdout=lf, stderr=subprocess.STDOUT, env=e), lf
            running[gpu] = job
            print(f"[{(time.time()-t0)/3600:5.2f}h] launch gpu{gpu} {job['name']}", flush=True)
        for gpu, job in list(running.items()):
            rc = job["proc"].poll()
            if rc is not None:
                job["lf"].close()
                ok = rc == 0 and job["marker"].exists()
                n_ok += ok; n_fail += (not ok)
                print(f"[{(time.time()-t0)/3600:5.2f}h] done   gpu{gpu} {job['name']} "
                      f"{'ok' if ok else f'FAIL(rc={rc})'} [{n_ok+n_fail}/{len(todo)}]", flush=True)
                del running[gpu]; free.append(gpu)
        time.sleep(3.0)
    print(f"\nFinished: ok={n_ok} fail={n_fail} in {(time.time()-t0)/3600:.2f} h")
    print("Next: python scripts/compare_results.py --outroot", outroot)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
