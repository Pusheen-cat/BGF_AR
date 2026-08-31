#!/usr/bin/env python
"""Compare BGF performance across bgf_lambda values, for DNA RNN and LSTM only.

bgf_lambda is BGF's EMA smoothing factor: the low-frequency gradient buffer is
updated as ``que <- (1 - lambda) * que + lambda * g`` (smaller lambda = slower-
moving buffer). This script sweeps it over {0.001, 0.003, 0.01, 0.03, 0.1} while
**holding bgf_alpha fixed at 0.95** and keeping every other setting identical to
the main experiments. Only RNN and LSTM are covered (as requested); the other
architectures are not part of this comparison.

Runs are scheduled across GPUs (one process per GPU) and are resumable.

    python scripts/reproduce_lambda_rnn_lstm.py --data data/processed/deepsea.h5 \
        --seeds 20 --gpus 0 1 2 3 4 5 6 7

Outputs:  outputs_lambda/<arch>/lam<value>/seed<k>/metrics/run_test_metrics.json
Then summarise with:  python scripts/compare_results.py --mode lambda
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
REPO = HERE.parent
PY = sys.executable

BGF_ALPHA = 0.95                                   # fixed (no hyperparameter search)
LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.1]
ARCHS = {"rnn": 0.4, "lstm": 1.1}                  # arch -> rough per-run cost (hours)


def lam_tag(lam):
    return ("%g" % lam).replace(".", "p")


def build_jobs(data, archs, lambdas, n_seeds, force, outroot):
    jobs = []
    for arch, lam, seed in itertools.product(archs, lambdas, range(n_seeds)):
        cfg = REPO / "configs" / f"{arch}_bgf.yaml"          # BGF config (alpha=0.95)
        rundir = outroot / arch / f"lam{lam_tag(lam)}" / f"seed{seed}"
        ckpt = rundir / "checkpoints" / "run.pt"
        marker = rundir / "metrics" / "run_test_metrics.json"
        train = [PY, str(HERE / "train.py"), "--config", str(cfg),
                 "--data", data, "--name", "run", "--outdir", str(rundir),
                 "--device", "cuda", "--seed", str(seed),
                 "--bgf_alpha", str(BGF_ALPHA), "--bgf_lambda", str(lam)]
        ev = [PY, str(HERE / "evaluate.py"), "--checkpoint", str(ckpt),
              "--data", data, "--split", "test", "--outdir", str(rundir),
              "--device", "cuda", "--name", "run", "--batch_size", "256"]
        shell = subprocess.list2cmdline(train) + " && " + subprocess.list2cmdline(ev)
        jobs.append({"name": f"{arch}.lam{lam_tag(lam)}.s{seed}", "shell": shell,
                     "marker": marker, "cost": ARCHS[arch]})
    jobs.sort(key=lambda j: j["cost"], reverse=True)
    todo = [j for j in jobs if force or not j["marker"].exists()]
    return jobs, todo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=list(ARCHS))
    ap.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--outroot", default=str(REPO / "outputs_lambda"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    outroot = Path(args.outroot)
    logdir = outroot / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    jobs, todo = build_jobs(args.data, args.archs, args.lambdas, args.seeds,
                            args.force, outroot)
    print(f"total={len(jobs)}  to run={len(todo)}  gpus={args.gpus}  "
          f"(BGF alpha={BGF_ALPHA}; lambdas={args.lambdas})")
    if args.dry_run:
        for j in todo:
            print(f"  {j['name']:24s} ~{j['cost']:.1f}h")
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
    print("Next: python scripts/compare_results.py --mode lambda --outroot", outroot)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
