#!/usr/bin/env python
"""Reproduce the SMILES / ChemBERTa AdamW-vs-BGF experiments.

Fine-tunes each of the three pretrained ChemBERTa encoders on four MoleculeNet
datasets under a scaffold split, with two optimizers:

    ChemBERTa-1 = DeepChem/ChemBERTa-77M-MTR   (77M params, multi-task-regression pretraining)
    ChemBERTa-2 = DeepChem/ChemBERTa-77M-MLM   (77M params, masked-language-model pretraining)
    ChemBERTa-3 = DeepChem/ChemBERTa-10M-MTR   (10M params, multi-task-regression pretraining)

    datasets: esol, bbbp (10 seeds), hiv, tox21 (5 seeds)
    optimizers: adamw (baseline)  vs  bgf (bgf_alpha = 0.95, fixed; no hyperparameter
                search; bgf_lambda = 0.01). Same seeds, same config, only the
                optimizer differs.

Each run drives ``python -m smiles_pp.study`` (test metric taken at the best-
validation epoch). Runs are scheduled across GPUs and are resumable (a run whose
final.json exists is skipped).

    python scripts/reproduce_smiles.py --gpus 0 1 2 3 4 5 6 7

Outputs:  outputs/<task>/<variant>/<optimizer>/seed<k>/final.json
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                  # code_release/smiles
PY = os.environ.get("SMILES_PY", "python")         # interpreter with smiles_pp installed/importable

BGF_ALPHA = 0.95                                    # fixed for every BGF run (no hyperparameter search)
BGF_LAMBDA = 0.01
VARIANTS = ["chemberta", "chemberta2", "chemberta3"]   # settings 1, 2, 3
# dataset -> reported number of seeds + rough per-run cost (hours) for scheduling
TASKS = {"esol": {"seeds": 10, "cost": 0.15}, "bbbp": {"seeds": 10, "cost": 0.15},
         "hiv": {"seeds": 5, "cost": 1.2}, "tox21": {"seeds": 5, "cost": 0.5}}
VARIANT_MULT = {"chemberta": 1.0, "chemberta2": 1.0, "chemberta3": 0.5}
OPTIMIZERS = ["adamw", "bgf"]


def build_jobs(tasks, variants, optimizers, seed_override, force, outroot):
    jobs = []
    for task, variant, opt in itertools.product(tasks, variants, optimizers):
        n_seeds = seed_override if seed_override else TASKS[task]["seeds"]
        for seed in range(n_seeds):
            rundir = outroot / task / variant / opt / f"seed{seed}"
            marker = rundir / "final.json"
            cmd = [PY, "-m", "smiles_pp.study", "--task", task, "--model", variant,
                   "--optimizer", opt, "--seed", str(seed), "--outdir", str(rundir),
                   "--device", "cuda"]
            if opt == "bgf":                        # make alpha=0.95 explicit on the command line
                cmd += ["--bgf-alpha", str(BGF_ALPHA), "--bgf-lambda", str(BGF_LAMBDA)]
            jobs.append({"name": f"{task}.{variant}.{opt}.s{seed}", "cmd": cmd,
                         "marker": marker, "cost": TASKS[task]["cost"] * VARIANT_MULT[variant]})
    jobs.sort(key=lambda j: j["cost"], reverse=True)
    todo = [j for j in jobs if force or not j["marker"].exists()]
    return jobs, todo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--optimizers", nargs="+", default=OPTIMIZERS, choices=OPTIMIZERS)
    ap.add_argument("--seeds", type=int, default=None,
                    help="seed count for ALL tasks (default: reported per-task counts)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--outroot", default=str(REPO / "outputs"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    outroot = Path(args.outroot)
    logdir = outroot / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    jobs, todo = build_jobs(args.tasks, args.variants, args.optimizers,
                            args.seeds, args.force, outroot)
    print(f"total={len(jobs)}  to run={len(todo)}  gpus={args.gpus}  "
          f"(BGF alpha={BGF_ALPHA}, lambda={BGF_LAMBDA})")
    if args.dry_run:
        for j in todo:
            print(f"  {j['name']:34s} ~{j['cost']:.2f}h")
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
                job["cmd"], cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT, env=e), lf
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
