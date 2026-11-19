"""Throughput of the vectorised environment.

    nice -n 10 .venv/bin/python -m arena.rl.bench --workers 64 --envs-per-worker 32 --opp random,simple,zoo.base_cautious

For every opponent (a league text, e.g. "random" or "simple:1,self:1") it reports environment steps per second
  env     random actions written by this process, no network: the ceiling given by the workers
  rollout the trainer's real collection loop (Trainer.collect): transfer + GPU inference + sampling + storage
together with `busy` (fraction of the wall time the average worker spent stepping) and, for the rollout, `wait`
(fraction of the time this process was blocked waiting for workers: high = the workers are the bottleneck).
Extra options are passed to the trainer (e.g. --channels 96 --blocks 8 --groups 1 --amp none --device cuda:1).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.rl.ppo import Trainer, build_parser  # noqa: E402
from arena.rl.vecenv import VecEnv  # noqa: E402


def main(argv=None):
    ap = build_parser()
    ap.add_argument("--opp", default="random,simple,zoo.base_cautious", help="leagues to time, separated by ','; use ';' for several")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--no-net", action="store_true", help="only the env ceiling")
    for a in ap._actions:  # --out is not needed here
        if a.dest == "out":
            a.required = False
    args = ap.parse_args(argv)
    args.total_steps = 1e12
    leagues = args.opp.split(";") if ";" in args.opp else args.opp.split(",")
    rows = []
    for league in leagues:
        env = VecEnv(league, workers=args.workers, envs_per_worker=args.envs_per_worker, groups=args.groups, seed=args.seed)
        try:
            rng = np.random.default_rng(0)
            for _ in range(20):
                env.step(rng.integers(0, 6, env.N), rng.integers(0, 6, env.N))
            b0, t0, n = int(env.busy_ns.sum()), time.perf_counter(), 0
            while time.perf_counter() - t0 < args.seconds / 2:
                env.step(rng.integers(0, 6, env.N), rng.integers(0, 6, env.N))
                n += env.N
            dt = time.perf_counter() - t0
            row = {"league": league, "env_sps": n / dt, "env_busy": (int(env.busy_ns.sum()) - b0) / 1e9 / (dt * env.W)}
            if not args.no_net:
                tr = Trainer(args, env)
                tr.collect()  # warm-up
                t0, steps, waits, busy, k = time.perf_counter(), 0, 0.0, 0.0, 0
                while time.perf_counter() - t0 < args.seconds or k < 2:
                    r = tr.collect()
                    steps += env.N * args.rollout
                    waits += r["wait_frac"]
                    busy += r["worker_busy"]
                    k += 1
                dt = time.perf_counter() - t0
                row.update(rollout_sps=steps / dt, wait=waits / k, busy=busy / k)
                del tr
            rows.append(row)
            print(row, flush=True)
        finally:
            env.close()
    print(f"\n{args.workers} workers x {args.envs_per_worker} envs, {args.groups} group(s), net {args.channels}x{args.blocks}, amp {args.amp}")
    print(f"{'league':<34}{'env steps/s':>13}{'busy':>7}{'rollout steps/s':>17}{'busy':>7}{'wait':>7}")
    for r in rows:
        print(f"{r['league']:<34}{r['env_sps']:>13.0f}{r['env_busy']:>7.2f}"
              + (f"{r['rollout_sps']:>17.0f}{r['busy']:>7.2f}{r['wait']:>7.2f}" if "rollout_sps" in r else ""))


if __name__ == "__main__":
    main()
