"""arena.bc.noise - how often does a teacher disagree with itself, and what does a student do in those states?

    nice -n 10 .venv/bin/python -m arena.bc.noise --teacher zoo.T [--student <submission folder>] \
        [--pairs "T|T, T|simple"] --games 64 --repeats 200 --procs 16 [--seed0 900000] [--json out.json]

The games are played like in arena.bc.gen (same pair syntax; the "+" actors' states are used, default both). In every
visited state the teacher is asked --repeats times (its answer distribution p_a(state)), the student once (argmax,
exactly as the server plays it). Everything is compared in the real frame. Averages over the states:

    self_agreement      P(two independent teacher answers agree) = E[sum_a p_a^2]
    ceiling             E[max_a p_a]: the best top-1 agreement ANY deterministic network can have with one teacher
                        answer per state. Compare train.py's holdout agreement with THIS number, not with 1.0.
    deterministic       share of the states where all teacher answers are the same
    student_agreement   E[p_student]: the student's expected top-1 agreement (what the holdout measures)
    student_mode        share of the states where the student plays the teacher's most frequent answer
    student_in_support  share of the states where the student plays an action the teacher gave at least once
    off_support         for the other states: rows = the teacher's most frequent action, columns = the student's
    danger              the same shares for the states with a bomb about to hit the player's cell (danger plane > 0)

Use many repeats for a cheap teacher (base_cautious: 200); for a 25 ms search teacher 8-16 is enough to see whether it
is deterministic. Map seeds default to 900000.. (below the evaluation maps; these states are not used for training).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from .common import ACTION_NAMES, EVAL_SEED0, ROOT, parse_pairs, plan_games
from .gen import Seat, Stats, _check_action, game_seed

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _job(job):
    import random

    import torch

    from arena.core import bot_factory
    from bombcompetition import Game, random_rules
    from bombcompetition.neural import preprocess, turn

    map_seed, specs, labelled, teacher, student, repeats = job
    torch.set_num_threads(1)
    seed = game_seed(specs, map_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    st = Stats()
    game = Game(random_rules(map_seed), map_seed)
    seats = [Seat(specs[p], labelled[p], teacher, False, st) for p in (0, 1)]
    net = bot_factory(f"net:{student}")() if student else None
    counts, sacts, danger = [], [], []
    while not game.done:
        both = []
        for p in (0, 1):
            obs = game.obs(p)
            a, ta, _ = seats[p].act(obs)
            both.append(a)
            if ta is None:
                continue
            c = np.zeros(6, dtype=np.int32)
            c[ta] += 1
            for _ in range(repeats - 1):
                c[_check_action(seats[p].tap.bot.act(obs))] += 1
            counts.append(c)
            sacts.append(int(net.act(obs)) if net is not None else -1)
            danger.append(float(preprocess(turn(obs))[5][turn(obs)["me"]]) > 0)
        game.step(*both)
    return (np.array(counts, dtype=np.int32).reshape(-1, 6), np.array(sacts, dtype=np.int64),
            np.array(danger, dtype=bool), st.teacher_errors)


def summarise(counts: np.ndarray, sacts: np.ndarray, has_student: bool) -> dict:
    n = len(counts)
    if n == 0:
        return {"states": 0}
    r = counts.sum(1).astype(np.float64)
    p = counts / r[:, None]
    out = {
        "states": n,
        "self_agreement": float(((counts * (counts - 1)).sum(1) / (r * (r - 1))).mean()),
        "ceiling": float(p.max(1).mean()),
        "deterministic": float((counts.max(1) == r).mean()),
        "teacher_action_share": {a: round(float(p[:, i].mean()), 4) for i, a in enumerate(ACTION_NAMES)},
    }
    if has_student:
        idx = np.arange(n)
        ps = p[idx, sacts]
        mode = counts.argmax(1)
        off = ps == 0
        conf = np.zeros((6, 6), dtype=np.int64)
        np.add.at(conf, (mode[off], sacts[off]), 1)
        out |= {
            "student_agreement": float(ps.mean()),
            "student_mode": float((counts[idx, sacts] == counts.max(1)).mean()),
            "student_in_support": float((~off).mean()),
            "off_support_states": int(off.sum()),
            "off_support": conf.tolist(),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", default=None)
    ap.add_argument("--pairs", default="T|T")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.seed0 < 0 or args.seed0 + args.games > EVAL_SEED0:
        raise SystemExit(f"map seeds must stay below {EVAL_SEED0}")
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2")
    from arena.core import bot_factory

    pairs = parse_pairs(args.pairs, args.teacher, args.student)
    for spec in sorted({args.teacher} | {s for p in pairs for s in (p.a, p.b)}):
        bot_factory(spec)
    plan_pair, plan_swap = plan_games(pairs, args.games, args.seed0)
    jobs = []
    for g in range(args.games):
        pair, swap = pairs[int(plan_pair[g])], bool(plan_swap[g])
        specs = (pair.b, pair.a) if swap else (pair.a, pair.b)
        labelled = (pair.label_b, pair.label_a) if swap else (pair.label_a, pair.label_b)
        jobs.append((args.seed0 + g, specs, labelled, args.teacher, args.student, args.repeats))
    procs = max(1, min(args.procs, args.games))
    with mp.get_context("fork").Pool(procs) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    counts = np.concatenate([r[0] for r in res])
    sacts = np.concatenate([r[1] for r in res])
    danger = np.concatenate([r[2] for r in res])
    out = {"teacher": args.teacher, "student": args.student, "pairs": args.pairs, "games": args.games,
           "repeats": args.repeats, "seed0": args.seed0, "teacher_errors": int(sum(r[3] for r in res)),
           **summarise(counts, sacts, bool(args.student)),
           "danger": summarise(counts[danger], sacts[danger], bool(args.student)),
           "per_pair": {}}
    for i, p in enumerate(pairs):
        sel = [k for k in range(args.games) if int(plan_pair[k]) == i]
        if sel:
            out["per_pair"][p.key] = summarise(np.concatenate([res[k][0] for k in sel]),
                                               np.concatenate([res[k][1] for k in sel]), bool(args.student))

    def line(name, d):
        if not d.get("states"):
            return f"{name:<44} no states"
        s = (f"{name:<44}{d['states']:>8} states | teacher: self-agreement {d['self_agreement']:.4f}, ceiling "
             f"{d['ceiling']:.4f}, deterministic {d['deterministic']:.4f}")
        if "student_agreement" in d:
            s += (f" | student: agreement {d['student_agreement']:.4f}, plays the mode {d['student_mode']:.4f}, "
                  f"in support {d['student_in_support']:.4f} ({d['off_support_states']} off)")
        return s

    print(f"teacher {args.teacher} asked {args.repeats}x per state" + (f", student {args.student}" if args.student else ""))
    print(line("all states", out))
    print(line("states with a bomb timed on my cell", out["danger"]))
    for k, d in out["per_pair"].items():
        print(line(k[:43], d))
    if args.student and out.get("off_support_states"):
        print("off-support states: rows = the teacher's most frequent action, columns = the student's action "
              f"({' '.join(ACTION_NAMES)})")
        for a, row in zip(ACTION_NAMES, out["off_support"]):
            print(f"  {a:<6}" + "".join(f"{v:>7}" for v in row))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
