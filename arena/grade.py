"""Local grading: play a candidate the way the leaderboard does (100 games per opponent, every map from both sides,
1 point per win) against the opponent panels of arena/panel.py.

    nice -n 10 .venv/bin/python -m arena.grade net:runs/some_run/best                  # validation panel, 100 games each
    nice -n 10 .venv/bin/python -m arena.grade net:runs/some_run/best --panel train+val --games 200
    nice -n 10 .venv/bin/python -m arena.grade zoo.x_solver --panel train --games 100   # bots can be graded too
    nice -n 10 .venv/bin/python -m arena.grade net:runs/a/best --opp zoo.ta_smart_a,simple --games 400

The summary line is the number to compare experiments with: `score` = mean wins per 100 games over the opponents
(what the leaderboard would show against an average panel member), plus the worst opponent and the loss/draw rates.
Evaluation maps are random_rules(1_000_000 + k) (or --seed0), never used for training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arena.core import DEFAULT_PROCS, EVAL_SEED0, format_leaderboard, leaderboard
from arena.panel import panel


def grade(me: str, opponents: list[str], games: int = 100, seed0: int = EVAL_SEED0, procs: int = DEFAULT_PROCS,
          verbose: bool = True) -> dict:
    out = leaderboard(me, opponents, games, seed0, procs, verbose=False)
    per = out["per_opp"]
    worst = min(per, key=lambda o: per[o]["win_rate"])
    out["score"] = 100 * out["mean_win_rate"]
    out["worst_opp"], out["worst_score"] = worst, 100 * per[worst]["win_rate"]
    out["self_bomb_rate"] = sum(t["self_bomb"] for t in per.values()) / max(1, sum(t["games"] for t in per.values()))
    out["timeout_rate"] = sum(t["timeout_draw"] for t in per.values()) / max(1, sum(t["games"] for t in per.values()))
    if verbose:
        print(format_leaderboard(out))
        print(f"SCORE {out['score']:.1f} wins/100 (mean over {len(per)} opponents) | worst {out['worst_score']:.1f} vs {worst} | "
              f"losses/100 {100 * out['mean_loss_rate']:.1f} | draws/100 {100 * out['mean_draw_rate']:.1f} | "
              f"self-bombs/100 {100 * out['self_bomb_rate']:.1f} | 300-step draws/100 {100 * out['timeout_rate']:.1f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("me", help="bot spec, e.g. net:runs/x/best")
    ap.add_argument("--panel", default="val", help="baselines | train | val | test | experts | all, or several joined by +")
    ap.add_argument("--opp", default=None, help="comma-separated specs (overrides --panel)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    opponents = [s for s in args.opp.split(",") if s] if args.opp else panel(args.panel)
    if not opponents:
        raise SystemExit(f"no opponents exist for panel {args.panel!r}")
    out = grade(args.me, opponents, args.games, args.seed0, args.procs)
    out["panel"], out["time"] = args.panel if not args.opp else "custom", time.strftime("%Y-%m-%d %H:%M:%S")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
