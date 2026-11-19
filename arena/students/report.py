"""Tables for the simulated student networks: leaderboard rows, the round robin, and how each network behaves.

    nice -n 10 .venv/bin/python -m arena.students.report            # reads runs/students/_results/*.json
    nice -n 10 .venv/bin/python -m arena.students.report --profile  # also plays recorded games for the behaviour table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "runs" / "students" / "_results"
OPPS = ["random", "simple", "zoo.base_cautious"]


def names() -> list[str]:
    root = ROOT / "runs" / "students"
    out = sorted(p.name for p in root.glob("train_*") if (p / "submission.pth").exists())
    return out + sorted(p.name for p in root.glob("test_*") if (p / "submission.pth").exists())


def short(name: str) -> str:
    kind, k = name.split("_")[:2]
    return ("tr" if kind == "train" else "te") + k


def leaderboard_table() -> str:
    lines = [f"{'network':<28}" + "".join(f"{'vs ' + o.replace('zoo.base_', ''):>22}" for o in OPPS) + f"{'wins/300':>10}",
             f"{'':<28}" + f"{'W-D-L (self|t/o)':>22}" * 3]
    for n in names():
        f = RES / f"{n}.leaderboard.json"
        if not f.exists():
            lines.append(f"{n:<28} (not graded)")
            continue
        per = json.loads(f.read_text())["per_opp"]
        row, total = f"{n:<28}", 0
        for o in OPPS:
            t = per[o]
            total += t["win"]
            row += f"{'%d-%d-%d (%d|%d)' % (t['win'], t['draw'], t['loss'], t['self_bomb'], t['timeout_draw']):>22}"
        lines.append(row + f"{total:>10}")
    lines.append("W-D-L over 100 games (each map from both sides); self = games lost to its own bomb, t/o = 300-step draws")
    return "\n".join(lines)


def round_robin_table() -> str:
    f = RES / "round_robin.json"
    if not f.exists():
        return "(no round robin yet)"
    out = json.loads(f.read_text())
    bots, W, D, stats = out["bots"], out["wins"], out["draws"], out["stats"]
    nm = [b.split("/")[-1] for b in bots]
    order = [bots.index(b) for b in out["ranking"]]
    lines = [f"round robin between the {len(bots)} student networks, {out['games']} games per pair "
             f"(cell = row's wins-draws-losses against the column)",
             f"{'#':>2} {'network':<28}{'W':>5}{'D':>5}{'L':>5}{'self':>6}  " + " ".join(f"{short(nm[j]):>9}" for j in order)]
    for rank, i in enumerate(order, 1):
        s = stats[bots[i]]
        cells = " ".join(f"{'-':>9}" if i == j else f"{f'{W[i][j]}-{D[i][j]}-{W[j][i]}':>9}" for j in order)
        lines.append(f"{rank:>2} {nm[i]:<28}{s['win']:>5}{s['draw']:>5}{s['loss']:>5}{s['self_bomb']:>6}  {cells}")
    return "\n".join(lines)


def profile(games: int = 40, procs: int = 16) -> str:
    """Recorded games against simple and base_cautious: what the network does (actions) and how its games end."""
    from arena.core import EVAL_SEED0, run_games

    opps = ["simple", "zoo.base_cautious"]
    jobs, keys = [], []
    for n in names():
        me = f"net:runs/students/{n}"
        for o in opps:
            for i in range(games):
                side, seed = i % 2, EVAL_SEED0 + 500 + i // 2
                jobs.append((me, o, seed, i, True) if side == 0 else (o, me, seed, i, True))
                keys.append((n, o, side))
    results = run_games(jobs, procs)
    prof: dict = {}
    for (n, o, side), r in zip(keys, results):
        p = prof.setdefault(n, {"acts": [0] * 6, "steps": 0, "games": 0, "kill": 0, "gift": 0, "selfbomb": 0, "killed": 0,
                                "timeout": 0, "both": 0})
        if "crash" in r:
            continue
        p["games"] += 1
        p["steps"] += r["steps"]
        for step in r["actions"]:
            p["acts"][step[side]] += 1
        w = r["winner"]
        if w == side:
            p["kill" if r["kills"][side] else "gift"] += 1
        elif w == 1 - side:
            p["selfbomb" if r["self_bomb"][side] else "killed"] += 1
        else:
            p["timeout" if min(r["lives"]) > 0 else "both"] += 1
    (RES / "behaviour.json").write_text(json.dumps(prof, indent=1))
    lines = [f"behaviour in {2 * games} recorded games ({games} vs simple, {games} vs base_cautious)",
             f"{'network':<28}{'STAY':>6}{'MOVE':>6}{'BOMB':>6}{'steps':>7}{'bombs/g':>8}   how its games end",
             f"{'':<28}{'% of actions':>18}"]
    for n, p in prof.items():
        tot = max(1, sum(p["acts"]))
        g = max(1, p["games"])
        lines.append(f"{n:<28}{100 * p['acts'][0] / tot:>6.0f}{100 * sum(p['acts'][1:5]) / tot:>6.0f}{100 * p['acts'][5] / tot:>6.0f}"
                     f"{p['steps'] / g:>7.0f}{p['acts'][5] / g:>8.1f}   "
                     f"kills {p['kill']}, opp self-bombs {p['gift']}, own self-bombs {p['selfbomb']}, killed {p['killed']}, "
                     f"300-step draws {p['timeout']}, both hit {p['both']}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    print(leaderboard_table())
    print()
    print(round_robin_table())
    if args.profile:
        print()
        print(profile(args.games, args.procs))
