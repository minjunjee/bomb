"""Local grading harness for the BOMB competition (shared by every experiment).

Everything here plays games with the REFERENCE engine (`bombcompetition`), exactly as the grading server does:
neural candidates are loaded from a folder holding `submission.py` + `submission.pth` and played through the
fixed `NeuralPolicy` (turn -> preprocess -> Net -> argmax).

Bot specs (strings)
-------------------
    random                 a random legal action (the notebook's RandomBot)
    simple                 bots/simple.py (the notebook's SimpleBot)
    zoo.<name>             arena/zoo/<name>.py, class `Policy` with `act(obs) -> int`  (a new instance every game)
    zoo.<name>:<arg>       arena/zoo/<name>.py, `make_policy(arg)` -> a zero-argument factory (parametrised bot families)
    noisy:<eps>:<spec>     <spec>, but with probability eps a random legal action instead
    path/to/file.py        any python file with a `Policy` class
    net:<folder>           <folder>/submission.py (class Net) + <folder>/submission.pth, played with NeuralPolicy
    net:<folder>@sample    the same network, but sampling from softmax(scores) instead of argmax (training only)

CLI
---
    python -m arena.core leaderboard --me net:runs/x --opp simple,zoo.foo --games 100
    python -m arena.core tournament --bots simple,zoo.a,zoo.b --games 50
    python -m arena.core gif --a zoo.a --b simple --map 1000 --out /tmp/x.gif

Seeds: a game is fully determined by (spec_a, spec_b, map_seed, game_seed). Evaluation maps start at 1_000_000
(never use seeds >= 1_000_000 for training).
"""

from __future__ import annotations

import os

# One thread per process: games are played in many worker processes (like the 1-CPU grading server).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib
import importlib.util
import json
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bombcompetition import RULES, Game, legal_actions, random_rules  # noqa: E402

EVAL_SEED0 = 1_000_000  # evaluation maps: random_rules(EVAL_SEED0 + k); training must stay below this
DEFAULT_PROCS = min(96, max(1, (os.cpu_count() or 2) - 2))


# ----------------------------------------------------------------------------------------------------------------
# bots
# ----------------------------------------------------------------------------------------------------------------
class RandomBot:
    """A random action that is not blocked (the notebook's RandomBot)."""

    def act(self, obs):
        return int(random.choice(legal_actions(obs)))


_FACTORIES: dict[str, object] = {}


def _load_module(path: Path):
    path = Path(path).resolve()
    name = f"_arena_mod_{zlib.crc32(str(path).encode())}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _net_factory(folder: str, sample: bool):
    import torch

    from bombcompetition.neural import NeuralPolicy, preprocess, scores, turn_action

    torch.set_num_threads(1)
    folder = Path(folder)
    if not folder.is_absolute():
        folder = ROOT / folder
    module = _load_module(folder / "submission.py")
    net = module.Net()
    net.load_state_dict(torch.load(folder / "submission.pth", map_location="cpu"))
    net.eval()  # the server check does NeuralPolicy(net.eval())
    if not sample:
        policy = NeuralPolicy(net)
        return lambda: policy  # stateless: one shared instance per process

    class Sampler:
        def act(self, obs):
            with torch.no_grad():
                p = torch.softmax(scores(net, preprocess, obs), 0)
                return turn_action(int(torch.multinomial(p, 1)), obs)

    return Sampler


def _noisy_factory(eps: float, inner):
    class Noisy:
        def __init__(self):
            self.bot = inner()

        def act(self, obs):
            a = self.bot.act(obs)  # always asked, so a stateful bot keeps its state
            return int(random.choice(legal_actions(obs))) if random.random() < eps else a

    return Noisy


def bot_factory(spec: str):
    """spec -> a zero-argument callable that makes a new bot (an object with act(obs) -> int)."""
    if spec in _FACTORIES:
        return _FACTORIES[spec]
    if spec == "random":
        f = RandomBot
    elif spec == "simple":
        from bots.simple import Policy as f
    elif spec.startswith("net:"):
        body = spec[4:]
        sample = body.endswith("@sample")
        f = _net_factory(body[: -len("@sample")] if sample else body, sample)
    elif spec.startswith("noisy:"):  # noisy:<eps>:<spec> = <spec>, but a random legal action with probability eps
        _, eps, inner = spec.split(":", 2)
        f = _noisy_factory(float(eps), bot_factory(inner))
    elif spec.startswith("zoo."):
        name, _, arg = spec[4:].partition(":")
        module = importlib.import_module(f"arena.zoo.{name}")
        f = module.make_policy(arg) if arg else module.Policy  # zoo.<name>:<arg> -> module.make_policy(arg) = a factory
    elif spec.endswith(".py"):
        p = Path(spec)
        f = _load_module(p if p.is_absolute() else ROOT / p).Policy
    else:
        raise ValueError(f"unknown bot spec: {spec!r}")
    _FACTORIES[spec] = f
    return f


# ----------------------------------------------------------------------------------------------------------------
# one game
# ----------------------------------------------------------------------------------------------------------------
def play_game(spec_a: str, spec_b: str, map_seed: int, game_seed: int = 0, record: bool = False) -> dict:
    """One game on random_rules(map_seed): spec_a plays A (player 0), spec_b plays B (player 1).
    Bots are run like the server's SafeBot: an exception or an invalid action = STAY + a fault."""
    random.seed(zlib.crc32(f"{spec_a}|{spec_b}|{map_seed}|{game_seed}".encode()))
    rules = random_rules(map_seed)
    game = Game(rules, game_seed)
    bots, faults, errors, spent, slowest = [], [0, 0], [[], []], [0.0, 0.0], [0.0, 0.0]
    for p, spec in enumerate((spec_a, spec_b)):
        try:
            bots.append(bot_factory(spec)())
        except Exception as e:  # noqa: BLE001
            bots.append(None)
            faults[p] += 1
            errors[p].append(f"init: {e!r}")
    actions = []
    while not game.done:
        step = []
        for p in (0, 1):
            a = 0
            if bots[p] is not None:
                t0 = time.perf_counter()
                try:
                    out = bots[p].act(game.obs(p))
                    if isinstance(out, bool) or not hasattr(out, "__index__") or not 0 <= int(out) <= 5:
                        raise ValueError(f"act returned {out!r}")
                    a = int(out)
                except Exception as e:  # noqa: BLE001
                    faults[p] += 1
                    if len(errors[p]) < 3:
                        errors[p].append(repr(e))
                dt = time.perf_counter() - t0
                spent[p] += dt
                slowest[p] = max(slowest[p], dt)
            step.append(a)
        game.step(*step)
        actions.append(step)
    w = game.winner()
    # self-bomb: lost a life although the opponent got no kill
    self_bomb = [int(RULES.lives - game.lives[p] - game.kills[1 - p] > 0) for p in (0, 1)]
    out = {
        "a": spec_a, "b": spec_b, "map_seed": map_seed, "game_seed": game_seed,
        "winner": w, "steps": game.t, "lives": list(game.lives), "kills": list(game.kills),
        "self_bomb": self_bomb, "faults": faults, "errors": errors,
        "ms_per_act": [1000 * spent[p] / max(1, game.t) for p in (0, 1)], "slowest_ms": [1000 * s for s in slowest],
    }
    if record:
        out["actions"] = actions
    return out


def _job(args):
    try:
        return play_game(*args)
    except Exception as e:  # noqa: BLE001  (a broken bot must not kill the whole evaluation)
        return {"a": args[0], "b": args[1], "map_seed": args[2], "game_seed": args[3], "crash": repr(e)}


def _init_worker():
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def run_games(jobs: list[tuple], procs: int = DEFAULT_PROCS, chunksize: int | None = None) -> list[dict]:
    """jobs: [(spec_a, spec_b, map_seed, game_seed[, record])] -> results in the same order."""
    if procs <= 1 or len(jobs) <= 2:
        return [_job(j) for j in jobs]
    import multiprocessing as mp

    procs = min(procs, len(jobs))
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=_init_worker) as pool:
        return pool.map(_job, jobs, chunksize=chunksize or max(1, len(jobs) // (procs * 4)))


# ----------------------------------------------------------------------------------------------------------------
# leaderboard: "me" against a list of opponents, the way the server grades
# ----------------------------------------------------------------------------------------------------------------
def _series_jobs(me: str, opp: str, games: int, seed0: int, record: bool = False):
    jobs = []
    for i in range(games):
        side, map_seed = i % 2, seed0 + i // 2  # A in even games, B in odd games; each map from both sides
        jobs.append((me, opp, map_seed, i, record) if side == 0 else (opp, me, map_seed, i, record))
    return jobs


def _tally(results: list[dict], games: int) -> dict:
    t = {"games": 0, "win": 0, "draw": 0, "loss": 0, "self_bomb": 0, "opp_self_bomb": 0, "timeout_draw": 0,
         "faults": 0, "opp_faults": 0, "crashes": 0, "steps": 0, "ms_per_act": 0.0, "slowest_ms": 0.0,
         "win_as_A": 0, "win_as_B": 0, "errors": []}
    for i, r in enumerate(results):
        if "crash" in r:
            t["crashes"] += 1
            t["errors"].append(r["crash"])
            continue
        side = i % 2
        w = r["winner"]
        t["games"] += 1
        t["win"] += w == side
        t["draw"] += w is None
        t["loss"] += w == 1 - side
        t["win_as_A" if side == 0 else "win_as_B"] += w == side
        t["self_bomb"] += r["self_bomb"][side]
        t["opp_self_bomb"] += r["self_bomb"][1 - side]
        t["timeout_draw"] += w is None and min(r["lives"]) > 0
        t["faults"] += r["faults"][side]
        t["opp_faults"] += r["faults"][1 - side]
        t["steps"] += r["steps"]
        t["ms_per_act"] += r["ms_per_act"][side]
        t["slowest_ms"] = max(t["slowest_ms"], r["slowest_ms"][side])
        for e in r["errors"][side]:
            if len(t["errors"]) < 5 and e not in t["errors"]:
                t["errors"].append(e)
    n = max(1, t["games"])
    t["avg_steps"] = t.pop("steps") / n
    t["ms_per_act"] = t["ms_per_act"] / n
    t["win_rate"], t["loss_rate"], t["draw_rate"] = t["win"] / n, t["loss"] / n, t["draw"] / n
    return t


def leaderboard(me: str, opponents: list[str], games: int = 100, seed0: int = EVAL_SEED0,
                procs: int = DEFAULT_PROCS, verbose: bool = True) -> dict:
    """`games` games against every opponent (maps seed0.., each map from both sides).
    Returns {"per_opp": {opp: tally}, "mean_win_rate": ..., "min_win_rate": ..., "mean_loss_rate": ...}."""
    jobs = []
    for opp in opponents:
        jobs += _series_jobs(me, opp, games, seed0)
    t0 = time.time()
    results = run_games(jobs, procs)
    per = {opp: _tally(results[k * games:(k + 1) * games], games) for k, opp in enumerate(opponents)}
    out = {
        "me": me, "games": games, "seed0": seed0, "per_opp": per,
        "mean_win_rate": sum(t["win_rate"] for t in per.values()) / len(per),
        "min_win_rate": min(t["win_rate"] for t in per.values()),
        "mean_loss_rate": sum(t["loss_rate"] for t in per.values()) / len(per),
        "mean_draw_rate": sum(t["draw_rate"] for t in per.values()) / len(per),
        "faults": sum(t["faults"] for t in per.values()),
        "seconds": time.time() - t0,
    }
    if verbose:
        print(format_leaderboard(out))
    return out


def format_leaderboard(out: dict) -> str:
    lines = [f"leaderboard for {out['me']}  ({out['games']} games per opponent, maps from seed {out['seed0']})",
             f"{'opponent':<34}{'W':>5}{'D':>5}{'L':>5}{'self':>6}{'oppself':>8}{'t/o':>5}{'flt':>5}{'steps':>7}{'ms/act':>8}{'max ms':>8}"]
    for opp, t in out["per_opp"].items():
        lines.append(f"{opp:<34}{t['win']:>5}{t['draw']:>5}{t['loss']:>5}{t['self_bomb']:>6}{t['opp_self_bomb']:>8}"
                     f"{t['timeout_draw']:>5}{t['faults']:>5}{t['avg_steps']:>7.1f}{t['ms_per_act']:>8.2f}{t['slowest_ms']:>8.1f}"
                     + (f"  CRASHES {t['crashes']}" if t["crashes"] else ""))
    lines.append(f"mean win rate {out['mean_win_rate']:.3f} | min win rate {out['min_win_rate']:.3f} | "
                 f"mean loss rate {out['mean_loss_rate']:.3f} | mean draw rate {out['mean_draw_rate']:.3f} | "
                 f"faults {out['faults']} | {out['seconds']:.0f}s")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# tournament: round robin between bots
# ----------------------------------------------------------------------------------------------------------------
def tournament(bots: list[str], games: int = 50, seed0: int = EVAL_SEED0, procs: int = DEFAULT_PROCS,
               verbose: bool = True) -> dict:
    """Every pair plays `games` games (each map from both sides). Returns win/draw/loss matrices and a ranking by
    points (1 per win, like the arena), ties broken by fewer losses."""
    pairs = [(i, j) for i in range(len(bots)) for j in range(i + 1, len(bots))]
    jobs = []
    for i, j in pairs:
        jobs += _series_jobs(bots[i], bots[j], games, seed0)
    t0 = time.time()
    results = run_games(jobs, procs)
    n = len(bots)
    W = [[0] * n for _ in range(n)]
    D = [[0] * n for _ in range(n)]
    stats = {b: {"win": 0, "draw": 0, "loss": 0, "self_bomb": 0, "faults": 0, "crashes": 0, "ms_per_act": 0.0,
                 "slowest_ms": 0.0, "games": 0} for b in bots}
    for k, (i, j) in enumerate(pairs):
        for g, r in enumerate(results[k * games:(k + 1) * games]):
            if "crash" in r:
                stats[bots[i]]["crashes"] += 1
                stats[bots[j]]["crashes"] += 1
                continue
            side_i = g % 2  # bots[i] is A in even games
            w = r["winner"]
            if w is None:
                D[i][j] += 1
                D[j][i] += 1
            elif w == side_i:
                W[i][j] += 1
            else:
                W[j][i] += 1
            for idx, side in ((i, side_i), (j, 1 - side_i)):
                s = stats[bots[idx]]
                s["games"] += 1
                s["self_bomb"] += r["self_bomb"][side]
                s["faults"] += r["faults"][side]
                s["ms_per_act"] += r["ms_per_act"][side]
                s["slowest_ms"] = max(s["slowest_ms"], r["slowest_ms"][side])
    for i, b in enumerate(bots):
        s = stats[b]
        s["win"] = sum(W[i])
        s["loss"] = sum(W[j][i] for j in range(n))
        s["draw"] = sum(D[i])
        s["ms_per_act"] /= max(1, s["games"])
    order = sorted(range(n), key=lambda i: (-stats[bots[i]]["win"], stats[bots[i]]["loss"]))
    out = {"bots": bots, "games": games, "seed0": seed0, "wins": W, "draws": D, "stats": stats,
           "ranking": [bots[i] for i in order], "seconds": time.time() - t0}
    if verbose:
        print(format_tournament(out))
    return out


def format_tournament(out: dict) -> str:
    bots, W, D, stats, games = out["bots"], out["wins"], out["draws"], out["stats"], out["games"]
    order = [bots.index(b) for b in out["ranking"]]
    short = [b.replace("zoo.", "")[:9] for b in bots]
    lines = [f"round robin, {games} games per pair (cell = row's wins-draws-losses vs column)",
             f"{'bot':<30}{'W':>6}{'D':>6}{'L':>6}{'self':>6}{'flt':>5}{'ms/act':>8}{'max ms':>8}   " +
             " ".join(f"{short[j]:>9}" for j in order)]
    for i in order:
        s = stats[bots[i]]
        cells = " ".join(f"{'-':>9}" if i == j else f"{f'{W[i][j]}-{D[i][j]}-{W[j][i]}':>9}" for j in order)
        lines.append(f"{bots[i]:<30}{s['win']:>6}{s['draw']:>6}{s['loss']:>6}{s['self_bomb']:>6}{s['faults']:>5}"
                     f"{s['ms_per_act']:>8.2f}{s['slowest_ms']:>8.1f}   {cells}"
                     + (f"  CRASHES {s['crashes']}" if s["crashes"] else ""))
    lines.append(f"{out['seconds']:.0f}s")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# replay as a GIF (to watch a game)
# ----------------------------------------------------------------------------------------------------------------
def save_gif(result: dict, path: str | Path, tile: int = 40, fps: int = 6):
    """result: play_game(..., record=True). The engine is deterministic given the map, the seed and the actions."""
    from PIL import Image

    from bombcompetition.render import to_rgb

    game = Game(random_rules(result["map_seed"]), result["game_seed"])
    frames = [to_rgb(game, tile)]
    for a in result["actions"]:
        game.step(*a)
        frames.append(to_rgb(game, tile))
    frames += [frames[-1]] * 8
    imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    imgs[0].save(path, format="GIF", save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0, optimize=True)


# ----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("leaderboard")
    lb.add_argument("--me", required=True)
    lb.add_argument("--opp", required=True, help="comma-separated bot specs")
    lb.add_argument("--games", type=int, default=100)
    lb.add_argument("--seed0", type=int, default=EVAL_SEED0)
    lb.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    lb.add_argument("--json", default=None)
    tn = sub.add_parser("tournament")
    tn.add_argument("--bots", required=True, help="comma-separated bot specs")
    tn.add_argument("--games", type=int, default=50)
    tn.add_argument("--seed0", type=int, default=EVAL_SEED0)
    tn.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    tn.add_argument("--json", default=None)
    gf = sub.add_parser("gif")
    gf.add_argument("--a", required=True)
    gf.add_argument("--b", required=True)
    gf.add_argument("--map", type=int, default=EVAL_SEED0)
    gf.add_argument("--seed", type=int, default=0)
    gf.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "leaderboard":
        out = leaderboard(args.me, [s for s in args.opp.split(",") if s], args.games, args.seed0, args.procs)
    elif args.cmd == "tournament":
        out = tournament([s for s in args.bots.split(",") if s], args.games, args.seed0, args.procs)
    else:
        r = play_game(args.a, args.b, args.map, args.seed, record=True)
        save_gif(r, args.out)
        out = {k: v for k, v in r.items() if k != "actions"}
        print(out)
    if getattr(args, "json", None):
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
