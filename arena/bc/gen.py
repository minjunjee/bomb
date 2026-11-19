"""arena.bc.gen - parallel generation of behaviour-cloning data with the REFERENCE engine (bombcompetition.Game).

    nice -n 10 .venv/bin/python -m arena.bc.gen --teacher zoo.T --pairs "<pairs>" --games N --procs P --seed0 S \
        --out runs/bc/<name>/data/<tag>

    --teacher SPEC     the bot whose actions are the labels: any harness spec; class Policy with act(obs) and,
                       optionally, action_values(obs) -> 6 floats in the real frame
    --pairs TEXT       who plays the games (see arena.bc.common for the full grammar):
                           "T|T, T|noisy:0.2:T*2, T|simple, +S|T, +S@sample|zoo.x*0.5"
                       pair = actor|actor[*weight]; an actor is any harness spec (zoo.x, noisy:<eps>:<spec>,
                       net:<folder>, net:<folder>@sample, ...); T = the teacher, S = net:<--student folder>.
                       LABELLING: a leading "+" marks the actors whose states are labelled; a pair without any "+"
                       labels BOTH players (the default). Labels ALWAYS come from the teacher, never from the actor.
    --games N          number of games; game g is played on map seed0 + g (all seeds must be < 1,000,000)
    --seed0 S          first map seed
    --procs P          worker processes (worker w plays games w, w+P, w+2P, ...)
    --out DIR          dataset folder: one shard set per worker (w<worker>_p<part>.npz) + manifest.json
    --student FOLDER   the submission folder that the token S stands for
    --no-q             do not store action values even if the teacher has action_values()
    --shard-size K     samples per shard part (default 200000)
    --overwrite        replace a finished dataset in --out (unfinished leftovers are always replaced)

What is stored for every labelled (step, player) - see arena.bc.common for the file layout:
    x = preprocess(turn(obs)) as exact uint8 codes, y = turn_action(teacher action, obs), q = the teacher's
    action_values(obs) moved to the turned frame, and meta: game index, step, player, the action actually played
    (turned frame), the final outcome for that player (+1 / 0 / -1) and the number of steps until the game ended.

Rules of the generator
    * sides are assigned at random per game; the plan (pair, sides) is a deterministic function of the arguments;
    * random, numpy.random and torch are seeded per game from (the two specs, the map seed): a game does not depend
      on the number of processes or on which worker plays it;
    * the teacher is asked ONCE per labelled state: when the actor is the teacher itself (T or noisy:<eps>:T) its
      own answer is the label, and when act() goes through self.action_values(obs) that result is reused as q
      (otherwise action_values is called a second time: manifest counts.q_second_calls shows how often);
    * a teacher exception / invalid answer does not kill the run: the sample is skipped and counted
      (counts.teacher_errors); a failing actor stays put and is counted (counts.actor_faults), like the harness;
    * net: actors run on the CPU with one thread in every worker;
    * every shard is written atomically; manifest.json is written last, so a folder with a manifest is complete.
"""

from __future__ import annotations

import os

# one thread per process, set before numpy / torch are imported (workers are forked from this process)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from .common import (EVAL_SEED0, FLIP_IDX, FORMAT, ROOT, encode_planes, pairs_as_json, parse_pairs, plan_games,
                     save_npz_atomic, write_json_atomic)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_OURS = re.compile(r"^(manifest\.json|w\d+_p\d+\.npz|w\d+\.json|\..*\.tmp)$")


def _check_action(out) -> int:
    if isinstance(out, bool) or not hasattr(out, "__index__") or not 0 <= int(out) <= 5:
        raise ValueError(f"act returned {out!r}")
    return int(out)


class Stats:
    def __init__(self):
        self.t_teacher = self.t_actor = self.t_pre = self.t_engine = 0.0
        self.teacher_calls = self.teacher_errors = self.q_second_calls = 0
        self.errors: list[str] = []

    def note(self, e):
        msg = repr(e)[:300]
        if len(self.errors) < 5 and msg not in self.errors:
            self.errors.append(msg)


class TeacherTap:
    """One teacher instance. label(obs) -> (action, q or None) in the REAL frame, with ONE search whenever the
    teacher's act() goes through its own action_values(obs): that call is intercepted and its result reused."""

    def __init__(self, bot, want_q: bool = True):
        self.bot = bot
        self.has_q = bool(want_q) and callable(getattr(bot, "action_values", None))
        self.second_calls = 0
        self._cur = self._hit = None
        if self.has_q:
            self._values = bot.action_values
            try:
                bot.action_values = self._tap  # the instance attribute hides the method for the teacher's own calls
            except Exception:  # noqa: BLE001  (__slots__ ...): no interception, action_values is called explicitly
                pass

    def _tap(self, obs, *args, **kwargs):
        out = self._values(obs, *args, **kwargs)
        if obs is self._cur:
            self._hit = out
        return out

    def label(self, obs):
        self._cur, self._hit = obs, None
        try:
            a = _check_action(self.bot.act(obs))
            q = None
            if self.has_q:
                if self._hit is None:
                    self.second_calls += 1
                    self._hit = self._values(obs)
                q = np.array(self._hit, dtype=np.float32).reshape(-1)
                if q.shape != (6,) or np.isnan(q).any():
                    raise ValueError(f"action_values returned {self._hit!r}")
        finally:
            self._cur = None
        return a, q


class Seat:
    """One side of one game: the actor that plays and, if its states are labelled, the teacher that labels them."""

    def __init__(self, spec: str, labelled: bool, teacher_spec: str, want_q: bool, st: Stats):
        from arena.core import bot_factory

        self.st, self.labelled, self.faults = st, labelled, 0
        self.bot = self.tap = None
        self.actor_is_teacher = self.noisy = False
        self.eps = 0.0
        if labelled:
            if spec == teacher_spec:
                self.actor_is_teacher = True
            elif spec.startswith("noisy:") and spec.split(":", 2)[2] == teacher_spec:
                self.actor_is_teacher = self.noisy = True
                self.eps = float(spec.split(":", 2)[1])
            try:
                self.tap = TeacherTap(bot_factory(teacher_spec)(), want_q)
            except Exception as e:  # noqa: BLE001
                st.note(e)
        if not self.actor_is_teacher:
            try:
                self.bot = bot_factory(spec)()
            except Exception:  # noqa: BLE001
                self.faults += 1

    def act(self, obs):
        """-> (the action played, the teacher's action or None, the teacher's q or None), all in the real frame."""
        from bombcompetition import legal_actions

        st, a = self.st, 0
        if not self.actor_is_teacher:
            t0 = time.perf_counter()
            if self.bot is not None:
                try:
                    a = _check_action(self.bot.act(obs))
                except Exception:  # noqa: BLE001  (the harness: a fault, the bot stays put)
                    self.faults += 1
            st.t_actor += time.perf_counter() - t0
            if not self.labelled:
                return a, None, None
        ta = q = None
        t0 = time.perf_counter()
        st.teacher_calls += 1
        if self.tap is None:
            st.teacher_errors += 1
        else:
            try:
                ta, q = self.tap.label(obs)
            except Exception as e:  # noqa: BLE001  (skip the sample, count it)
                st.teacher_errors += 1
                st.note(e)
        st.t_teacher += time.perf_counter() - t0
        if self.actor_is_teacher:
            if ta is None:
                self.faults += 1  # stays put
            else:
                a = ta
                if self.noisy and random.random() < self.eps:  # the same draws as arena.core's noisy: wrapper
                    a = int(random.choice(legal_actions(obs)))
        return a, ta, q


def game_seed(specs, map_seed: int) -> int:
    return zlib.crc32(f"bc|{specs[0]}|{specs[1]}|{map_seed}".encode())


def play_labelled_game(map_seed: int, specs, labelled, teacher_spec: str, want_q: bool, st: Stats):
    """One game on random_rules(map_seed): specs[p] plays side p, labelled[p] says whether p's states are labelled.
    Returns (samples: dict of arrays or None, info: dict)."""
    import torch

    from bombcompetition import Game, random_rules
    from bombcompetition.neural import preprocess, turn, turn_action

    seed = game_seed(specs, map_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    game = Game(random_rules(map_seed), map_seed)
    seats = [Seat(specs[p], labelled[p], teacher_spec, want_q, st) for p in (0, 1)]
    xs, ys, qs, steps, players, acts = [], [], [], [], [], []
    while not game.done:
        both = []
        for p in (0, 1):
            obs = game.obs(p)
            a, ta, q = seats[p].act(obs)
            both.append(a)
            if ta is None:  # not labelled, or the teacher failed (counted)
                continue
            t0 = time.perf_counter()
            xs.append(preprocess(turn(obs)).numpy())       # the network's input: always the TURNED frame
            st.t_pre += time.perf_counter() - t0
            ys.append(turn_action(ta, obs))                # the teacher's action in the turned frame
            acts.append(turn_action(a, obs))               # what was actually played, turned frame
            if q is not None:
                qs.append(q[FLIP_IDX] if p == 1 else q)    # q_turned[a'] = q_real[FLIP[a']]
            steps.append(obs["step"])
            players.append(p)
        t0 = time.perf_counter()
        game.step(*both)
        st.t_engine += time.perf_counter() - t0
    for s in seats:
        if s.tap is not None:
            st.q_second_calls += s.tap.second_calls
    w, total = game.winner(), game.t
    info = {"winner": -1 if w is None else int(w), "steps": total, "samples": len(xs),
            "faults": [s.faults for s in seats]}
    if not xs:
        return None, info
    assert len(qs) in (0, len(xs))
    player = np.array(players, dtype=np.uint8)
    step = np.array(steps, dtype=np.int16)
    out = {
        "x": encode_planes(np.stack(xs)),  # exact: raises unless decode(encode(x)) == x for every sample
        "y": np.array(ys, dtype=np.uint8),
        "step": step,
        "player": player,
        "act": np.array(acts, dtype=np.uint8),
        "outcome": (np.zeros(len(xs)) if w is None else np.where(player == w, 1, -1)).astype(np.int8),
        "togo": (total - step).astype(np.int16),
    }
    if qs:
        out["q"] = np.stack(qs).astype(np.float32)
    return out, info


def _worker(wid: int, a: dict, pairs, plan_pair, plan_swap, counters):
    import torch

    torch.set_num_threads(1)  # net: actors: CPU, one thread
    st = Stats()
    out = Path(a["out"])
    buf, grow, shards = [], [], []
    n_buf = part = 0
    per_pair = [{"games": 0, "samples": 0, "label_eq_played": 0, "first_win": 0, "draw": 0, "first_loss": 0, "steps": 0,
                 "actor_faults": 0} for _ in pairs]
    label_hist, played_hist, agree = np.zeros(6, np.int64), np.zeros(6, np.int64), 0
    t_start = time.perf_counter()

    def flush():
        nonlocal buf, grow, n_buf, part
        if not grow:
            return
        arrays = {}
        if buf:
            for k in buf[0]:
                arrays[k] = np.concatenate([b[k] for b in buf])
        for i, k in enumerate(("g_index", "g_seed", "g_pair", "g_swap", "g_winner", "g_steps", "g_samples")):
            arrays[k] = np.array([r[i] for r in grow], dtype=np.int32)
        name = f"w{wid:03d}_p{part:03d}.npz"
        save_npz_atomic(out / name, **arrays)
        shards.append({"file": name, "samples": n_buf, "games": len(grow)})
        buf, grow, n_buf = [], [], 0
        part += 1

    for g in range(wid, a["games"], a["procs"]):
        pi, swap = int(plan_pair[g]), bool(plan_swap[g])
        pair = pairs[pi]
        specs = (pair.b, pair.a) if swap else (pair.a, pair.b)
        labelled = (pair.label_b, pair.label_a) if swap else (pair.label_a, pair.label_b)
        samples, info = play_labelled_game(a["seed0"] + g, specs, labelled, a["teacher"], a["want_q"], st)
        n = info["samples"]
        same = 0
        if samples is not None:
            samples["game"] = np.full(n, g, dtype=np.int32)
            buf.append(samples)
            label_hist += np.bincount(samples["y"], minlength=6)
            played_hist += np.bincount(samples["act"], minlength=6)
            same = int((samples["y"] == samples["act"]).sum())
            agree += same
        grow.append((g, a["seed0"] + g, pi, int(swap), info["winner"], info["steps"], n))
        n_buf += n
        first = 1 if swap else 0  # the side the pair's FIRST actor plays
        pp = per_pair[pi]
        pp["games"] += 1
        pp["samples"] += n
        pp["label_eq_played"] += same
        pp["steps"] += info["steps"]
        pp["actor_faults"] += sum(info["faults"])
        pp["draw" if info["winner"] < 0 else ("first_win" if info["winner"] == first else "first_loss")] += 1
        counters[2 * wid] += 1
        counters[2 * wid + 1] += n
        if n_buf >= a["shard_size"]:
            flush()
    flush()
    write_json_atomic(out / f"w{wid:03d}.json", {
        "worker": wid, "shards": shards, "per_pair": per_pair, "seconds": time.perf_counter() - t_start,
        "t_teacher": st.t_teacher, "t_actor": st.t_actor, "t_pre": st.t_pre, "t_engine": st.t_engine,
        "teacher_calls": st.teacher_calls, "teacher_errors": st.teacher_errors, "q_second_calls": st.q_second_calls,
        "errors": st.errors, "label_hist": label_hist.tolist(), "played_hist": played_hist.tolist(),
        "label_eq_played": agree, "torch_threads": torch.get_num_threads(),
    })


def generate(teacher: str, pairs_text: str, games: int, procs: int, seed0: int, out, student: str | None = None,
             no_q: bool = False, shard_size: int = 200_000, overwrite: bool = False, progress: float = 10.0,
             argv: list[str] | None = None) -> dict:
    """The whole run (also the python API). Returns the manifest."""
    from arena.core import bot_factory

    if games < 1 or seed0 < 0 or seed0 + games > EVAL_SEED0:
        raise SystemExit(f"map seeds {seed0}..{seed0 + games - 1} must stay inside [0, {EVAL_SEED0}): "
                         f"evaluation maps start at {EVAL_SEED0}")
    pairs = parse_pairs(pairs_text, teacher, student)
    for spec in sorted({teacher} | {s for p in pairs for s in (p.a, p.b)}):
        bot_factory(spec)  # an unknown spec / a missing file fails here, not silently inside the workers
    probe = bot_factory(teacher)()
    if not callable(getattr(probe, "act", None)):
        raise SystemExit(f"teacher {teacher!r} has no act(obs)")
    has_q = callable(getattr(probe, "action_values", None))
    want_q = has_q and not no_q
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and not overwrite:
        raise SystemExit(f"{out} already holds a finished dataset (pass --overwrite to replace it)")
    for f in out.iterdir():
        if _OURS.match(f.name):
            f.unlink()
    procs = max(1, min(procs, games))
    plan_pair, plan_swap = plan_games(pairs, games, seed0)
    a = {"teacher": teacher, "games": games, "procs": procs, "seed0": seed0, "out": str(out), "want_q": want_q,
         "shard_size": shard_size}
    print(f"[gen] teacher {teacher} (action_values: {'yes' if has_q else 'no'}{', not stored' if has_q and no_q else ''})"
          f" | {games} games on map seeds {seed0}..{seed0 + games - 1} | {procs} procs -> {out}", flush=True)
    for i, p in enumerate(pairs):
        print(f"[gen]   pair {i}: {p.key}  x{int((plan_pair == i).sum())}", flush=True)

    ctx = mp.get_context("fork")
    counters = ctx.Array("q", 2 * procs, lock=False)
    workers = [ctx.Process(target=_worker, args=(w, a, pairs, plan_pair, plan_swap, counters), daemon=True)
               for w in range(procs)]
    t0 = last = time.time()
    for w in workers:
        w.start()
    while True:
        bad = [w for w in workers if w.exitcode not in (None, 0)]
        if bad:
            for w in workers:
                if w.is_alive():
                    w.terminate()
            raise SystemExit(f"[gen] {len(bad)} worker(s) died (exit codes {[w.exitcode for w in bad]}); "
                             f"no manifest written, {out} is incomplete")
        if not any(w.is_alive() for w in workers):
            break
        time.sleep(0.2)
        if time.time() - last >= progress:
            last = time.time()
            done, n = sum(counters[0::2]), sum(counters[1::2])
            print(f"[gen] {done}/{games} games, {n} samples, {n / (last - t0):.0f} samples/s "
                  f"({n / (last - t0) / procs:.1f}/s per proc), {last - t0:.0f}s", flush=True)
    wall = time.time() - t0

    stats = []
    for w in range(procs):
        f = out / f"w{w:03d}.json"
        stats.append(json.loads(f.read_text()))
        f.unlink()
    shards = [s for st in stats for s in st["shards"]]
    samples = sum(s["samples"] for s in shards)
    tot = lambda k: sum(st[k] for st in stats)  # noqa: E731
    per_pair = {}
    for i, p in enumerate(pairs):
        d = {k: sum(st["per_pair"][i][k] for st in stats) for k in stats[0]["per_pair"][i]}
        d["avg_steps"] = d.pop("steps") / max(1, d["games"])
        d["label_eq_played"] = d["label_eq_played"] / max(1, d["samples"])  # how often the actors played the label
        per_pair[p.key] = d
    worker_seconds = tot("seconds")
    calls = max(1, tot("teacher_calls"))
    manifest = {
        "format": FORMAT, "complete": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": argv if argv is not None else sys.argv[1:],
        "args": {"teacher": teacher, "pairs": pairs_text, "games": games, "procs": procs, "seed0": seed0,
                 "student": student, "no_q": no_q, "shard_size": shard_size},
        "teacher": teacher, "pairs": pairs_as_json(pairs), "games": games, "seed0": seed0,
        "samples": samples, "has_q": bool(want_q), "planes": "uint8 codes, arena.bc.common.LUT (exact)",
        "shards": shards,
        "counts": {
            "per_pair": per_pair,
            "teacher_calls": tot("teacher_calls"), "teacher_errors": tot("teacher_errors"),
            "q_second_calls": tot("q_second_calls"),
            "actor_faults": sum(d["actor_faults"] for d in per_pair.values()),
            "label_hist": np.sum([st["label_hist"] for st in stats], 0).tolist(),
            "played_hist": np.sum([st["played_hist"] for st in stats], 0).tolist(),
            "label_eq_played": tot("label_eq_played") / max(1, samples),
            "errors": sorted({e for st in stats for e in st["errors"]})[:10],
        },
        "timing": {
            "wall_s": wall, "worker_s": worker_seconds,
            "samples_per_s": samples / max(1e-9, wall),
            "samples_per_s_per_proc": samples / max(1e-9, worker_seconds),
            "teacher_ms_per_label": 1000 * tot("t_teacher") / calls,
            "preprocess_ms_per_sample": 1000 * tot("t_pre") / max(1, samples),
            "actor_ms_per_sample": 1000 * tot("t_actor") / max(1, samples),
            "engine_share": tot("t_engine") / max(1e-9, worker_seconds),
            "teacher_share": tot("t_teacher") / max(1e-9, worker_seconds),
            "torch_threads_per_worker": max(st["torch_threads"] for st in stats),
        },
    }
    write_json_atomic(out / "manifest.json", manifest)
    c, t = manifest["counts"], manifest["timing"]
    print(f"[gen] done: {samples} samples from {games} games in {wall:.1f}s | {t['samples_per_s_per_proc']:.0f} "
          f"labelled samples/s per process | teacher {t['teacher_ms_per_label']:.3f} ms/label, preprocess "
          f"{t['preprocess_ms_per_sample']:.3f} ms | labels {c['label_hist']} | label == played "
          f"{c['label_eq_played']:.3f}", flush=True)
    if c["teacher_errors"] or c["actor_faults"] or (want_q and c["q_second_calls"]):
        print(f"[gen] WARNING: teacher_errors {c['teacher_errors']} (samples skipped), actor_faults "
              f"{c['actor_faults']}, q_second_calls {c['q_second_calls']} {c['errors']}", flush=True)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--student", default=None)
    ap.add_argument("--no-q", action="store_true")
    ap.add_argument("--shard-size", type=int, default=200_000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress", type=float, default=10.0, help="seconds between progress lines")
    args = ap.parse_args(argv)
    generate(args.teacher, args.pairs, args.games, args.procs, args.seed0, args.out, args.student, args.no_q,
             args.shard_size, args.overwrite, args.progress, argv=list(argv) if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    main()
