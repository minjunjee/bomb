"""Tests of arena.rl.vecenv.   Run:  nice -n 10 .venv/bin/python -m arena.rl.test_vecenv   (or pytest arena/rl)

test_exact_*      the vectorised environment gives EXACTLY the observations, rewards, dones and episode results of a
                  plain single-process loop (Game + preprocess(turn(obs)) + turn_action) for the same seeds and
                  actions: both sides, bots / self-play / frozen-opponent entries, 300-step draws, faulty bots,
                  1 or 2 worker groups, fork and spawn.
test_league       set_league() changes the opponent distribution of new episodes.
test_shutdown     close() and a crashing trainer leave no worker process behind.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.core import bot_factory  # noqa: E402
from arena.rl.vecenv import (DRAW, KIND_BOT, LOSS, WIN, VecEnv, det_seed, env_seed, sample_index,  # noqa: E402
                             spec_kind)
from bombcompetition import Game, random_rules  # noqa: E402
from bombcompetition.neural import preprocess, turn, turn_action  # noqa: E402

TESTBOT = "arena/rl/_testbot.py"
REW = {"win": 1.0, "draw": 0.0, "loss": -0.5}


def reference(specs, probs, seed, n_env, acts0, acts1, steps):
    """The plain loop, one environment after the other. Returns per-step arrays shaped like the vecenv's."""
    out = {k: [] for k in ("obs0", "obs1", "rew0", "rew1", "done", "result", "ep_len", "ep_opp", "cur_opp",
                           "cur_side", "map_seed")}
    cols = []
    for i in range(n_env):
        es = env_seed(seed, i)
        rng = random.Random(es)
        col = {k: [] for k in out}
        episode = 0

        def start():
            nonlocal episode
            episode += 1
            ms = rng.randrange(1_000_000)
            side = rng.randrange(2)
            k = sample_index(probs, rng.random())
            game = Game(random_rules(ms), ms)
            bot = None
            if spec_kind(specs[k]) == KIND_BOT:
                random.seed(det_seed(es, episode, -1))
                bot = bot_factory(specs[k])()
            return game, side, k, bot, ms

        game, side, k, bot, ms = start()
        zero = np.zeros((13, 9, 9), np.float32)
        col["obs0"].append(preprocess(turn(game.obs(side))).numpy())
        col["obs1"].append(preprocess(turn(game.obs(1 - side))).numpy() if bot is None else zero)
        col["cur_opp"].append(k), col["cur_side"].append(side), col["map_seed"].append(ms)
        for t in range(steps):
            me = turn_action(int(acts0[t, i]), game.obs(side))  # the learner's turned action -> a real action
            if bot is None:
                other = turn_action(int(acts1[t, i]), game.obs(1 - side))
            else:
                random.seed(det_seed(es, episode, game.t))
                other = 0
                try:
                    a = bot.act(game.obs(1 - side))  # real observation
                    if isinstance(a, bool) or not hasattr(a, "__index__") or not 0 <= int(a) <= 5:
                        raise ValueError(a)
                    other = int(a)
                except Exception:  # noqa: BLE001
                    pass
            pair = [0, 0]
            pair[side], pair[1 - side] = me, other
            game.step(*pair)
            if game.done:
                w = game.winner()
                res = DRAW if w is None else WIN if w == side else LOSS
                r0 = {WIN: REW["win"], DRAW: REW["draw"], LOSS: REW["loss"]}[res]
                r1 = {WIN: REW["loss"], DRAW: REW["draw"], LOSS: REW["win"]}[res]
                col["rew0"].append(r0), col["rew1"].append(r1), col["done"].append(1), col["result"].append(res)
                col["ep_len"].append(game.t), col["ep_opp"].append(k)
                game, side, k, bot, ms = start()
            else:
                col["rew0"].append(0.0), col["rew1"].append(0.0), col["done"].append(0), col["result"].append(0)
                col["ep_len"].append(0), col["ep_opp"].append(0)
            col["obs0"].append(preprocess(turn(game.obs(side))).numpy())
            col["obs1"].append(preprocess(turn(game.obs(1 - side))).numpy() if bot is None else zero)
            col["cur_opp"].append(k), col["cur_side"].append(side), col["map_seed"].append(ms)
        cols.append(col)
    return {k: np.stack([np.asarray(c[k]) for c in cols], 1) for k in out}  # (time, env, ...)


def run_exact(specs, weights, workers, envs, groups, seed, steps, start_method=None, verbose=True):
    n = workers * envs
    rs = np.random.default_rng(seed + 99)
    acts0 = rs.integers(0, 6, (steps, n))
    acts1 = rs.integers(0, 6, (steps, n))
    calm = np.arange(n) % 3 == 0  # every third environment never bombs: long games, 300-step draws
    acts0[:, calm] = rs.integers(0, 5, (steps, int(calm.sum())))
    acts1[:, calm] = rs.integers(0, 5, (steps, int(calm.sum())))
    probs = (np.asarray(weights, float) / np.sum(weights)).tolist()
    ref = reference(specs, probs, seed, n, acts0, acts1, steps)
    env = VecEnv(list(zip(specs, weights)), workers=workers, envs_per_worker=envs, groups=groups, seed=seed,
                 reward_win=REW["win"], reward_draw=REW["draw"], reward_loss=REW["loss"], deterministic=True,
                 start_method=start_method)
    try:
        net_opp = lambda opp: np.array([spec_kind(specs[k]) != KIND_BOT for k in opp])  # noqa: E731

        def check_state(t):
            assert np.array_equal(env.cur_opp, ref["cur_opp"][t]), f"t={t}: opponent"
            assert np.array_equal(env.cur_side, ref["cur_side"][t]), f"t={t}: side"
            assert np.array_equal(env.map_seed, ref["map_seed"][t]), f"t={t}: map seed"
            assert (env.map_seed < 1_000_000).all()
            assert np.array_equal(env.obs0, ref["obs0"][t]), f"t={t}: obs0 differs"
            m = net_opp(env.cur_opp)
            assert np.array_equal(env.obs1[m], ref["obs1"][t][m]), f"t={t}: obs1 differs"

        check_state(0)
        for t in range(steps):
            env.act0[:] = acts0[t]
            env.act1[:] = acts1[t]
            order = list(range(env.G))
            if t % 2:
                order.reverse()  # the order in which groups are stepped must not matter
            for g in order:
                env.send(g)
            for g in reversed(order):
                env.wait(g)
            d = ref["done"][t].astype(bool)
            assert np.array_equal(env.done.astype(bool), d), f"t={t}: done"
            assert np.array_equal(env.rew0, ref["rew0"][t].astype(np.float32)), f"t={t}: rew0"
            was_net = net_opp(ref["cur_opp"][t])  # the opponent of the episode this step belonged to
            assert np.array_equal(env.rew1[was_net], ref["rew1"][t][was_net].astype(np.float32)), f"t={t}: rew1"
            assert np.array_equal(env.result[d], ref["result"][t][d]), f"t={t}: result"
            assert np.array_equal(env.ep_len[d], ref["ep_len"][t][d]), f"t={t}: ep_len"
            assert np.array_equal(env.ep_opp[d], ref["ep_opp"][t][d]), f"t={t}: ep_opp"
            check_state(t + 1)
        faults = int(env.faults.sum())
    finally:
        env.close()
    res, d = ref["result"], ref["done"].astype(bool)
    stats = {"episodes": int(d.sum()), "win": int((res == WIN).sum()), "draw": int((res == DRAW).sum()),
             "loss": int((res == LOSS).sum()), "timeouts": int((ref["ep_len"] == 300).sum()),
             "as_A": int((ref["cur_side"] == 0).sum()), "as_B": int((ref["cur_side"] == 1).sum()), "faults": faults}
    if verbose:
        print(f"  exact: specs={specs} W={workers} E={envs} G={groups} start={env.start_method} steps={steps} -> {stats}")
    return stats


def test_exact_bots_and_selfplay():
    """Everything at once: rule bots (their random module calls reproduced), the faulty test bot, self, past."""
    specs = ["random", "simple", "zoo.base_cautious", TESTBOT, "self", "past", "noisy:0.3:simple"]
    s = run_exact(specs, [1, 1, 1, 1, 2, 1, 1], workers=4, envs=6, groups=2, seed=3, steps=700)
    assert s["win"] > 0 and s["loss"] > 0 and s["draw"] > 0 and s["timeouts"] > 0, s
    assert s["as_A"] > 0 and s["as_B"] > 0 and s["faults"] > 0, s


def test_exact_one_group():
    s = run_exact(["simple", "self", TESTBOT], [1, 1, 1], workers=3, envs=4, groups=1, seed=11, steps=400)
    assert s["episodes"] > 10, s


def test_exact_spawn():
    s = run_exact(["random", "self", TESTBOT], [1, 1, 1], workers=2, envs=3, groups=2, seed=5, steps=150,
                  start_method="spawn")
    assert s["episodes"] > 5, s


def test_league():
    env = VecEnv("random:1,simple:1,self:1", workers=2, envs_per_worker=8, seed=1)
    try:
        assert len(set(env.cur_opp.tolist())) > 1
        env.set_league([0, 0, 1])
        seen = np.zeros(env.N, bool)
        for _ in range(400):
            env.step(np.full(env.N, 5), np.full(env.N, 5))  # everybody bombs: games end in 5 steps
            seen |= env.done.astype(bool)
            if seen.all():
                break
        assert seen.all()
        assert (env.cur_opp == 2).all(), env.cur_opp
        try:
            env.set_league([1, 1])
            raise AssertionError("a wrong-sized league must be rejected")
        except ValueError:
            pass
    finally:
        env.close()
    print("  league: ok")


def _children(pid):
    """Live child processes of pid, without helpers that end with their parent (ps itself, the resource tracker)."""
    out = subprocess.run(["ps", "-o", "pid=,args=", "--ppid", str(pid)], capture_output=True, text=True).stdout
    rows = [line.strip().split(None, 1) for line in out.splitlines() if line.strip()]
    return [(int(p), a) for p, a in rows if not a.startswith("ps ") and "resource_tracker" not in a]


def test_shutdown():
    env = VecEnv("random:1,self:1", workers=4, envs_per_worker=4, groups=2, seed=0)
    pids = [p.pid for p in env.procs]
    env.send(0)  # close while a group is stepping
    env.close()
    time.sleep(0.2)
    alive = [p for p in pids if os.path.exists(f"/proc/{p}")]
    assert not alive, f"workers still alive after close(): {alive}"
    # a trainer that dies without close() (SIGKILL): the workers notice and exit by themselves
    code = ("import os, sys, time; sys.path.insert(0, %r)\n"
            "from arena.rl.vecenv import VecEnv\n"
            "env = VecEnv('random:1', workers=3, envs_per_worker=2)\n"
            "print(' '.join(str(p.pid) for p in env.procs), flush=True)\n"
            "time.sleep(60)\n") % str(ROOT)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, cwd=ROOT)
    pids = [int(x) for x in p.stdout.readline().split()]
    assert len(pids) == 3 and all(os.path.exists(f"/proc/{x}") for x in pids)
    p.kill()
    p.wait()
    t0 = time.time()
    while time.time() - t0 < 15 and any(os.path.exists(f"/proc/{x}") for x in pids):
        time.sleep(0.25)
    alive = [x for x in pids if os.path.exists(f"/proc/{x}")]
    assert not alive, f"orphan workers after the trainer was killed: {alive}"
    left = _children(os.getpid())
    assert not left, f"this test process still has children: {left}"
    print("  shutdown: ok (close() while stepping, and SIGKILL of the trainer: no orphans)")


if __name__ == "__main__":
    t0 = time.time()
    for fn in (test_exact_one_group, test_exact_bots_and_selfplay, test_exact_spawn, test_league, test_shutdown):
        print(fn.__name__)
        fn()
    print(f"ALL TESTS PASSED in {time.time() - t0:.1f}s")
