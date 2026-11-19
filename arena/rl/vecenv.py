"""Multi-process vectorised BOMB environment on the REFERENCE engine (bombcompetition.Game), for league training.

Layout
------
W worker processes x E environments each = N environments. Everything the trainer touches lives in ONE shared
memory block (numpy views, row i = environment i; worker w owns rows [w*E, (w+1)*E)):

    obs0  (N, 13, 9, 9) f32   the learner's input  x = preprocess(turn(obs))  for the side it plays in env i
    obs1  (N, 13, 9, 9) f32   the same for the OTHER side; only written while the opponent of env i is a network
    act0  (N,) u8             the learner's action = a network output index in the TURNED frame (written by the trainer)
    act1  (N,) u8             the other side's action, turned frame (read only when the opponent is a network)
    rew0  (N,) f32            reward of the last step from the learner's point of view (win / draw / loss at the end)
    rew1  (N,) f32            the same step seen from the other side (its own win = +reward_win)
    done  (N,) u8             1 = the last step ended an episode (someone was hit, or 300 steps); the env was reset
                              at once, so obs0/obs1 already hold the FIRST observation of the next episode
    result, ep_len, ep_opp    of the episode that just ended (valid where done == 1): result 1 = the learner (side 0
                              of the env, obs0/act0) won, 2 = draw, 3 = lost; number of steps; league index
    cur_opp, cur_side, map_seed   of the episode now running: league index of the opponent, the player index (0 = A,
                              1 = B) the learner plays, the map seed

Every episode: a fresh map random_rules(seed) with seed < 1_000_000 (evaluation maps start at 1_000_000), the
learner's side at random, the opponent drawn from the league distribution (`set_league`, may change at any time).

League entries (strings)
    any harness spec (random, simple, zoo.x, zoo.x:arg, noisy:0.1:zoo.x, net:<folder>, file.py)
                  -> the bot runs inside the worker: arena.core.bot_factory(spec)() per game, real (un-turned) obs,
                     an exception or an invalid action = STAY (counted in `faults`), like the server's SafeBot
    self          -> the other side is played by the trainer through act1 (the current weights: both sides give data)
    past          -> the same mechanism; the trainer answers with a frozen past snapshot

Stepping is asynchronous per GROUP of workers (G contiguous groups): `send(g)` lets group g step with the actions in
act0/act1, `wait(g)` blocks until its results are in shared memory. While one group steps, the trainer runs the
network for another. `step()` = all groups at once (synchronous; tests).

Determinism: env i owns random.Random(env_seed(seed, i)) and draws, per episode and in this order,
map_seed = randrange(map_seeds), side = randrange(2), opponent = sample_index(league, random()). With
deterministic=True the global `random` module is re-seeded before every bot call (det_seed), so a plain
single-process loop reproduces the worker exactly (see test_vecenv.py); training leaves the global stream alone.

Workers are started with fork (fast) unless CUDA is already initialised in this process, then with spawn.
They exit on close(), when the trainer dies (parent pid check), and are daemonic: no orphans.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
import weakref
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLIP = (0, 2, 1, 4, 3, 5)      # turned frame <-> real frame for player 1 (bombcompetition.neural.FLIP)
TRAIN_MAP_SEEDS = 1_000_000    # evaluation maps are random_rules(seed >= 1_000_000): training stays below
OBS_SHAPE = (13, 9, 9)
WIN, DRAW, LOSS = 1, 2, 3      # `result` codes (0 = no episode ended)
KIND_BOT, KIND_SELF, KIND_PAST = 0, 1, 2
CMD_STEP, CMD_RESET, CMD_CLOSE = 0, 1, 2


# ----------------------------------------------------------------------------------------------------------------
# small pure helpers (shared with the reference loop of the test)
# ----------------------------------------------------------------------------------------------------------------
def parse_league(text: str) -> list[tuple[str, float]]:
    """"simple:1,random:1,zoo.base_cautious:4,self:2" -> [(spec, weight)]. The LAST ':<number>' of an entry is its
    weight (default 1): "noisy:0.2:simple" has weight 1, "noisy:0.2:simple:3" weight 3; for a numeric bot argument
    write the weight explicitly ("zoo.x:7:1")."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        head, sep, tail = part.rpartition(":")
        weight = None
        if sep and head not in ("noisy", "net"):
            try:
                weight = float(tail)
            except ValueError:
                weight = None
        if weight is None:
            head, weight = part, 1.0
        if head.startswith("noisy:") and head.count(":") < 2:  # "noisy:0.2" is not a spec: the number was eps
            head, weight = part, 1.0
        out.append((head, weight))
    if not out:
        raise ValueError("empty league")
    return out


def spec_kind(spec: str) -> int:
    return KIND_SELF if spec == "self" else KIND_PAST if spec == "past" else KIND_BOT


def env_seed(seed: int, index: int) -> int:
    """The seed of environment `index`'s private random.Random."""
    return (int(seed) * 1_000_003 + int(index) * 7919 + 12345) & 0x7FFFFFFFFFFFFFFF


def det_seed(eseed: int, episode: int, t: int) -> int:
    """deterministic=True: the global `random` is seeded with this before the bot is built (t = -1) and before its
    act() at game step t."""
    return ((int(eseed) % 1_000_000_007) * 1_000_003 + int(episode) * 1009 + int(t) + 1) & 0x7FFFFFFFFFFFFFFF


def sample_index(probs, u: float) -> int:
    """Index drawn from the (unnormalised) weights `probs` with one uniform u in [0, 1)."""
    total = 0.0
    for p in probs:
        total += p
    x = u * total
    last = 0
    for i, p in enumerate(probs):
        if p > 0:
            last = i
            x -= p
            if x < 0:
                return i
    return last


def _fields(n_env: int, n_workers: int, n_specs: int):
    return [
        ("obs0", np.float32, (n_env, *OBS_SHAPE)), ("obs1", np.float32, (n_env, *OBS_SHAPE)),
        ("act0", np.uint8, (n_env,)), ("act1", np.uint8, (n_env,)),
        ("rew0", np.float32, (n_env,)), ("rew1", np.float32, (n_env,)),
        ("done", np.uint8, (n_env,)), ("result", np.uint8, (n_env,)),
        ("ep_len", np.int32, (n_env,)), ("ep_opp", np.int32, (n_env,)),
        ("cur_opp", np.int32, (n_env,)), ("cur_side", np.uint8, (n_env,)), ("map_seed", np.int32, (n_env,)),
        ("league_p", np.float64, (n_specs,)), ("cmd", np.int32, (n_workers,)), ("err", np.int32, (n_workers,)),
        ("faults", np.int64, (n_workers,)), ("busy_ns", np.int64, (n_workers,)), ("seed", np.int64, (1,)),
    ]


def _arena_size(fields) -> int:
    off = 0
    for _, dtype, shape in fields:
        off = (off + 63) // 64 * 64 + int(np.prod(shape)) * np.dtype(dtype).itemsize
    return off + 64


def _views(arena: torch.Tensor, fields) -> dict:
    buf, out, off = arena.numpy(), {}, 0
    for name, dtype, shape in fields:
        off = (off + 63) // 64 * 64
        out[name] = np.ndarray(shape, dtype=dtype, buffer=buf, offset=off)
        off += int(np.prod(shape)) * np.dtype(dtype).itemsize
    return out


# ----------------------------------------------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------------------------------------------
class _Env:
    __slots__ = ("rng", "eseed", "episode", "game", "side", "opp", "bot", "net_opp")


def _worker(wid, lo, hi, arena, n_env, n_workers, specs, go, done_sem, parent_pid, opt):
    views = None
    try:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[v] = "1"
        torch.set_num_threads(1)
        try:
            os.nice(max(0, 10 - os.nice(0)))  # shared machine: never above nice 10
        except OSError:
            pass
        from arena.core import bot_factory
        from bombcompetition import Game, random_rules
        from bombcompetition.neural import preprocess, turn

        views = _views(arena, _fields(n_env, n_workers, len(specs)))
        obs0, obs1, act0, act1 = views["obs0"], views["obs1"], views["act0"], views["act1"]
        rew0, rew1, done, result = views["rew0"], views["rew1"], views["done"], views["result"]
        ep_len, ep_opp, cur_opp, cur_side = views["ep_len"], views["ep_opp"], views["cur_opp"], views["cur_side"]
        map_seed_arr, league_p, cmd = views["map_seed"], views["league_p"], views["cmd"]
        faults_arr, busy_arr = views["faults"], views["busy_ns"]
        kinds = [spec_kind(s) for s in specs]
        factories = {i: bot_factory(s) for i, s in enumerate(specs) if kinds[i] == KIND_BOT}
        r_win, r_draw, r_loss = opt["reward_win"], opt["reward_draw"], opt["reward_loss"]
        det, map_seeds = opt["deterministic"], opt["map_seeds"]
        assert 0 < map_seeds <= TRAIN_MAP_SEEDS
        envs = [_Env() for _ in range(hi - lo)]
        nfaults = 0

        def start_episode(env, i, probs):
            nonlocal nfaults
            rng = env.rng
            env.episode += 1
            ms = rng.randrange(map_seeds)
            side = rng.randrange(2)
            k = sample_index(probs, rng.random())
            game = Game(random_rules(ms), ms)
            env.game, env.side, env.opp, env.net_opp = game, side, k, kinds[k] != KIND_BOT
            env.bot = None
            if env.net_opp:
                obs1[i] = preprocess(turn(game.obs(1 - side))).numpy()
            else:
                if det:
                    random.seed(det_seed(env.eseed, env.episode, -1))
                try:
                    env.bot = factories[k]()
                except Exception:  # noqa: BLE001  (like play_game: a bot that cannot be built stays put)
                    nfaults += 1
            obs0[i] = preprocess(turn(game.obs(side))).numpy()
            cur_opp[i], cur_side[i], map_seed_arr[i] = k, side, ms

        while True:
            if not go.acquire(timeout=3.0):
                if os.getppid() != parent_pid:  # the trainer is gone
                    return
                continue
            c = int(cmd[wid])
            if c == CMD_CLOSE:
                return
            t0 = time.perf_counter_ns()
            probs = league_p.tolist()
            if c == CMD_RESET:
                base = int(views["seed"][0])
                rew0[lo:hi] = 0
                rew1[lo:hi] = 0
                done[lo:hi] = 0
                result[lo:hi] = 0
                for e, env in enumerate(envs):
                    env.eseed = env_seed(base, lo + e)
                    env.rng = random.Random(env.eseed)
                    env.episode = 0
                    start_episode(env, lo + e, probs)
            else:
                a0s, a1s = act0[lo:hi].tolist(), act1[lo:hi].tolist()
                r0s, r1s, dns = [0.0] * (hi - lo), [0.0] * (hi - lo), [0] * (hi - lo)
                for e, env in enumerate(envs):
                    i = lo + e
                    game, side = env.game, env.side
                    a = a0s[e]
                    if side:
                        a = FLIP[a]
                    if env.net_opp:
                        b = a1s[e]
                        if not side:  # the other side is player 1: its turned action is flipped back
                            b = FLIP[b]
                    else:
                        b = 0
                        bot = env.bot
                        if bot is not None:
                            if det:
                                random.seed(det_seed(env.eseed, env.episode, game.t))
                            try:
                                out = bot.act(game.obs(1 - side))  # the real (un-turned) observation
                                if isinstance(out, bool) or not hasattr(out, "__index__") or not 0 <= int(out) <= 5:
                                    raise ValueError(out)
                                b = int(out)
                            except Exception:  # noqa: BLE001
                                nfaults += 1
                    if side:
                        game.step(b, a)
                    else:
                        game.step(a, b)
                    if game.done:  # a hit, or 300 steps: a true end either way
                        w = game.winner()
                        if w is None:
                            result[i], r0s[e], r1s[e] = DRAW, r_draw, r_draw
                        elif w == side:
                            result[i], r0s[e], r1s[e] = WIN, r_win, r_loss
                        else:
                            result[i], r0s[e], r1s[e] = LOSS, r_loss, r_win
                        dns[e] = 1
                        ep_len[i], ep_opp[i] = game.t, env.opp
                        start_episode(env, i, probs)  # the terminal observation is never published
                    else:
                        obs0[i] = preprocess(turn(game.obs(side))).numpy()
                        if env.net_opp:
                            obs1[i] = preprocess(turn(game.obs(1 - side))).numpy()
                rew0[lo:hi] = r0s
                rew1[lo:hi] = r1s
                done[lo:hi] = dns
            faults_arr[wid] = nfaults
            busy_arr[wid] += time.perf_counter_ns() - t0
            done_sem.release()
    except KeyboardInterrupt:
        pass
    except BaseException:  # noqa: BLE001
        sys.stderr.write(f"[vecenv worker {wid}] crashed:\n{traceback.format_exc()}\n")
        sys.stderr.flush()
        try:
            if views is not None:
                views["err"][wid] = 1
            done_sem.release()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------------------------
# trainer side
# ----------------------------------------------------------------------------------------------------------------
_LIVE: "weakref.WeakSet[VecEnv]" = weakref.WeakSet()


@atexit.register
def _close_all():
    for env in list(_LIVE):
        env.close()


class VecEnv:
    """See the module docstring. league: "spec:weight,..." or [(spec, weight)]."""

    def __init__(self, league, workers: int = 8, envs_per_worker: int = 16, groups: int = 1, seed: int = 0,
                 reward_win: float = 1.0, reward_draw: float = 0.0, reward_loss: float = -0.5,
                 deterministic: bool = False, start_method: str | None = None, timeout: float = 600.0,
                 map_seeds: int = TRAIN_MAP_SEEDS):
        if isinstance(league, str):
            league = parse_league(league)
        self.specs = [s for s, _ in league]
        self.weights = np.array([w for _, w in league], dtype=np.float64)
        if (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError("league weights must be >= 0 and not all zero")
        if not 0 < map_seeds <= TRAIN_MAP_SEEDS:
            raise ValueError("training map seeds must stay below 1_000_000")
        self.kinds = np.array([spec_kind(s) for s in self.specs], dtype=np.int64)
        self.W, self.E, self.N = int(workers), int(envs_per_worker), int(workers) * int(envs_per_worker)
        self.G = max(1, min(int(groups), self.W))
        self.timeout = timeout
        self._closed = True
        self._fields = _fields(self.N, self.W, len(self.specs))
        self._arena = torch.zeros(_arena_size(self._fields), dtype=torch.uint8).share_memory_()
        self._v = _views(self._arena, self._fields)
        for name, arr in self._v.items():
            setattr(self, name, arr)
        self.league_p[:] = self.weights / self.weights.sum()
        self.seed[0] = seed
        # contiguous groups of workers -> contiguous row ranges
        bounds = [round(g * self.W / self.G) for g in range(self.G + 1)]
        self.group_workers = [list(range(bounds[g], bounds[g + 1])) for g in range(self.G)]
        self.group_rows = [(bounds[g] * self.E, bounds[g + 1] * self.E) for g in range(self.G)]
        if start_method is None:
            start_method = "spawn" if torch.cuda.is_initialized() else "fork"
        self.start_method = start_method
        ctx = mp.get_context(start_method)
        self._go = [ctx.Semaphore(0) for _ in range(self.W)]
        self._done = [ctx.Semaphore(0) for _ in range(self.G)]
        self._pending = [False] * self.G
        opt = {"reward_win": float(reward_win), "reward_draw": float(reward_draw), "reward_loss": float(reward_loss),
               "deterministic": bool(deterministic), "map_seeds": int(map_seeds)}
        self.procs = []
        self._closed = False
        _LIVE.add(self)
        try:
            for g, ws in enumerate(self.group_workers):
                for w in ws:
                    p = ctx.Process(target=_worker, name=f"vecenv-{w}", daemon=True,
                                    args=(w, w * self.E, (w + 1) * self.E, self._arena, self.N, self.W, self.specs,
                                          self._go[w], self._done[g], os.getpid(), opt))
                    p.start()
                    self.procs.append(p)
            self.reset()
        except BaseException:
            self.close()
            raise

    # -- league ---------------------------------------------------------------------------------------------
    def set_league(self, probs):
        """New opponent distribution (any non-negative weights, one per league entry); used from the next episode
        start of every environment."""
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != self.league_p.shape or (p < 0).any() or p.sum() <= 0:
            raise ValueError("bad league distribution")
        self.league_p[:] = p / p.sum()

    # -- stepping -------------------------------------------------------------------------------------------
    def send(self, g: int, command: int = CMD_STEP):
        assert not self._pending[g], "group already stepping"
        ws = self.group_workers[g]
        self.cmd[ws[0]:ws[-1] + 1] = command
        self._pending[g] = True
        for w in ws:
            self._go[w].release()

    def wait(self, g: int):
        assert self._pending[g], "nothing to wait for"
        sem, t0 = self._done[g], time.time()
        for _ in self.group_workers[g]:
            while not sem.acquire(timeout=2.0):
                self._check(t0)
        self._pending[g] = False
        if self.err.any():
            raise RuntimeError(f"vecenv worker(s) {np.nonzero(self.err)[0].tolist()} crashed (see stderr)")

    def _check(self, t0):
        if self.err.any():
            raise RuntimeError(f"vecenv worker(s) {np.nonzero(self.err)[0].tolist()} crashed (see stderr)")
        dead = [p.name for p in self.procs if not p.is_alive()]
        if dead:
            raise RuntimeError(f"vecenv workers died: {dead[:5]}")
        if time.time() - t0 > self.timeout:
            raise TimeoutError("vecenv workers did not answer in time")

    def reset(self, seed: int | None = None):
        """Restart every environment (new private random streams from `seed`). Returns obs0."""
        for g in range(self.G):
            if self._pending[g]:
                self.wait(g)
        if seed is not None:
            self.seed[0] = seed
        for g in range(self.G):
            self.send(g, CMD_RESET)
        for g in range(self.G):
            self.wait(g)
        return self.obs0

    def step(self, actions0=None, actions1=None):
        """Synchronous step of all environments (actions in the turned frame). Returns (obs0, rew0, done)."""
        if actions0 is not None:
            self.act0[:] = actions0
        if actions1 is not None:
            self.act1[:] = actions1
        for g in range(self.G):
            self.send(g)
        for g in range(self.G):
            self.wait(g)
        return self.obs0, self.rew0, self.done

    # -- shutdown -------------------------------------------------------------------------------------------
    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.cmd[:] = CMD_CLOSE
            for s in self._go:
                s.release()
        except Exception:  # noqa: BLE001
            pass
        deadline = time.time() + 5.0
        for p in self.procs:
            p.join(timeout=max(0.05, deadline - time.time()))
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
                p.join(timeout=2.0)
        _LIVE.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
