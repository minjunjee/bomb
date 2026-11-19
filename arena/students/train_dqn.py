"""Simulated "other teams": the course notebook's baseline DQN (cells 14-19), plus what a student would change.

The default arguments ARE the notebook: the default submission Net (conv 32, conv 32, flatten, linear 128, linear 6),
the two TODOs filled in the obvious way (epsilon-greedy over all 6 actions; r + gamma * (1 - done) * max Q_target),
the default reward (+1 kill, -1 hit), 1500 games against SimpleBot on random maps (A and B in turn), gamma 0.95,
Adam 3e-4, batch 128, a 100,000-step replay buffer, one update every 4 steps once 1000 steps are stored, target
network copied every 2000 steps, smooth L1 loss, epsilon 1.0 -> 0.05 over the first 750 games.

Options = the notebook's "Things to try" (cell 20): another seed, more episodes, a self-bomb penalty / shaping
rewards, double DQN, a bigger network (residual blocks), a mix of opponents, self-play against past copies,
behaviour cloning from a rule-based bot (here zoo.base_cautious stands in for "a bot the student wrote") followed
by DQN fine-tuning, and "pick the better network" with a local evaluation.

Only implementation details differ from the notebook (same algorithm): the replay buffer is a tensor ring on the
device instead of a deque of tuples, preprocess(turn(obs)) is computed once per observation instead of three times,
and maps are random_rules(seed) with seeds drawn below 1,000,000 (the evaluation maps start at 1,000,000).

    nice -n 10 .venv/bin/python -m arena.students.train_dqn --name train_1_base_s0 --seed 0 --device cuda:2

Output: runs/students/<name>/submission.py (its own Net class) + submission.pth (CPU state_dict) + train.log + meta.json
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import importlib.util
import json
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bombcompetition import RULES, BomberEnv, Game, blast_cells, legal_actions, random_rules  # noqa: E402
from bombcompetition.neural import NeuralPolicy, preprocess, turn, turn_action  # noqa: E402
from bots.simple import Policy as SimpleBot  # noqa: E402

from arena.students.student_nets import ARCHS, write_submission_py  # noqa: E402
from arena.zoo.base_cautious import Policy as CautiousBot  # noqa: E402

TRAIN_SEED_LIMIT = 1_000_000  # evaluation maps are random_rules(seed >= 1_000_000): never used here


class RandomBot:
    """The notebook's RandomBot: a random action that is not blocked."""

    def act(self, obs):
        return int(random.choice(legal_actions(obs)))


BOTS = {"simple": SimpleBot, "random": RandomBot, "cautious": CautiousBot}


# ----------------------------------------------------------------------------------------------------------------
# the two TODOs (cell 14), written the obvious way
# ----------------------------------------------------------------------------------------------------------------
def choose_action(net, obs, eps, device, x=None):
    """TODO 1: epsilon-greedy. A REAL action 0-5 for obs (the network sees the turned board)."""
    if random.random() < eps:
        return random.randrange(6)
    if x is None:
        x = preprocess(turn(obs)).to(device)
    with torch.no_grad():
        q = net(x[None])[0]
    return turn_action(int(q.argmax()), obs)


def td_target(r, done, q_next, gamma):
    """TODO 2: r + gamma * max_a' Q_target(s', a'), without the future part when done."""
    return r + gamma * (1.0 - done) * q_next.max(1).values


# ----------------------------------------------------------------------------------------------------------------
# rewards (cell 17 and "Things to try")
# ----------------------------------------------------------------------------------------------------------------
def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _danger(obs):
    """1 / fuse of the most urgent bomb whose blast covers my cell (0 = safe)."""
    worst = 0.0
    for b in obs["bombs"]:
        if obs["me"] in blast_cells(obs["walls"], b["pos"], RULES.radius):
            worst = max(worst, 1.0 / b["fuse"])
    return worst


def make_reward(kind: str, gamma: float):
    def default(prev_obs, action, obs, deaths):  # the notebook's my_reward
        me = obs["player"]
        return float(sum((d.killer == me) - (d.player == me) for d in deaths))

    def selfpen(prev_obs, action, obs, deaths):  # "a bigger penalty for self-bombs (d.killer is None)"
        me = obs["player"]
        r = 0.0
        for d in deaths:
            if d.killer == me:
                r += 1.0
            if d.player == me:
                r -= 2.0 if d.killer is None else 1.0
        return r

    def win(prev_obs, action, obs, deaths):  # +1 whenever the opponent is hit (its own bomb too), -1 when I am hit
        me = obs["player"]
        r = 0.0
        for d in deaths:
            r += -1.0 if d.player == me else 1.0
        return r

    def potential(obs):  # "how good a position is: near the opponent, out of every blast"
        return -0.02 * _manhattan(obs["me"], obs["opp"]) - 0.2 * _danger(obs)

    def pot(prev_obs, action, obs, deaths):  # potential-based shaping on top of the self-bomb penalty
        return selfpen(prev_obs, action, obs, deaths) + gamma * potential(obs) - potential(prev_obs)

    def naive(prev_obs, action, obs, deaths):  # hand-made step bonuses (not potential-based)
        me = obs["player"]
        r = 0.0
        for d in deaths:
            if d.killer == me:
                r += 1.0
            if d.player == me:
                r -= 1.5 if d.killer is None else 1.0
        if action == 5 and prev_obs["opp"] in blast_cells(prev_obs["walls"], prev_obs["me"], RULES.radius):
            r += 0.1  # a bomb that threatens the opponent
        if _danger(obs) > 0:
            r -= 0.05  # standing inside a blast
        r += 0.01 * (_manhattan(prev_obs["me"], prev_obs["opp"]) - _manhattan(obs["me"], obs["opp"]))  # come closer
        return r - 0.002  # do not waste time

    return {"default": default, "selfpen": selfpen, "win": win, "pot": pot, "naive": naive}[kind]


# ----------------------------------------------------------------------------------------------------------------
# replay buffer: the notebook's deque(maxlen=100_000) of (s, a, r, s2, done), stored as tensors
# ----------------------------------------------------------------------------------------------------------------
class Replay:
    def __init__(self, capacity: int, device: str):
        self.s = torch.zeros((capacity, 13, 9, 9), device=device)
        self.s2 = torch.zeros((capacity, 13, 9, 9), device=device)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.n, self.i, self.device = capacity, 0, 0, device

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.s2[i], self.a[i], self.r[i], self.d[i] = s, s2, a, r, d
        self.i = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, k: int):
        idx = np.asarray(random.sample(range(self.n), k))  # like random.sample(memory, BATCH)
        t = torch.from_numpy(idx).to(self.device)
        dev = self.device
        return (self.s[t], torch.from_numpy(self.a[idx]).to(dev), torch.from_numpy(self.r[idx]).to(dev),
                self.s2[t], torch.from_numpy(self.d[idx]).to(dev))


# ----------------------------------------------------------------------------------------------------------------
# behaviour cloning data: games of the teacher bot (the "ambitious student" path)
# ----------------------------------------------------------------------------------------------------------------
def teacher_games(n_games: int, reward_fn, rng: random.Random, noise: float, log):
    """Play the teacher against itself (half of the games), SimpleBot and RandomBot.
    Returns X (N, 13, 9, 9), Y (N,) = the teacher's action on the turned board for every teacher step, and the
    transitions (s, a, r, s2, done) of the steps where the teacher's own action was played (for the replay buffer).
    noise: probability that a random action is played instead (the label stays the teacher's action)."""
    X, Y, demos = [], [], []
    t0 = time.time()
    for g in range(n_games):
        game = Game(random_rules(rng.randrange(TRAIN_SEED_LIMIT)))
        other = (None, None, SimpleBot, RandomBot)[g % 4]
        bots = [CautiousBot(), CautiousBot()]
        teacher = [True, True]
        if other is not None:
            side = (g // 4) % 2
            bots[side], teacher[side] = other(), False
        obs = [game.obs(0), game.obs(1)]
        xs = [preprocess(turn(o)) for o in obs]
        while not game.done:
            want = [int(bots[p].act(obs[p])) for p in (0, 1)]
            played = [random.randrange(6) if teacher[p] and random.random() < noise else want[p] for p in (0, 1)]
            deaths = game.step(*played)
            obs2 = [game.obs(0), game.obs(1)]
            xs2 = [preprocess(turn(o)) for o in obs2]
            ended = float(min(game.lives) <= 0)
            for p in (0, 1):
                if teacher[p]:
                    y = turn_action(want[p], obs[p])
                    X.append(xs[p])
                    Y.append(y)
                    if played[p] == want[p]:
                        demos.append((xs[p], y, float(reward_fn(obs[p], want[p], obs2[p], deaths)), xs2[p], ended))
            obs, xs = obs2, xs2
    log(f"teacher data: {n_games} games, {len(X)} labelled steps, {len(demos)} transitions ({time.time() - t0:.0f}s)")
    return torch.stack(X), torch.tensor(Y), demos


# ----------------------------------------------------------------------------------------------------------------
def local_eval(net, opponents=("simple",), games: int = 20) -> dict:
    """The notebook's "Local evaluation" cell: fixed maps random_rules(1000 + i // 2), each from both sides."""
    policy = NeuralPolicy(net)
    out = {}
    for name in opponents:
        w = d = l = 0
        for i in range(games):
            side = i % 2
            game = Game(random_rules(1000 + i // 2))
            bot = BOTS[name]()
            while not game.done:
                mine, theirs = policy.act(game.obs(side)), bot.act(game.obs(1 - side))
                game.step(*((mine, theirs) if side == 0 else (theirs, mine)))
            res = game.winner()
            w, d, l = w + (res == side), d + (res is None), l + (res == 1 - side)
        out[name] = (w, d, l)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output folder runs/students/<name>")
    ap.add_argument("--out-root", default="runs/students")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="base", choices=sorted(ARCHS))
    # the notebook's hyper-parameters
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=int, default=750)
    ap.add_argument("--memory", type=int, default=100_000)
    ap.add_argument("--learn-start", type=int, default=1000)
    ap.add_argument("--learn-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=2000)
    # what students change
    ap.add_argument("--reward", default="default", choices=["default", "selfpen", "win", "pot", "naive"])
    ap.add_argument("--double", action="store_true", help="double DQN")
    ap.add_argument("--opp", default="simple:1", help="per game: simple | random | cautious | self, with weights")
    ap.add_argument("--sp-every", type=int, default=200, help="self-play: keep a past copy every this many games")
    ap.add_argument("--sp-keep", type=int, default=10)
    ap.add_argument("--bc-games", type=int, default=0, help="behaviour cloning from zoo.base_cautious first")
    ap.add_argument("--bc-epochs", type=int, default=15)
    ap.add_argument("--bc-noise", type=float, default=0.0)
    ap.add_argument("--bc-loss", default="ce", choices=["ce", "margin"], help="ce: cross-entropy; margin: DQfD pre-training")
    ap.add_argument("--margin", type=float, default=0.0, help="> 0: DQfD margin loss on teacher steps during DQN")
    ap.add_argument("--demo-frac", type=float, default=0.0, help="part of every batch taken from the teacher's games")
    ap.add_argument("--pick-best", action="store_true", help="keep the copy with the best local evaluation")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-opp", default="simple")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=1e9, help="stop early (and save) after this wall time")
    args = ap.parse_args(argv)

    folder = ROOT / args.out_root / args.name
    folder.mkdir(parents=True, exist_ok=True)
    log_file = open(folder / "train.log", "a", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    stop = {"flag": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__("flag", True))

    torch.set_num_threads(1)
    device = args.device
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    map_rng = random.Random(SEED * 7919 + 13)  # training maps: seeds below 1,000,000 only

    # the student's submission cell -> submission.py, and Net imported from it (as the notebook does)
    write_submission_py(folder, args.arch)
    spec = importlib.util.spec_from_file_location(f"student_submission_{args.name}", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Net = module.Net

    GAMMA, LR, BATCH = args.gamma, args.lr, args.batch
    EPS_START, EPS_END, EPS_DECAY = args.eps_start, args.eps_end, args.eps_decay
    my_reward = make_reward(args.reward, GAMMA)
    opp_names, opp_weights = zip(*[(p.partition(":")[0], float(p.partition(":")[2] or 1)) for p in args.opp.split(",")])
    eval_opps = tuple(s for s in args.eval_opp.split(",") if s)
    log(f"config: {json.dumps(vars(args))}")

    net = Net().to(device)
    log(f"Net ({args.arch}): {sum(p.numel() for p in net.parameters()):,} parameters")
    t_start = time.time()

    # ---- the ambitious student: clone a rule-based bot first -------------------------------------------------
    demo = None
    bc_info = {}
    if args.bc_games > 0:
        X, Y, demos = teacher_games(args.bc_games, my_reward, map_rng, args.bc_noise, log)
        X, Y = X.to(device), Y.to(device)
        n_val = max(1, len(X) // 20)
        perm = torch.randperm(len(X), device=device)
        val, tr = perm[:n_val], perm[n_val:]
        if args.bc_loss == "margin" or args.demo_frac > 0:  # the teacher's steps as a second, permanent replay buffer
            demo = Replay(len(demos), device)
            demo.s[:] = torch.stack([t[0] for t in demos]).to(device)
            demo.s2[:] = torch.stack([t[3] for t in demos]).to(device)
            demo.a[:], demo.r[:], demo.d[:] = [t[1] for t in demos], [t[2] for t in demos], [t[4] for t in demos]
            demo.n = len(demos)
        del demos
        bc_opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        bc_target = copy.deepcopy(net)
        k = 0
        for epoch in range(args.bc_epochs):
            order = tr[torch.randperm(len(tr), device=device)]
            total, count = 0.0, 0
            for i in range(0, len(order), BATCH):
                if args.bc_loss == "ce":
                    idx = order[i:i + BATCH]
                    loss = F.cross_entropy(net(X[idx]), Y[idx])
                else:  # DQfD pre-training on the teacher's transitions: TD loss + large-margin loss
                    s, a, r, s2, done = demo.sample(min(BATCH, demo.n))
                    q_all = net(s)
                    q = q_all.gather(1, a[:, None]).squeeze(1)
                    with torch.no_grad():
                        y = td_target(r, done, bc_target(s2), GAMMA)
                    m = torch.full_like(q_all, args.margin or 0.8).scatter_(1, a[:, None], 0.0)
                    loss = F.smooth_l1_loss(q, y) + ((q_all + m).max(1).values - q).mean()
                    k += 1
                    if k % 500 == 0:
                        bc_target.load_state_dict(net.state_dict())
                bc_opt.zero_grad()
                loss.backward()
                bc_opt.step()
                total, count = total + loss.item(), count + 1
            with torch.no_grad():
                acc = float((net(X[val]).argmax(1) == Y[val]).float().mean())
            log(f"cloning epoch {epoch + 1}/{args.bc_epochs}: loss {total / max(1, count):.4f}, "
                f"held-out agreement with the teacher {acc:.3f}")
        bc_info = {"bc_samples": int(len(X)), "bc_val_acc": acc}
        del X, Y

    target = copy.deepcopy(net)  # the target network
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    memory = Replay(args.memory, device)  # the replay buffer
    n_demo = int(round(BATCH * args.demo_frac)) if demo is not None else 0

    def learn():
        s, a, r, s2, done = memory.sample(BATCH - n_demo)
        if n_demo:
            ds, da, dr, ds2, dd = demo.sample(n_demo)
            s, a, r, s2, done = (torch.cat(p) for p in ((ds, s), (da, a), (dr, r), (ds2, s2), (dd, done)))
        q_all = net(s)
        q = q_all.gather(1, a[:, None]).squeeze(1)  # Q(s, a)
        with torch.no_grad():
            if args.double:  # double DQN: the online network picks a', the target network scores it
                best = net(s2).argmax(1, keepdim=True)
                y = r + GAMMA * (1.0 - done) * target(s2).gather(1, best).squeeze(1)
            else:
                y = td_target(r, done, target(s2), GAMMA)  # TODO 2
        loss = F.smooth_l1_loss(q, y)
        if n_demo and args.margin > 0:  # DQfD: the teacher's action must beat the others by a margin
            qd, ad = q_all[:n_demo], a[:n_demo]
            m = torch.full_like(qd, args.margin).scatter_(1, ad[:, None], 0.0)
            loss = loss + ((qd + m).max(1).values - qd.gather(1, ad[:, None]).squeeze(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def cpu_state():
        return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    best_score, best_state, best_ep, evals = -1, None, None, []

    def evaluate(ep):
        nonlocal best_score, best_state, best_ep
        res = local_eval(net, eval_opps, args.eval_games)
        score = sum(v[0] for v in res.values())
        evals.append({"episode": ep, "result": res})
        mark = ""
        if score >= best_score:  # ties: the later copy
            best_score, best_state, best_ep, mark = score, cpu_state(), ep, "  <- best so far"
        log(f"local evaluation after {ep} games: " + ", ".join(f"vs {k} {w}W {d}D {l}L" for k, (w, d, l) in res.items()) + mark)

    if args.pick_best:
        evaluate(0)

    pool = deque(maxlen=args.sp_keep)  # self-play: past copies of the network
    t0, recent, steps, ep_done = time.time(), deque(maxlen=50), 0, 0
    stats = deque(maxlen=200)  # (opponent, result, steps, self_bomb, greedy bombs, greedy actions)
    for ep in range(args.episodes):
        if stop["flag"] or time.time() - t_start > args.max_minutes * 60:
            log(f"stopped early after {ep} games ({'signal' if stop['flag'] else 'time limit'})")
            break
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY)
        kind = opp_names[0] if len(opp_names) == 1 else random.choices(opp_names, opp_weights)[0]
        if kind == "self":
            if not pool or ep - pool_ep >= args.sp_every:
                pool.append(copy.deepcopy(net).eval())
                pool_ep = ep
            opponent = NeuralPolicy(random.choice(pool)).act  # a past copy, as the notebook suggests
        else:
            opponent = BOTS[kind]().act
        env = BomberEnv(opponent=opponent, player=ep % 2, reward_fn=my_reward,
                        rules=random_rules(map_rng.randrange(TRAIN_SEED_LIMIT)))  # A and B in turn
        obs, _ = env.reset()
        x = preprocess(turn(obs)).to(device)
        done, ep_steps, n_greedy, n_bomb = False, 0, 0, 0
        while not done:
            a = choose_action(net, obs, eps, device, x)  # TODO 1
            obs2, r, terminated, truncated, info = env.step(a)
            x2 = preprocess(turn(obs2)).to(device)
            memory.add(x, turn_action(a, obs), r, x2, float(terminated))  # transition(obs, a, r, obs2, terminated)
            obs, x, done, steps, ep_steps = obs2, x2, terminated or truncated, steps + 1, ep_steps + 1
            if memory.n >= args.learn_start and steps % args.learn_every == 0:
                learn()
            if steps % args.target_every == 0:
                target.load_state_dict(net.state_dict())
        me = obs["player"]
        mine, theirs = info["lives"][me], info["lives"][1 - me]
        res = "W" if mine > theirs else "L" if mine < theirs else "D"  # a self-bomb is a loss
        self_bomb = int(RULES.lives - mine - info["kills"][1 - me] > 0)
        recent.append(res)
        stats.append((kind, res, ep_steps, self_bomb))
        ep_done = ep + 1
        if (ep + 1) % 50 == 0:
            W, D, L = (recent.count(v) for v in "WDL")
            last = list(stats)[-50:]
            log(f"episode {ep + 1}/{args.episodes} (eps {eps:.2f}): last {len(recent)} games {W}W {D}D {L}L  "
                f"self-bombs {sum(s[3] for s in last)}  steps/game {sum(s[2] for s in last) / len(last):.0f}  "
                f"total steps {steps}  ({time.time() - t0:.0f}s)")
        if args.pick_best and (ep + 1) % args.eval_every == 0 and ep + 1 < args.episodes:
            evaluate(ep + 1)

    if args.pick_best:
        evaluate(ep_done)
        state, picked = best_state, best_ep
    else:
        state, picked = cpu_state(), ep_done
    torch.save({k: v.detach().cpu() for k, v in state.items()}, folder / "submission.pth")  # the notebook's last cell
    last = list(stats)
    per_opp = {}
    for kind in set(s[0] for s in last):
        rows = [s for s in last if s[0] == kind]
        per_opp[kind] = {"games": len(rows), "W": sum(s[1] == "W" for s in rows), "D": sum(s[1] == "D" for s in rows),
                         "L": sum(s[1] == "L" for s in rows), "self_bombs": sum(s[3] for s in rows),
                         "steps_per_game": sum(s[2] for s in rows) / len(rows)}
    meta = {"name": args.name, "args": vars(args), "episodes_done": ep_done, "env_steps": steps,
            "picked_copy_after_games": picked, "local_evals": evals, "last_200_training_games": per_opp,
            "minutes": (time.time() - t_start) / 60, "params": sum(p.numel() for p in net.parameters()),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), **bc_info}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"saved {folder / 'submission.pth'} (copy after {picked} games; {ep_done} games, {steps} steps, "
        f"{meta['minutes']:.1f} min); last 200 training games: {json.dumps(per_opp)}")


if __name__ == "__main__":
    main()
