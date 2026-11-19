"""Minimal, verified behaviour-cloning data generator: the teacher labels every visited state of both players.

    nice -n 10 .venv/bin/python -m arena.mini.gen --teacher zoo.x_matrix --out runs/mini/data/xm0 --games 6000 --procs 120 --seed0 20000

Each game: a random pair from PAIRS (who actually plays), sides at random. An actor is "T" (the teacher), "T:<eps>"
(the teacher, but a random legal action with probability eps), or any harness spec. For every step and both players:
x = preprocess(turn(obs)) (float32, exact), y = turn_action(teacher.act(obs), obs), a = the action actually played
(turned frame), and after the game z = the result for that player (+1 win, 0 draw, -1 loss) and the steps left.
Shards: <out>/shard_XXX.npz with x, y, a, z, left, meta (map seed, step, player). Map seeds must stay < 1_000_000.
"""
import argparse, os, random, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from bombcompetition import Game, legal_actions, random_rules
from bombcompetition.neural import preprocess, turn, turn_action
from arena.core import bot_factory

RULE = ["zoo.base_cautious", "zoo.ta_smart_a", "zoo.ta_smart_b", "zoo.evader", "zoo.ta_final_a", "zoo.ta_final_b", "zoo.param:@"]
PAIRS = ([("T", r) for r in RULE] * 2 + [("T:0.1", r) for r in RULE] + [("T:0.25", r) for r in RULE[:4]]
         + [("T", "T"), ("T:0.1", "T"), ("T:0.1", "T:0.1"), ("T:0.3", "T:0.1"), ("T", "simple"), ("T:0.1", "random"),
            ("T", "noisy:0.2:zoo.base_cautious"), ("T", "noisy:0.15:zoo.ta_smart_a"), ("T", "noisy:0.15:zoo.ta_final_a")])


def work(args):
    wid, seeds, teacher_spec, out, pairs = args
    T = bot_factory(teacher_spec)
    X, Y, A, Z, L, M = [], [], [], [], [], []
    skipped = 0
    for s in seeds:
        random.seed(s * 7919 + 13)
        pair = list(pairs[random.randrange(len(pairs))])
        if random.random() < 0.5:
            pair.reverse()
        pair = [k.replace("@", str(random.randrange(200))) for k in pair]
        g = Game(random_rules(s))
        teachers = [T(), T()]
        bots = [None if k.startswith("T") else bot_factory(k)() for k in pair]
        eps = [float(k[2:]) if k.startswith("T:") else 0.0 for k in pair]
        start = len(Y)
        idx_p = []
        while not g.done:
            acts = []
            for p in (0, 1):
                obs = g.obs(p)
                try:
                    ta = int(teachers[p].act(obs))
                except Exception:
                    ta = None
                    skipped += 1
                if bots[p] is not None:
                    try:
                        a = int(bots[p].act(obs))
                    except Exception:
                        a = 0
                elif ta is None or random.random() < eps[p]:
                    a = int(random.choice(legal_actions(obs)))
                else:
                    a = ta
                if ta is not None:
                    X.append(preprocess(turn(obs)).numpy()); Y.append(turn_action(ta, obs)); A.append(turn_action(a, obs))
                    M.append((s, g.t, p)); idx_p.append(p)
                acts.append(a)
            g.step(*acts)
        w = g.winner()
        for k, p in enumerate(idx_p):
            Z.append(0 if w is None else (1 if w == p else -1))
            L.append(g.t - M[start + k][1])
    np.savez(f"{out}/shard_{wid:03d}.npz", x=np.stack(X).astype(np.float32), y=np.array(Y, np.int8), a=np.array(A, np.int8),
             z=np.array(Z, np.int8), left=np.array(L, np.int16), meta=np.array(M, np.int32))
    return len(Y), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=6000); ap.add_argument("--procs", type=int, default=96)
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--pairs", default=None, help="override: 'a|b,a|b,...'")
    args = ap.parse_args()
    assert args.seed0 + args.games < 1_000_000
    pairs = [tuple(p.split("|")) for p in args.pairs.split(",")] if args.pairs else PAIRS
    import multiprocessing as mp
    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.games))
    chunks = [(w, seeds[w::args.procs], args.teacher, args.out, pairs) for w in range(args.procs)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.procs) as pool:
        res = pool.map(work, chunks, chunksize=1)
    n, sk = sum(r[0] for r in res), sum(r[1] for r in res)
    print(f"{n} samples ({sk} skipped) from {args.games} games in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
