"""SmartBot: the hidden leaderboard opponent (rule-based, one step of scoring, a little random).

Every step each of the 6 actions gets a score and the best one is played:

  dodge   the cell I end on must not explode in this step, and from it I must still be able to walk to a cell that
          no bomb on the board will hit (exact timing: chains, bombs block the way, the shrink). The sooner I can
          stand on such a cell the better, a cell that explodes late is better than one that explodes soon, and a
          way out that the opponent could walk into (block it, or step on the same cell as I do) counts as risky.
  chase   fewer steps to the opponent is better.
  trap    BOMB is only scored when the opponent stands in the blast and I can get away: the fewer first moves that
          still save it, the higher the score (none left = a kill); with many ways out, only now and then.
  shrink  shortly before the map shrinks, cells that are about to turn into walls are bad.

It never plans for bombs the opponent may drop later (only: when it drops a bomb, it checks that it could also get
away if the opponent dropped one in the same step), and it does not look for traps that take more than one bomb.

Time k = the end of the k-th step from now (k = 1 is the step being decided): a bomb with fuse f in obs explodes
at time f, a bomb dropped now at time RULES.fuse. Cells are numbers: row * 9 + col.
"""

import random

from bombcompetition import DELTAS, RULES

N = 9
STEP = tuple(dr * N + dc for dr, dc in DELTAS)  # what each action adds to my cell number
EVERY = RULES.shrink_every or 10 ** 9
NEVER = 99

W_SAFE = 1.0     # per step I still need to reach a cell that no bomb hits
W_WHEN = 1.0     # / steps until the cell I end on explodes
W_DIST = 0.3     # per step between me and the opponent
W_RING = 1.0     # per step to the cells that stay open, when the shrink is near
RING_SOON = 8    # ... "near" = this many steps
W_RISK = 2.0     # if the opponent could get in the way of my way out (x2 if already in this step)
NOISE = 0.25
TRAP = {0: 100.0, 1: 6.0, 2: 4.0}  # BOMB bonus by the number of first moves that still save the opponent
PRESSURE = 3.2                     # ... with more ways out: this times a random number (a bomb now and then)


def _tables(open_):
    """For every open cell: the open cells next to it, and the cells a bomb on it hits."""
    nb, bl = {}, {}
    for i in open_:
        nb[i] = tuple(i + s for s in STEP[1:5] if i + s in open_)
        hit = [i]
        for s in STEP[1:5]:
            for k in range(1, RULES.radius + 1):
                if i + s * k not in open_:
                    break
                hit.append(i + s * k)
        bl[i] = frozenset(hit)
    return nb, bl


def _steps(nb, sources, blocked=()):
    """{cell: steps from the nearest source}, not entering blocked cells."""
    d = dict.fromkeys(sources, 0)
    queue = list(d)
    for c in queue:
        for n in nb[c]:
            if n not in d and n not in blocked:
                d[n] = d[c] + 1
                queue.append(n)
    return d


def _chain(bombs, bl):
    """{bomb cell: time it explodes}: a bomb inside an earlier blast goes off with it."""
    t = dict(bombs)
    for _ in range(len(t) - 1):
        for c in t:
            for d in t:
                if t[d] > t[c] and d in bl[c]:
                    t[d] = t[c]
    return t


class Policy:
    def __init__(self):
        self.key = None

    def act(self, obs):
        try:
            return int(self._act(obs))
        except Exception:  # never crash a game
            return 0

    # ---- tables for the walls of the moment (rebuilt only when the map shrinks) -------------------------------
    def _prep(self, walls, step):
        w = walls.tolist()
        open_ = {r * N + c for r in range(N) for c in range(N) if not w[r][c]}
        self.nb, self.bl = _tables(open_)
        self.dist = {}
        k = step // EVERY + 1  # the ring that closes next
        depth = lambda i: min(i // N, i % N, N - 1 - i // N, N - 1 - i % N)
        ring, inner = [i for i in open_ if depth(i) == k], sorted(i for i in open_ if depth(i) > k)
        self.shrinks = bool(inner) and k * EVERY < RULES.max_steps
        self.ring, self.to_inner = frozenset(), {}
        if self.shrinks:
            self.ring = frozenset(ring)
            self.nb2, self.bl2 = _tables(set(inner))
            # where a player on the ring is put: the nearest open cell inside (ties: smallest row, col)
            self.land = {i: sorted(inner, key=lambda x: (abs(x // N - i // N) + abs(x % N - i % N), x)) for i in ring}
            self.to_inner = _steps(self.nb, inner)

    # ---- when and where it explodes ----------------------------------------------------------------------------
    def _sched(self, bombs, ks):
        """bombs {cell: fuse} -> (hit, gone, first, last): hit[k] = the cells that explode at time k, gone[bomb cell]
        = the time that bomb leaves the board, first/last[cell] = the first/last time the cell is hit.
        If the map shrinks at time ks before the last explosion, later blasts use the new walls and the bombs on
        the ring are gone."""
        t = _chain(bombs, self.bl)
        late = ks < max(t.values())
        if late:
            t2 = _chain({c: f for c, f in bombs.items() if t[c] > ks and c not in self.ring}, self.bl2)
        hit, gone = {}, {}
        for c, tc in t.items():
            if tc <= ks:
                gone[c] = tc
                hit.setdefault(tc, set()).update(self.bl[c])
            elif c in self.ring:
                gone[c] = ks
            else:
                gone[c] = t2[c]
                hit.setdefault(t2[c], set()).update(self.bl2[c])
        first, last = {}, {}
        for k in sorted(hit):
            for c in hit[k]:
                first.setdefault(c, k)
                last[c] = k
        if hit and max(hit) > ks:  # a closing cell is no place to wait: who stands there is moved into what comes later
            for c in self.ring:
                last[c] = max(last.get(c, 0), ks)
        return hit, gone, first, last

    def _run(self, starts, sched, ks, block, lead):
        """starts {first action: the cell it puts the player on}. For every action: (the earliest time the player can
        stand on a cell that is never hit again, 0 = there is no way out; the last time it is still alive).
        The cell `block` is never entered, and a cell is only entered at time k if lead[cell] > k."""
        hit, gone, _, last = sched
        nb, front = self.nb, {}
        safe, life = dict.fromkeys(starts, 0), dict.fromkeys(starts, 0)
        todo = len(starts)
        for k in range(1, max(hit) + 1 if hit else 2):
            h = hit.get(k, ())
            new = {}
            if k == 1:
                for a, c in starts.items():
                    if c not in h:
                        new[c] = new.get(c, 0) | 1 << a
            else:
                for c, m in front.items():
                    if c not in h:
                        new[c] = new.get(c, 0) | m
                    for n in nb[c]:  # a bomb that is still there blocks the way
                        if n not in h and n != block and gone.get(n, 0) < k and lead.get(n, NEVER) > k:
                            new[n] = new.get(n, 0) | m
            if k == ks:  # the shrink: new walls, a player on the ring is moved inside
                nb, front = self.nb2, {}
                for c, m in new.items():
                    if c in self.ring:
                        c = next((x for x in self.land[c] if gone.get(x, 0) <= k), self.land[c][0])
                    front[c] = front.get(c, 0) | m
            else:
                front = new
            ok = up = 0
            for c, m in front.items():
                up |= m
                if last.get(c, 0) < k:
                    ok |= m
            for a in starts:
                if up >> a & 1:
                    life[a] = k
                    if ok >> a & 1 and not safe[a]:
                        safe[a] = k
                        todo -= 1
            if not todo or not front:
                break
        return {a: (safe[a], life[a]) for a in starts}

    def _escape(self, starts, sched, ks, other, lead):
        """For every first action (safe, life, risk). risk 0 = there is a way out that stays ahead of the other player
        (lead = its steps to every cell), so it can neither block it nor step on a cell at the same time as I do;
        1 = it could get in the way later; 2 = already the first move may fail; 3 = it must step aside for me."""
        free = {a: c for a, c in starts.items() if a in (0, 5) or lead.get(c, NEVER) > 1}
        out = {a: (*r, 0) for a, r in self._run(free, sched, ks, other, lead).items() if r[0]}
        if len(out) < len(starts):
            rest = {a: c for a, c in starts.items() if a not in out}
            res = self._run(rest, sched, ks, other, {})
            if out or any(r[0] for r in res.values()):
                out.update((a, (*r, 1 if a in free else 2)) for a, r in res.items())
            else:  # no way out: maybe the other player steps aside
                out.update((a, (*r, 3)) for a, r in self._run(rest, sched, ks, -1, {}).items())
        return out

    # ---- the score of one action ---------------------------------------------------------------------------------
    def _score(self, cell, res, first, dist, ks):
        safe, life, risk = res
        if not safe:  # hit for sure: at least live as long as possible
            return -1000.0 + 20 * life + random.random()
        s = -W_SAFE * (safe - 1) - W_RISK * risk - W_DIST * dist.get(cell, 10) + NOISE * random.random()
        if cell in first:
            s -= W_WHEN / first[cell]
        if ks <= RING_SOON:
            s -= W_RING * self.to_inner.get(cell, 0)
        return s

    def _act(self, obs):
        walls, step, p = obs["walls"], obs["step"], obs["player"]
        key = (walls.tobytes(), step // EVERY)
        if key != self.key:
            self._prep(walls, step)
            self.key = key
        me, opp = obs["me"][0] * N + obs["me"][1], obs["opp"][0] * N + obs["opp"][1]
        bombs = {b["pos"][0] * N + b["pos"][1]: b["fuse"] for b in obs["bombs"]}
        mine = sum(b["owner"] == p for b in obs["bombs"])
        ks = EVERY - step % EVERY if self.shrinks else NEVER  # the map shrinks at time ks
        nb = self.nb

        ends = {0: me}  # action -> the cell it puts me on
        for a in (1, 2, 3, 4):
            n = me + STEP[a]
            if n in nb and n not in bombs and n != opp:
                ends[a] = n
        dist = self.dist.get(opp) or self.dist.setdefault(opp, _steps(nb, [opp]))  # the opponent's steps to every cell

        if bombs:
            sched = self._sched(bombs, ks)
            res, first = self._escape(ends, sched, ks, opp, dist), sched[2]
        else:
            res, first = dict.fromkeys(ends, (1, NEVER, 0)), {}
        score = {a: self._score(c, res[a], first, dist, ks) for a, c in ends.items()}

        # BOMB: only when the opponent stands in the blast and I can get away from it
        if mine < RULES.max_bombs and me not in bombs and opp in self.bl[me]:
            sched = self._sched({**bombs, me: RULES.fuse}, ks)
            mres = self._escape({5: me}, sched, ks, opp, dist)[5]
            if mres[0] and mres[2] < 3:
                oends = {0: opp}
                for a in (1, 2, 3, 4):
                    n = opp + STEP[a]
                    if n in nb and n not in bombs and n != me:
                        oends[a] = n
                ways = sum(1 for r in self._run(oends, sched, ks, me, {}).values() if r[0])
                if ways and len(bombs) - mine < RULES.max_bombs and opp not in bombs:
                    # it may drop a bomb in this same step: I must get away from both
                    both = self._sched({**bombs, me: RULES.fuse, opp: RULES.fuse}, ks)
                    if not self._run({5: me}, both, ks, opp, {})[5][0]:
                        ways = -1
                if ways == 0 or ways > 0 and not mres[2]:  # a kill, or a way out that cannot be blocked
                    bonus = TRAP[ways] if ways in TRAP else PRESSURE * random.random()
                    score[5] = self._score(me, mres, sched[2], dist, ks) + bonus

        return max(score, key=score.get)
