"""A PARAMETRISED bot family: one implementation, many behaviours (opponent population for training and grading).

    zoo.param            class Policy with the default preset (DEFAULT, the strongest one)
    zoo.param:chaser     a named preset, see PRESETS (ordered from terrible to decent)
    zoo.param:17         random_config(17): a configuration sampled deterministically from the integer seed
    zoo.param:chaser+eps=0.3+chase=flee     a preset or seed with some parameters overridden

    from arena.zoo.param import Policy, PRESETS, random_config, config_of, make_policy
    bot = make_policy("hunter")()      # make_policy(arg) -> a zero-argument factory
    bot = Policy({"chase": "flee"})    # a dict: DEFAULTS with these keys replaced

The parameters are independent behavioural axes (DEFAULTS below lists them). In short:

    danger        what it knows about explosions: none | own (only its own bombs) | opp (only the opponent's) |
                  static (every blast cell is "dangerous", no timing: the classic danger-map bot) |
                  nochain (exact fuses, but it does not know chain reactions) | exact
    horizon       how many steps ahead it plans its flight (explosions later than that are ignored)
    block         now | always: is the opponent's body in the way only this step, or in all of its flight plans
    escape        none | timed: does it check its own way out before it drops a bomb
    trigger       never | adjacent | near (Manhattan <= 2, through walls, like SimpleBot) | cornered (in the blast and
                  the opponent has at most one free neighbour) | inline (the opponent is inside the blast)
    trap_k        k >= 0: it also bombs when the bomb leaves the opponent at most k surviving first moves (exact search)
    bomb_rate     random bombing per step
    bombs         how many of the 3 bombs it uses at once
    rebomb        SimpleBot's bug: while the trigger holds it keeps pressing BOMB (it freezes on its own bomb)
    freeze        steps it stands still after dropping a bomb
    chase         bfs | greedy (Manhattan, runs into walls) | keep (distance keep_dist) | flee | random | centre |
                  patrol | still
    eps           rate of random but safe moves;  blunder: rate of completely random actions;  bump: those may
                  also walk into walls
    shrink_lead   it heads inside this many steps before a sudden-death shrink (0: it ignores the shrink)
    shrink_model  its flight search knows the shrink (new walls, vanished bombs, shorter blasts, the teleport)
    foresee       0 | 1 | 2: it avoids cells where the opponent's next bomb (dropped now / after one move) traps it
    wary          it knows the opponent's body can spoil a plan (a cell the opponent stands on is closed, and when
                  both players step onto one cell nobody moves): it bombs only with a way out the opponent cannot
                  reach in time, and flees along such ways when there is one
    urgent        it leaves a blast zone by the quickest way and does not walk into one (otherwise it goes on with
                  its usual movement as long as a way out is left)
    deny          when the opponent is running from a blast it steps onto / contests the opponent's exit cell
    dirs          direction preference for ties, e.g. (1, 2, 3, 4); None = random

Timing convention: "time k" = the end of the k-th step from now (the step being decided is k = 1). A bomb whose obs
fuse is f explodes at time f, a bomb dropped now at time RULES.fuse; a bomb is in the way during the moves of the
steps 1..its explosion time. All cell sets are 81-bit integers (bit r * 9 + c), which keeps one act around 0.05 ms.
All randomness comes from the global `random` module (the harness seeds it per game).
"""

import functools
import math
import os
import random
import zlib

import numpy as np

from bombcompetition import RULES

N = 9
NN = N * N
FUSE, RADIUS, MAXB = int(RULES.fuse), int(RULES.radius), int(RULES.max_bombs)
EVERY, MAX_STEPS = int(RULES.shrink_every or 0), int(RULES.max_steps)
FULL = (1 << NN) - 1
BIT = [1 << i for i in range(NN)]
ROW = [i // N for i in range(NN)]
COL = [i % N for i in range(NN)]
STEP = (0, -N, N, -1, 1, 0)  # change of the cell index per action
STRICT = bool(os.environ.get("PARAM_STRICT"))  # tests: let exceptions through (the harness then counts a fault)

DEFAULTS = {
    "danger": "exact", "horizon": 9, "block": "always", "escape": "timed",
    "trigger": "inline", "trap_k": -1, "bomb_rate": 0.0, "bombs": 3, "rebomb": False, "freeze": 0,
    "chase": "bfs", "keep_dist": 3, "eps": 0.0, "blunder": 0.0, "bump": False,
    "shrink_lead": 0, "shrink_model": False, "foresee": 0, "wary": False, "urgent": False, "deny": False,
    "dirs": None,
}
CHOICES = {
    "danger": ("none", "own", "opp", "static", "nochain", "exact"),
    "block": ("now", "always"),
    "escape": ("none", "timed"),
    "trigger": ("never", "adjacent", "near", "cornered", "inline"),
    "chase": ("bfs", "greedy", "keep", "flee", "random", "centre", "patrol", "still"),
}


def _preset(**kw):
    return {**DEFAULTS, **kw}


# Ordered from terrible to decent (the measured round-robin order is in the final report of this bot).
PRESETS = {
    # never does anything
    "afk": _preset(danger="none", trigger="never", chase="still", escape="none"),
    # uniformly random over the 6 actions, walls included
    "drunk": _preset(danger="none", blunder=1.0, bump=True, escape="none", trigger="never", chase="random"),
    # SimpleBot: bombs when you are within 2 cells and then freezes on the bomb
    "freezer": _preset(danger="none", trigger="near", rebomb=True, escape="none", chase="random"),
    # runs at you and bombs next to you, never looks at any bomb
    "kamikaze": _preset(danger="none", trigger="adjacent", escape="none", chase="bfs", bombs=1, block="now"),
    # random walk, jumps away only two steps before an explosion, a rare random bomb
    "wanderer": _preset(danger="nochain", horizon=2, trigger="never", bomb_rate=0.03, chase="random", block="now"),
    # litters bombs, dodges its own ones with exact timing but does not look at yours
    "litterbug": _preset(danger="own", trigger="adjacent", bomb_rate=0.2, chase="random", block="now", eps=0.1),
    # the classic danger-map bot: no timing, greedy chase, fixed direction order
    "naive": _preset(danger="static", trigger="inline", chase="greedy", block="now", eps=0.05, dirs=(1, 2, 3, 4)),
    # never bombs, runs away, dodges exactly: a draw machine
    "coward": _preset(trigger="never", chase="flee", shrink_lead=10, shrink_model=True),
    # sits in the centre and bombs what comes into its blast
    "camper": _preset(chase="centre", bombs=2, eps=0.05, shrink_model=True),
    # keeps 3 cells away, one bomb at a time
    "sniper": _preset(chase="keep", keep_dist=3, bombs=1, eps=0.1),
    # = the yardstick zoo.base_cautious: exact dodging, bombs when you are in its blast, BFS chase
    "chaser": _preset(eps=0.1),
    # a guess at the TA "SmartBot": dodges, chases, bombs only when the opponent cannot escape
    "smart": _preset(trigger="never", trap_k=0, eps=0.05, shrink_model=True),
    # chaser + sure-kill bombs + does not walk into a trap the opponent can close right now
    "trapper": _preset(trap_k=0, foresee=1, eps=0.05, shrink_model=True, urgent=True),
    # trapper + looks one opponent move further, squeezes (trap_k=1), walks inside before the shrink
    "hunter": _preset(trap_k=1, foresee=2, eps=0.02, shrink_model=True, shrink_lead=4, wary=True, urgent=True,
                      deny=True),
}
DEFAULT = "hunter"


# ----------------------------------------------------------------------------------------------------------------
# boards: everything that depends only on the walls, cached
# ----------------------------------------------------------------------------------------------------------------
def _rings():
    ring, inner = [0] * 6, [0] * 6
    for i in range(NN):
        q = min(ROW[i], COL[i], N - 1 - ROW[i], N - 1 - COL[i])
        for k in range(6):
            if q == k:
                ring[k] |= BIT[i]
            elif q > k:
                inner[k] |= BIT[i]
    return ring, inner


_RING, _INNER = _rings()  # ring k of the board, and the cells inside it


class _Board:
    __slots__ = ("open", "blast", "_after", "_tele", "_centre", "_way")

    def __init__(self, open_mask):
        self.open = open_mask
        blast = [0] * NN  # blast[i]: the cells a bomb on cell i hits
        for i in range(NN):
            if (open_mask >> i) & 1:
                x = BIT[i]
                for d in (-N, N, -1, 1):
                    j = i
                    for _ in range(RADIUS):
                        j += d
                        if j < 0 or j >= NN or not (open_mask >> j) & 1:
                            break
                        x |= BIT[j]
                blast[i] = x
        self.blast = blast
        self._after, self._tele, self._centre, self._way = {}, {}, -1, None

    def after(self, k):
        """The board after sudden-death shrink k."""
        b = self._after.get(k)
        if b is None:
            b = self._after[k] = _board(self.open & ~_RING[k])
        return b

    def tele(self, k):
        """{ring cell: the inner cells it can be teleported to, nearest first (the engine's order)}."""
        t = self._tele.get(k)
        if t is None:
            inner = [i for i in range(NN) if (self.open >> i) & (_INNER[k] >> i) & 1]
            t = {}
            for p in range(NN):
                if (self.open >> p) & (_RING[k] >> p) & 1:
                    r, c = ROW[p], COL[p]
                    t[p] = sorted(inner, key=lambda x: (abs(ROW[x] - r) + abs(COL[x] - c), x))[:8]
            self._tele[k] = t
        return t

    def _nearest(self, r, c):
        cells = [i for i in range(NN) if (self.open >> i) & 1]
        return min(cells, key=lambda x: (abs(ROW[x] - r) + abs(COL[x] - c), x)) if cells else N * (N // 2) + N // 2

    def centre(self):
        if self._centre < 0:
            self._centre = self._nearest(N // 2, N // 2)
        return self._centre

    def waypoints(self):
        """The open cells nearest to the 4 corners of the open area, clockwise."""
        if self._way is None:
            cells = [i for i in range(NN) if (self.open >> i) & 1] or [self.centre()]
            r0, r1 = min(ROW[i] for i in cells), max(ROW[i] for i in cells)
            c0, c1 = min(COL[i] for i in cells), max(COL[i] for i in cells)
            self._way = [self._nearest(r, c) for r, c in ((r0, c0), (r0, c1), (r1, c1), (r1, c0))]
        return self._way


_BY_MASK, _BY_BYTES = {}, {}


def _board(open_mask):
    b = _BY_MASK.get(open_mask)
    if b is None:
        if len(_BY_MASK) > 20000:
            _BY_MASK.clear()
            _BY_BYTES.clear()
        b = _BY_MASK[open_mask] = _Board(open_mask)
    return b


def _board_of(walls):
    if walls.__class__ is not np.ndarray or walls.dtype != np.bool_:
        walls = np.asarray(walls, dtype=bool)
    key = walls.tobytes()
    b = _BY_BYTES.get(key)
    if b is None:
        if walls.shape != (N, N):
            raise ValueError("this bot plays 9x9 boards")
        m = 0
        for i, w in enumerate(walls.ravel().tolist()):
            if not w:
                m |= BIT[i]
        b = _BY_BYTES[key] = _board(m)
    return b


# ----------------------------------------------------------------------------------------------------------------
# exact bomb timing
# ----------------------------------------------------------------------------------------------------------------
def _chain(blast, t):
    """t = {bomb cell: time it explodes}, in place: a bomb inside an earlier blast goes off with it."""
    again = True
    while again:
        again = False
        for c, tc in t.items():
            bc = blast[c]
            for d, td in t.items():
                if td > tc and (bc >> d) & 1:
                    t[d] = tc
                    again = True
    return t


def _timeline(B, fuses, chain, step, shrink, horizon, obst):
    """fuses = {bomb cell: obs fuse} (not empty). -> (K, ok, free, ks, teleport):
    ok[k] = cells one can stand on at time k (open and not exploding then), free[k] = cells one can walk into during
    step k (open, no bomb left there), for k = 1..K; K = the last explosion looked at (at most `horizon`).
    obst = cells that can never be entered. With `shrink`, a sudden-death shrink at the end of step ks < K is
    modelled: the ring becomes wall, bombs on it vanish, the others are stopped by the new walls."""
    blast = B.blast
    t = dict(fuses)
    if chain and len(t) > 1:
        _chain(blast, t)
    K = max(t.values())
    if shrink and EVERY:
        k = step // EVERY + 1
        ks = EVERY * k - step
        if ks < K and ks < horizon and EVERY * k < MAX_STEPS and B.open & _INNER[k]:
            return _timeline_shrink(B, fuses, t, chain, horizon, obst, k, ks)
    if K > horizon:
        K = horizon
    opn = B.open
    hit = [0] * (K + 2)
    last = [0] * (K + 2)  # last[k]: the bombs that leave the board at time k
    for c, tc in t.items():
        if tc <= K:
            hit[tc] |= blast[c]
            last[tc] |= BIT[c]
        else:
            last[K + 1] |= BIT[c]
    ok = [0] * (K + 1)
    free = [0] * (K + 1)
    pres = last[K + 1] | obst
    for j in range(K, 0, -1):
        pres |= last[j]
        free[j] = opn & ~pres
        ok[j] = opn & ~hit[j]
    return K, ok, free, 0, None


def _timeline_shrink(B, fuses, t, chain, horizon, obst, k, ks):
    ring = _RING[k]
    B2 = B.after(k)
    blast, blast2 = B.blast, B2.blast
    late, lateb = {}, 0  # the bombs still on the board after the shrink: their chains are recomputed
    for c, tc in t.items():
        if tc > ks and not (ring >> c) & 1:
            late[c] = fuses[c]
            lateb |= BIT[c]
    if chain and len(late) > 1:
        _chain(blast2, late)
    K = max(late.values()) if late else ks
    if K > horizon:
        K = horizon
    hit = [0] * (K + 2)
    last = [0] * (K + 2)
    for c, tc in t.items():
        if tc <= ks:
            hit[tc] |= blast[c]
            last[tc] |= BIT[c]
        elif (ring >> c) & 1:
            last[ks] |= BIT[c]  # vanishes with the ring
    for c, tc in late.items():
        if tc <= K:
            hit[tc] |= blast2[c]
            last[tc] |= BIT[c]
        else:
            last[K + 1] |= BIT[c]
    ok = [0] * (K + 1)
    free = [0] * (K + 1)
    pres = last[K + 1] | obst
    o1, o2 = B.open, B2.open
    for j in range(K, 0, -1):
        pres |= last[j]
        o = o2 if j > ks else o1
        free[j] = o & ~pres
        ok[j] = o & ~hit[j]
    return K, ok, free, ks, (ring & o1, o2, B.tele(k), lateb)


def _unteleport(S, tp):
    """S = good cells right after the shrink -> the cells that were good just before the teleport."""
    ring_open, open2, tele, lateb = tp
    S &= open2
    out = S
    m = ring_open
    while m:
        low = m & -m
        m ^= low
        cand = tele[low.bit_length() - 1]
        dest = cand[0]
        for d in cand:
            if not (lateb >> d) & 1:
                dest = d
                break
        if (S >> dest) & 1:
            out |= low
    return out


def _alive1(tl):
    """The cells where standing at time 1 (after this step's move and explosions) still leaves a way to survive
    every explosion of the timeline. Backward induction over the time steps."""
    K, ok, free, ks, tp = tl
    S = ok[K]
    for k in range(K - 1, 0, -1):
        x = S & free[k + 1]
        S |= (x << 1) | (x >> 1) | (x << N) | (x >> N)
        if k == ks:
            S = _unteleport(S, tp)
        S &= ok[k]
    return S


def _guard1(tl, oi, mi):
    """Like _alive1, but only with plans the opponent's body cannot spoil: a cell the opponent could stand on or
    step onto at step k is not entered at step k. -> (the good cells at time 1, the cells it can contest now)."""
    K, ok, free, ks, tp = tl
    cloud = [0] * (K + 2)
    c = BIT[oi]
    c |= ((c << 1) | (c >> 1) | (c << N) | (c >> N)) & free[1] & ~BIT[mi]
    first = cloud[1] = c
    for k in range(2, K + 1):
        c |= ((c << 1) | (c >> 1) | (c << N) | (c >> N)) & free[k]
        cloud[k] = c
    S = ok[K]
    for k in range(K - 1, 0, -1):
        x = S & free[k + 1] & ~cloud[k + 1]
        S |= (x << 1) | (x >> 1) | (x << N) | (x >> N)
        if k == ks:
            S = _unteleport(S, tp)
        S &= ok[k]
    return S, first


def _exit_times(tl, cells):
    """For each cell: standing there at time 1, the earliest time one can stand on a cell that no bomb of the
    timeline hits any more (1 = the cell itself is clear)."""
    K, ok, free = tl[0], tl[1], tl[2]
    clear = [0] * (K + 2)
    c = FULL
    for j in range(K, 0, -1):
        c &= ok[j]
        clear[j] = c
    out = []
    for n in cells:
        f = BIT[n] & ok[1]
        d = 1
        while f and not f & clear[d] and d < K:
            d += 1
            f = (f | (((f << 1) | (f >> 1) | (f << N) | (f >> N)) & free[d])) & ok[d]
        out.append(d if f else K + 1)
    return out


def _lifetime(tl, cell):
    """For how many time steps does a player standing on `cell` at time 1 survive at best (K = all of them)?"""
    K, ok, free = tl[0], tl[1], tl[2]
    f = BIT[cell] & ok[1]
    if not f:
        return 0
    for k in range(2, K + 1):
        x = f & FULL
        f = (f | (((x << 1) | (x >> 1) | (x << N) | (x >> N)) & free[k])) & ok[k]
        if not f:
            return k - 1
    return K


def _escapes(B, fuses, step, who, other):
    """The number of first actions (stay or a move) after which the player on `who` can still survive these bombs
    (exact timing, chains and shrink included); `other` is the cell of the other player."""
    if fuses:
        tl = _timeline(B, fuses, True, step, True, 99, 0)
        S = _alive1(tl)
        free1 = tl[2][1] & ~BIT[other]
    else:
        S, free1 = FULL, B.open & ~BIT[other]
    n = (S >> who) & 1
    for d in (-N, N, -1, 1):
        q = who + d
        if (free1 >> q) & (S >> q) & 1:
            n += 1
    return n


def _dist_to(src, passable, cells):
    """Walking steps from the cells of `src` to each cell of the list `cells` through `passable` (99: no way)."""
    want = 0
    for n in cells:
        want |= BIT[n]
    out = {}
    f = seen = src
    d = 0
    while True:
        h = f & want
        if h:
            for n in cells:
                if (h >> n) & 1:
                    out[n] = d
            want &= ~h
            if not want:
                break
        f = ((f << 1) | (f >> 1) | (f << N) | (f >> N)) & passable & ~seen
        if not f:
            break
        seen |= f
        d += 1
    return [out.get(n, 99) for n in cells]


# ----------------------------------------------------------------------------------------------------------------
# the policy
# ----------------------------------------------------------------------------------------------------------------
class Policy:
    def __init__(self, config=None):
        cfg = self.config = config_of(config)
        self.danger = cfg["danger"]
        self.horizon = cfg["horizon"]
        self.block_always = cfg["block"] == "always"
        self.escape = cfg["escape"] != "none"
        self.trigger = cfg["trigger"]
        self.trap_k = cfg["trap_k"]
        self.bomb_rate = cfg["bomb_rate"]
        self.bombs = cfg["bombs"]
        self.rebomb = cfg["rebomb"]
        self.freeze = cfg["freeze"]
        self.chase = cfg["chase"]
        self.keep_dist = cfg["keep_dist"]
        self.eps = cfg["eps"]
        self.blunder = cfg["blunder"]
        self.bump = cfg["bump"]
        self.shrink_lead = cfg["shrink_lead"]
        self.shrink_model = cfg["shrink_model"]
        self.foresee = cfg["foresee"]
        self.wary = cfg["wary"]
        self.urgent = cfg["urgent"]
        self.deny = cfg["deny"]
        dirs = cfg["dirs"]
        self.rank = None if dirs is None else {a: i for i, a in enumerate(tuple(dirs) + (0,))}
        self.frozen = 0
        self.wp = random.randrange(4) if self.chase == "patrol" else 0
        self.wp_age = 0

    def act(self, obs):
        if STRICT:
            return int(self._act(obs))
        try:
            a = int(self._act(obs))
            if 0 <= a <= 5:
                return a
        except Exception:  # noqa: BLE001  (a bot must never raise)
            pass
        return 0

    # ------------------------------------------------------------------------------------------------------------
    def _act(self, obs):
        B = _board_of(obs["walls"])
        opn = B.open
        me, opp = obs["me"], obs["opp"]
        mi = me[0] * N + me[1]
        oi = opp[0] * N + opp[1]
        player, step = obs["player"], obs["step"]
        allf = {}  # every bomb: cell -> fuse
        bbits = ownb = 0
        mine = 0
        for b in obs["bombs"]:
            p = b["pos"]
            c = p[0] * N + p[1]
            f = b["fuse"]
            allf[c] = f if f > 1 else 1
            bbits |= BIT[c]
            if b["owner"] == player:
                mine += 1
                ownb |= BIT[c]
        theirs = len(allf) - mine
        blockers = bbits | BIT[oi]
        cands = [(0, mi)]  # (action, my cell after it) for STAY and the moves that are not blocked
        for a in (1, 2, 3, 4):
            n = mi + STEP[a]
            if (opn >> n) & 1 and not (blockers >> n) & 1:
                cands.append((a, n))

        # ---- blunders and bugs --------------------------------------------------------------------------------
        if self.blunder and random.random() < self.blunder:
            if self.bump:
                return random.randrange(6)
            return random.choice([a for a, _ in cands] + [5])
        if self.frozen > 0:
            self.frozen -= 1
            return 0
        blast = B.blast
        gap = abs(ROW[mi] - ROW[oi]) + abs(COL[mi] - COL[oi])
        have = mine < self.bombs
        trig = False
        if have:
            t = self.trigger
            if t == "inline":
                trig = (blast[mi] >> oi) & 1 == 1
            elif t == "near":
                trig = gap <= 2
            elif t == "adjacent":
                trig = gap == 1
            elif t == "cornered":
                if (blast[mi] >> oi) & 1:
                    free = opn & ~bbits & ~BIT[mi]
                    trig = sum((free >> (oi + d)) & 1 for d in (-N, N, -1, 1)) <= 1
            if not trig and self.bomb_rate and random.random() < self.bomb_rate:
                trig = True
            if trig and self.rebomb:
                return 5  # SimpleBot's bug: no check at all, not even "am I standing on my bomb already"

        # ---- which cells are good to stand on after this step --------------------------------------------------
        danger = self.danger
        static = danger == "static"
        chain = danger != "nochain"
        body = BIT[oi] if self.block_always else 0
        seen, unseen, D, tl = allf, 0, 0, None
        if static:
            h = self.horizon
            for c, f in allf.items():
                if f <= h:
                    D |= blast[c]
            if not (D >> mi) & 1:
                S = FULL & ~D  # never walk into a blast zone
            else:  # in a blast zone: the neighbours on a shortest way out of it (no timing)
                passable = opn & ~blockers
                want = 0
                for _, n in cands[1:]:
                    want |= BIT[n]
                S = 0
                f = reach = passable & ~D
                while f:
                    if f & want:
                        S = f & want
                        break
                    f = ((f << 1) | (f >> 1) | (f << N) | (f >> N)) & passable & ~reach
                    reach |= f
        else:
            if danger == "none":
                seen, unseen = {}, bbits
            elif danger == "own":
                seen = {c: f for c, f in allf.items() if (ownb >> c) & 1}
                unseen = bbits & ~ownb
            elif danger == "opp":
                seen = {c: f for c, f in allf.items() if not (ownb >> c) & 1}
                unseen = ownb
            if seen:
                tl = _timeline(B, seen, chain, step, self.shrink_model, self.horizon, unseen | body)
                S = _alive1(tl)
            else:
                S = FULL

        # ---- bomb? ----------------------------------------------------------------------------------------------
        if have and not (bbits >> mi) & 1:
            kill = False
            if not trig and self.trap_k >= 0 and (gap <= 4 or bbits):
                f2 = dict(allf)
                f2[mi] = FUSE
                e1 = _escapes(B, f2, step, oi, mi)
                if e1 <= self.trap_k and e1 < _escapes(B, allf, step, oi, mi):
                    trig = True
                    kill = e1 == 0
            if trig:
                ok = True
                if self.escape:
                    if static:  # a cell outside every blast within reach before the new bomb goes off (no timing)
                        passable = opn & ~blockers
                        f = BIT[mi]
                        for _ in range(FUSE - 1):
                            f |= ((f << 1) | (f >> 1) | (f << N) | (f >> N)) & passable
                        ok = f & ~(D | blast[mi]) != 0
                    else:
                        s2 = dict(seen)
                        s2[mi] = FUSE
                        obst = unseen | body
                        tb = _timeline(B, s2, chain, step, self.shrink_model, 99, obst)
                        if self.wary and not kill:  # only with a way out that the opponent cannot spoil
                            ok = (_guard1(tb, oi, mi)[0] >> mi) & 1
                        else:
                            ok = (_alive1(tb) >> mi) & 1
                        if ok and self.foresee and gap <= 5 and theirs < MAXB and not (bbits >> oi) & 1:
                            s2[oi] = FUSE  # ... and if the opponent answers with a bomb right now?
                            ok = (_alive1(_timeline(B, s2, chain, step, self.shrink_model, 99, obst)) >> mi) & 1
                if ok:
                    self.frozen = self.freeze
                    return 5

        # ---- move -------------------------------------------------------------------------------------------------
        safe = [x for x in cands if (S >> x[1]) & 1]
        if not safe:
            if tl is None:
                return random.choice(cands)[0]
            tl = _timeline(B, seen, chain, step, self.shrink_model, self.horizon, unseen)
            if body:  # the opponent may step out of the way
                S = _alive1(tl)
                safe = [x for x in cands if (S >> x[1]) & 1]
            if not safe:  # doomed as far as it can see: live as long as possible
                life = [_lifetime(tl, n) for _, n in cands]
                best = max(life)
                return random.choice([x for x, v in zip(cands, life) if v == best])[0]
        if len(safe) == 1:
            return safe[0][0]

        if tl is not None and self.wary:  # the plans the opponent's body cannot spoil come first
            G, first = _guard1(tl, oi, mi)
            here = (G >> mi) & 1
            sure = [x for x in safe if (G >> x[1]) & 1 and (here or x[0] == 0 or not (first >> x[1]) & 1)]
            if sure:
                safe = sure
                if len(safe) == 1:
                    return safe[0][0]
        if self.foresee and not static and gap <= 5 and theirs < MAXB:
            safe = self._robust(B, safe, seen, chain, step, unseen | body, mi, oi, bbits)
            if len(safe) == 1:
                return safe[0][0]
        if self.eps and random.random() < self.eps:
            return random.choice(safe)[0]

        sc = self._scores(B, safe, mi, oi, bbits, step)
        if tl is not None:
            if self.urgent:  # first of all out of the blast zones
                sc = list(zip(_exit_times(tl, [n for _, n in safe]), sc))
            if self.deny and gap == 2 and (S >> mi) & 1:
                dn = self._deny(B, allf, step, safe, mi, oi)
                if dn:
                    sc = list(zip(dn, sc))
        best = min(sc)
        top = [x[0] for x, v in zip(safe, sc) if v == best]
        if len(top) == 1:
            return top[0]
        rank = self.rank
        if rank is None:
            return random.choice(top)
        return min(top, key=rank.__getitem__)

    # ------------------------------------------------------------------------------------------------------------
    def _robust(self, B, safe, seen, chain, step, obst, mi, oi, bbits):
        """Keep the safe actions that survive the most "what if" bombs of the opponent: a bomb dropped right now on
        its cell, and (foresee 2) one dropped next step on a neighbouring cell."""
        opn = B.open
        hyp = []
        if not (bbits >> oi) & 1:
            hyp.append((oi, FUSE))
        if self.foresee >= 2:
            for d in (-N, N, -1, 1):
                q = oi + d
                if (opn >> q) & 1 and not (bbits >> q) & 1 and q != mi:
                    hyp.append((q, FUSE + 1))
        if not hyp:
            return safe
        risk = [0] * len(safe)
        shrink, horizon = self.shrink_model, self.horizon
        for q, f in hyp:
            s2 = dict(seen)
            s2[q] = f
            Sh = _alive1(_timeline(B, s2, chain, step, shrink, horizon, obst))
            for i, (_, n) in enumerate(safe):
                if n == q:
                    n = mi  # we would bump into each other: I stay where I am
                if not (Sh >> n) & 1:
                    risk[i] += 1
        low = min(risk)
        return [x for x, r in zip(safe, risk) if r == low]

    # ------------------------------------------------------------------------------------------------------------
    def _deny(self, B, allf, step, safe, mi, oi):
        """The opponent is two cells away and has to run from a blast: per safe action of mine, how many ways out it
        keeps when I step onto (or contest) one of its exit cells. None when it has room anyway."""
        blast = B.blast
        for c in allf:
            if (blast[c] >> oi) & 1:
                break
        else:
            return None
        tl = _timeline(B, allf, True, step, True, 99, 0)
        So = _alive1(tl)
        free1 = tl[2][1] & ~BIT[mi]
        stay = (So >> oi) & 1
        exits = [q for q in (oi - N, oi + N, oi - 1, oi + 1) if (free1 >> q) & (So >> q) & 1]
        if stay + len(exits) > 2:
            return None
        return [stay + sum(q != n for q in exits) for _, n in safe]

    # ------------------------------------------------------------------------------------------------------------
    def _scores(self, B, safe, mi, oi, bbits, step):
        """A score per safe action (lower = better) from the movement style."""
        opn = B.open
        cells = [n for _, n in safe]
        passable = (opn & ~bbits & ~BIT[oi]) | BIT[mi]
        pen = None
        if self.shrink_lead and EVERY:
            k = step // EVERY + 1
            if EVERY * k - step <= self.shrink_lead and EVERY * k < MAX_STEPS and opn & _INNER[k]:
                ring = _RING[k]
                if (ring >> mi) & 1:  # get inside before the ring closes
                    return _dist_to(opn & _INNER[k] & ~bbits & ~BIT[oi], passable, cells)
                pen = [50 if (ring >> n) & 1 else 0 for n in cells]
        style = self.chase
        if style == "bfs":
            sc = _dist_to(BIT[oi], passable, cells)
        elif style == "keep":
            k = self.keep_dist
            sc = [abs(d - k) for d in _dist_to(BIT[oi], passable, cells)]
        elif style == "flee":
            sc = [-d for d in _dist_to(BIT[oi], passable, cells)]
        elif style == "greedy":
            r, c = ROW[oi], COL[oi]
            sc = [abs(ROW[n] - r) + abs(COL[n] - c) for n in cells]
        elif style == "centre":
            sc = _dist_to(BIT[B.centre()], passable, cells)
        elif style == "patrol":
            way = B.waypoints()
            w = way[self.wp % len(way)]
            self.wp_age += 1
            if w == mi or self.wp_age > 24:
                self.wp += 1
                self.wp_age = 0
                w = way[self.wp % len(way)]
            sc = _dist_to(BIT[w], passable, cells)
            if min(sc) >= 99:
                self.wp += 1
                self.wp_age = 0
        elif style == "random":
            sc = [random.random() + (a == 0) for a, _ in safe]
        else:  # "still"
            sc = [int(a != 0) for a, _ in safe]
        if pen:
            sc = [s + p for s, p in zip(sc, pen)]
        return sc


# ----------------------------------------------------------------------------------------------------------------
# configurations
# ----------------------------------------------------------------------------------------------------------------
def random_config(seed):
    """A configuration sampled deterministically from an integer seed. A hidden "skill" in [0, 1] tilts every
    quality axis (so the family holds terrible, mediocre and decent bots), the style axes are independent."""
    rng = random.Random(int(seed))
    skill = rng.random()

    def tilted(options, sigma=0.9):  # options ordered from bad to good
        pos = skill * (len(options) - 1)
        w = [math.exp(-(((i - pos) / sigma) ** 2) / 2) + 0.03 for i in range(len(options))]
        return rng.choices(options, w)[0]

    cfg = dict(DEFAULTS)
    cfg["danger"] = tilted(["none", "own", "opp", "static", "nochain", "exact", "exact"])
    cfg["horizon"] = tilted([1, 2, 3, 4, 9, 9])
    cfg["block"] = tilted(["now", "always"])
    cfg["escape"] = tilted(["none", "timed", "timed"])
    cfg["trigger"] = rng.choices(["never", "adjacent", "near", "cornered", "inline"], [1, 2, 2, 1, 4])[0]
    cfg["trap_k"] = rng.choice([0, 1, 2]) if rng.random() < 0.1 + 0.6 * skill else -1
    cfg["bomb_rate"] = tilted([0.3, 0.15, 0.05, 0.0, 0.0, 0.0])
    cfg["bombs"] = rng.choice([1, 2, 3, 3])
    cfg["rebomb"] = rng.random() < 0.3 * (1 - skill) ** 2
    cfg["freeze"] = tilted([3, 2, 1, 0, 0, 0, 0])
    cfg["chase"] = rng.choices(CHOICES["chase"], [8, 2, 2, 2, 2, 2, 2, 1])[0]
    cfg["keep_dist"] = rng.choice([2, 3, 4])
    cfg["eps"] = tilted([0.5, 0.3, 0.2, 0.1, 0.05, 0.0])
    cfg["blunder"] = tilted([0.4, 0.2, 0.1, 0.03, 0.0, 0.0, 0.0])
    cfg["bump"] = rng.random() < 0.5 * (1 - skill)
    cfg["shrink_lead"] = rng.choice([0, 0, 3, 8, 20])
    cfg["shrink_model"] = rng.random() < skill
    cfg["foresee"] = tilted([0, 0, 0, 1, 2])
    cfg["wary"] = tilted([False, False, False, True])
    cfg["urgent"] = tilted([False, False, True])
    cfg["deny"] = rng.random() < 0.5 * skill
    cfg["dirs"] = None if rng.random() < 0.5 else tuple(rng.sample((1, 2, 3, 4), 4))
    return cfg


def _value(key, text):
    """A parameter value written as text (for 'name+key=value')."""
    old = DEFAULTS[key]
    low = text.strip().lower()
    if key == "dirs":
        return None if low in ("", "none", "random") else tuple(int(ch) for ch in low if ch.isdigit())
    if isinstance(old, bool):
        return low in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(float(low))
    if isinstance(old, float):
        return float(low)
    return low


def _clean(cfg):
    """A full, valid configuration: unknown keys dropped, bad values replaced by the default."""
    out = dict(DEFAULTS)
    for key, v in cfg.items():
        if key not in DEFAULTS:
            continue
        try:
            if key in CHOICES:
                v = v if v in CHOICES[key] else DEFAULTS[key]
            elif key == "dirs":
                v = tuple(int(a) for a in v) if v else None
                if v is not None and sorted(v) != [1, 2, 3, 4]:
                    v = None
            elif key in ("rebomb", "bump", "shrink_model", "wary", "urgent", "deny"):
                v = bool(v)
            elif key in ("eps", "blunder", "bomb_rate"):
                v = min(1.0, max(0.0, float(v)))
            elif key == "horizon":
                v = min(99, max(1, int(v)))
            elif key == "bombs":
                v = min(MAXB, max(0, int(v)))
            elif key == "trap_k":
                v = min(5, max(-1, int(v)))
            elif key == "foresee":
                v = min(2, max(0, int(v)))
            else:  # freeze, keep_dist, shrink_lead
                v = min(99, max(0, int(v)))
        except Exception:  # noqa: BLE001
            v = DEFAULTS[key]
        out[key] = v
    return out


def config_of(arg=None):
    """None (the default preset), a preset name, an integer seed (int or text), 'name+key=value+...' or a dict of
    parameters -> a full configuration dict. Any other text is hashed to a seed, so every spec works."""
    if arg is None or arg == "":
        return dict(PRESETS[DEFAULT])
    if isinstance(arg, dict):
        return _clean(arg)
    if isinstance(arg, int):
        return _clean(random_config(arg))
    head, *extra = str(arg).strip().split("+")
    head = head.strip()
    if head in PRESETS:
        cfg = dict(PRESETS[head])
    elif head.lstrip("-").isdigit():
        cfg = random_config(int(head))
    elif head == "":
        cfg = dict(PRESETS[DEFAULT])
    else:
        cfg = random_config(zlib.crc32(head.encode()))
    for item in extra:
        key, _, text = item.partition("=")
        key = key.strip()
        if key in DEFAULTS:
            try:
                cfg[key] = _value(key, text)
            except Exception:  # noqa: BLE001
                pass
    return _clean(cfg)


def make_policy(arg=None):
    """arg: a preset name or an integer seed (as text), optionally '+key=value' overrides -> a zero-argument
    factory (the harness spec zoo.param:<arg>)."""
    factory = functools.partial(Policy, config_of(arg))
    factory.config = factory.args[0]
    return factory
