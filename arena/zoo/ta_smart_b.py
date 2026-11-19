"""ta_smart_b: a TA-style "SmartBot" with an aggressive temperament (rule-based, no learning).

How it plays
1. It never takes an action after which it cannot outrun the bombs on the board (exact timing, chain reactions and
   the sudden-death shrink included).
2. Kill first: if a bomb dropped now (together with the bombs already ticking: up to 3 of its own, used to cut the
   ways out) leaves the opponent no way to survive, it drops it. It also stands in the way when that alone dooms
   the opponent.
3. Pressure: it bombs whenever the opponent is inside the blast it would make, or when the opponent is already
   running from a bomb and the new one takes away most of the cells it could still run to, as long as it keeps an
   escape for itself.
4. Otherwise it walks toward the opponent by the shortest path, preferring cells from which it could bomb (or kill)
   on the next step if the opponent did not move.
5. Random tie-breaks and a small share of random (safe) moves.

Its typical hand-written imperfection (on purpose): when it plans its own escape it treats the opponent as standing
still and does not think about bombs the opponent may drop next. So it can be body-blocked, out-raced to its escape
cell, or sealed into a pocket by a second bomb. Its only memory: a move that did not happen (both players wanted
the same cell) is not tried again on the next step if there is another way.

Timing convention: "time k" = the end of the k-th step from now (the step being decided is k = 1). A bomb whose obs
fuse is f explodes at time f; a bomb dropped now explodes at time RULES.fuse. Sets of cells are 81-bit integers
(bit row * 9 + col), so one step of "where can I be" is a handful of integer operations.

`zoo.ta_smart_b:key=value,...` (make_policy) gives variants, e.g. `zoo.ta_smart_b:eps=0.15` or `:press=0.5`.
"""

import random

import numpy as np

from bombcompetition import RULES

_FUSE = int(RULES.fuse)
_RAD = int(RULES.radius)
_MAXB = int(RULES.max_bombs)
_EVERY = int(RULES.shrink_every or 0)
_MAXT = int(RULES.max_steps)
_DELTA = (0, -9, 9, -1, 1)  # cell-index change of STAY, UP, DOWN, LEFT, RIGHT
_Z8 = (0,) * 8
_MAPS = {}  # walls bytes -> _Map (walls only change at a shrink)
_ERRORS = [0]  # exceptions swallowed by act() (must stay 0; tests look at it)
STRICT = False  # tests set this to True to see exceptions


class _Map:
    """What depends only on the walls: the open cells and the blast of a bomb on every cell."""

    __slots__ = ("open", "bl", "rings")

    def __init__(self, walls):
        w = walls.tolist()
        openm = 0
        bl = [0] * 81
        for r in range(1, 8):
            for c in range(1, 8):
                if w[r][c]:
                    continue
                i = r * 9 + c
                openm |= 1 << i
                mask = 1 << i
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    for k in range(1, _RAD + 1):
                        rr, cc = r + dr * k, c + dc * k
                        if not (0 <= rr < 9 and 0 <= cc < 9) or w[rr][cc]:
                            break
                        mask |= 1 << (rr * 9 + cc)
                bl[i] = mask
        self.open = openm
        self.bl = bl
        self.rings = {}

    def ring(self, k):
        """Shrink number k: (open cells of ring k, open cells inside it, the latter as a sorted list of
        (row, col, index)), or None if nothing inside is open (then the board does not shrink)."""
        info = self.rings.get(k, 0)
        if info == 0:
            ringm = innerm = 0
            inner = []
            for r in range(1, 8):
                for c in range(1, 8):
                    i = r * 9 + c
                    if not (self.open >> i) & 1:
                        continue
                    g = min(r, c, 8 - r, 8 - c)
                    if g == k:
                        ringm |= 1 << i
                    elif g > k:
                        innerm |= 1 << i
                        inner.append((r, c, i))
            info = (ringm, innerm, inner) if inner else None
            self.rings[k] = info
        return info


def _schedule(bl, cells, times, ks, ringm, tend):
    """When does every bomb go off? Chains included; a bomb on the closing ring that has not exploded when the board
    shrinks (time ks) just vanishes, and nothing explodes after the last step of the game (time tend).
    Returns (hit, bge, last, tf): hit[k] = cells exploding at time k, bge[k] = bombs that are still on the board
    during the move of step k, last = the last explosion time, tf[i] = time of bomb i (0 = it never explodes)."""
    n = len(cells)
    hit = [0] * 8
    bge = [0] * 8
    t = list(times)
    last = 0
    if ks > 6:
        if n > 1:
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    ti = t[i]
                    bi = bl[cells[i]]
                    for j in range(n):
                        if t[j] > ti and (bi >> cells[j]) & 1:
                            t[j] = ti
                            changed = True
        for i in range(n):
            ti = t[i]
            c = cells[i]
            bit = 1 << c
            if ti > tend:  # the game is over before it goes off
                for k in range(1, 8):
                    bge[k] |= bit
                t[i] = 0
                continue
            hit[ti] |= bl[c]
            for k in range(1, ti + 1):
                bge[k] |= bit
            if ti > last:
                last = ti
        return hit, bge, last, t
    done = [False] * n
    for _ in range(n):
        i = -1
        ti = 99
        for j in range(n):
            if not done[j] and t[j] < ti:
                ti = t[j]
                i = j
        done[i] = True
        c = cells[i]
        bit = 1 << c
        if ti > ks and (ringm >> c) & 1:  # still there when the ring closes: it vanishes
            for k in range(1, ks + 1):
                bge[k] |= bit
            t[i] = 0
            continue
        blc = bl[c]
        for j in range(n):
            if not done[j] and t[j] > ti and (blc >> cells[j]) & 1:
                t[j] = ti
        hit[ti] |= blc
        for k in range(1, ti + 1):
            bge[k] |= bit
        if ti > last:
            last = ti
    return hit, bge, last, t


def _later(hit, last):
    """L[k] = cells that explode at some time after k."""
    L = [0] * 8
    acc = 0
    k = last
    while k > 0:
        acc |= hit[k]
        L[k - 1] = acc
        k -= 1
    return L


def _tele(inner, i, avoid):
    """Where the shrink puts a player standing on ring cell i (the engine's rule)."""
    r0, c0 = divmod(i, 9)
    best = -1
    bd = 99
    for r, c, j in inner:
        if (avoid >> j) & 1:
            continue
        d = abs(r - r0) + abs(c - c0)
        if d < bd:
            bd = d
            best = j
    if best < 0:
        for r, c, j in inner:
            d = abs(r - r0) + abs(c - c0)
            if d < bd:
                bd = d
                best = j
    return best


def _tele_mask(F, ringm, inner, avoid):
    R = F & ringm
    if not R:
        return F
    F &= ~ringm
    while R:
        low = R & -R
        F |= 1 << _tele(inner, low.bit_length() - 1, avoid)
        R ^= low
    return F


def _plan(openm, bge, last, blk, ks, ring, who, other):
    """For player `who`, with the other player standing still on cell `other`:
    fr[k] = the cells `who` may walk into during step k (k = 0..last+1; blk = -1: the other player is no obstacle),
    sh = what the search needs to know about a shrink before the last explosion (None if there is none)."""
    if ks > last:
        return [openm & ~bge[k] & blk for k in range(last + 2)], None
    ringm, innerm, inner = ring
    after = bge[ks + 1] & innerm  # bombs that are still there after the shrink
    ob = 1 << other
    o2 = (1 << _tele(inner, other, after)) if ringm & ob else ob  # the other player is moved inside too
    avoid = after | (o2 if who == 1 else ob)  # player 0 is moved first
    blk2 = ~o2 if blk != -1 else -1
    fr = [(openm & blk if k <= ks else innerm & blk2) & ~bge[k] for k in range(last + 2)]
    return fr, (ks, ringm, innerm, inner, avoid)


def _wait(F, k, hit, sh):
    """Standing on F during step k: what is left at time k (0 = hit)."""
    F &= ~hit[k]
    if F and sh is not None and sh[0] == k:
        F = _tele_mask(F, sh[1], sh[3], sh[4])
    return F


def _escape(F, k, last, hit, later, fr, sh):
    """F = the cells I may stand on at time k-1 (alive so far). The first time at which I can stand on a cell that no
    bomb will hit any more, or 0 if there is no way to survive."""
    if sh is None:
        if F & ~later[k - 1]:
            return k - 1
        while k <= last:
            F = (((F << 1) | (F >> 1) | (F << 9) | (F >> 9)) & fr[k] | F) & ~hit[k]
            if not F:
                return 0
            if F & ~later[k]:
                return k
            k += 1
        return 0
    ks, ringm, innerm, inner, avoid = sh
    safe = F & ~later[k - 1]
    if k - 1 < ks:
        safe &= innerm  # a cell of the closing ring is no place to rest
    if safe:
        return k - 1
    while k <= last:
        F = (((F << 1) | (F >> 1) | (F << 9) | (F >> 9)) & fr[k] | F) & ~hit[k]
        if not F:
            return 0
        if k == ks:
            F = _tele_mask(F, ringm, inner, avoid)
        safe = F & ~later[k]
        if k < ks:
            safe &= innerm
        if safe:
            return k
        k += 1
    return 0


def _fate(G, k, last, hit, fr, sh, fr1=None):
    """G = the cells a player may stand on at time k-1. The cells it can stand on alive at time `last` (0 = it cannot
    survive). fr1: the cells it may walk into during step k, if they differ from fr[k]."""
    if fr1 is not None:
        G = (((G << 1) | (G >> 1) | (G << 9) | (G >> 9)) & fr1 | G) & ~hit[k]
        if G and sh is not None and sh[0] == k:
            G = _tele_mask(G, sh[1], sh[3], sh[4])
        k += 1
    if sh is None:
        while G and k <= last:
            G = (((G << 1) | (G >> 1) | (G << 9) | (G >> 9)) & fr[k] | G) & ~hit[k]
            k += 1
        return G
    ks = sh[0]
    while G and k <= last:
        G = (((G << 1) | (G >> 1) | (G << 9) | (G >> 9)) & fr[k] | G) & ~hit[k]
        if G and k == ks:
            G = _tele_mask(G, sh[1], sh[3], sh[4])
        k += 1
    return G


def _dists(src, free, want):
    """Walking distance from the cells `src` over the cells `free` to every cell of `want`: {cell: steps}."""
    out = {}
    F = src
    seen = src
    d = 0
    while F:
        got = F & want
        if got:
            want &= ~got
            while got:
                low = got & -got
                out[low.bit_length() - 1] = d
                got ^= low
            if not want:
                break
        F = ((F << 1) | (F >> 1) | (F << 9) | (F >> 9)) & free & ~seen
        seen |= F
        d += 1
    return out


class Policy:
    eps = 0.04      # share of random (safe) moves while nothing else is going on
    press = 1.0     # how often it bombs when the opponent is inside the blast and it can get away
    comfort = 1     # 1: a pressure bomb needs an escape that still works after losing one step
    flee = 0        # inside a blast zone: 0 run to safety at once (ties: random), 1 ties toward the opponent,
                    # 2 ties away from it, 3 keep chasing while there is a step to spare
    enter = 1       # 1: it walks through blast zones when the timing leaves a step to spare
    cut = 1         # 1: it bombs to take escape cells away from an opponent that is already running
    block = 1       # 1: it uses its body to keep a doomed opponent doomed
    hunt = 1        # 1: it prefers cells from which the next bomb would kill / reach an opponent that stays put

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.debug = None  # tests put a dict here to receive what the bot saw
        self._tried = None  # (step, my cell, target cell) of my last move, to notice a move that did not happen

    def act(self, obs):
        try:
            a = int(self._act(obs))
            if 1 <= a <= 4:
                me = obs["me"]
                m = me[0] * 9 + me[1]
                self._tried = (obs["step"], m, m + _DELTA[a])
            else:
                self._tried = None
            return a
        except Exception:
            if STRICT:
                raise
            _ERRORS[0] += 1
            self._tried = None
            return 0

    # ----------------------------------------------------------------------------------------------------------
    def _act(self, obs):
        walls = obs["walls"]
        try:
            key = walls.tobytes()
        except AttributeError:
            walls = np.asarray(walls, dtype=bool)
            key = walls.tobytes()
        mp = _MAPS.get(key)
        if mp is None:
            if len(_MAPS) > 256:
                _MAPS.clear()
            mp = _MAPS[key] = _Map(walls)
        openm = mp.open
        bl = mp.bl
        me = obs["me"]
        op = obs["opp"]
        m = me[0] * 9 + me[1]
        o = op[0] * 9 + op[1]
        mb = 1 << m
        ob = 1 << o
        nob = ~ob
        nmb = ~mb
        P = obs["player"]
        s = obs["step"]

        cells = []
        times = []
        own = []
        allb = 0
        mine = 0
        for b in obs["bombs"]:
            p = b["pos"]
            c = p[0] * 9 + p[1]
            f = b["fuse"]
            cells.append(c)
            times.append(1 if f < 1 else (6 if f > 6 else f))
            allb |= 1 << c
            if b["owner"] == P:
                own.append(True)
                mine += 1
            else:
                own.append(False)

        # sudden death: the board shrinks at the end of step `left` from now (time ks)
        ks = 99
        ring = None
        left = 0
        if _EVERY:
            left = _EVERY - s % _EVERY
            if s + left < _MAXT:
                ring = mp.ring(s // _EVERY + 1)
                if ring is not None:
                    ks = left
        ringm = ring[0] if ring is not None else 0
        tend = _MAXT - s  # explosions later than this never happen

        if cells:
            hit, bge, last, tf = _schedule(bl, cells, times, ks, ringm, tend)
            later = _later(hit, last)
            danger = later[0]
            fr, sh = _plan(openm, bge, last, nob, ks, ring, P, o)
        else:
            hit = bge = later = _Z8
            last = 0
            tf = ()
            danger = 0
            fr = sh = None

        # My last move did not happen: the opponent wanted the same cell (then neither moves). Do not run into
        # it again if there is another way.
        contested = -1
        tr = self._tried
        if tr is not None and tr[0] == s - 1 and tr[1] == m:
            contested = tr[2]

        # ---- what each move leads to: (action, cell, time at which I am safe for good, where I am at time 1),
        #      only the survivable ones
        legal = []
        opts = []
        again = None
        for a in (0, 1, 2, 3, 4):
            n = m + _DELTA[a]
            nbit = 1 << n
            if a and (not openm & nbit or allb & nbit or n == o):
                continue
            legal.append((a, n))
            if not last:
                x = (a, n, 1, nbit)
            else:
                x = None
                F1 = _wait(nbit, 1, hit, sh)
                if F1:
                    t = _escape(F1, 2, last, hit, later, fr, sh)
                    if t:
                        x = (a, n, t, F1)
            if x is not None:
                if n == contested:
                    again = x
                else:
                    opts.append(x)
        if again is not None and (not opts or (len(opts) == 1 and not last)):
            # no other way (or nothing else to do than standing still): try the contested cell again
            opts.append(again)
            again = None
        squeezed = False
        if not opts and last:  # the opponent is in my way: plan as if it will step aside
            squeezed = True
            fr, sh = _plan(openm, bge, last, -1, ks, ring, P, o)
            for a, n in legal:
                F1 = _wait(1 << n, 1, hit, sh)
                if F1:
                    t = _escape(F1, 2, last, hit, later, fr, sh)
                    if t:
                        opts.append((a, n, t, F1))

        can_drop = mine < _MAXB and not allb & mb
        dbg = self.debug
        if dbg is not None:
            dbg.clear()
            dbg["alive"] = {a: t for a, n, t, F1 in opts}
            dbg["last"] = last
            dbg["squeezed"] = squeezed
            dbg["contested"] = contested

        if not opts:
            return self._doomed(legal, can_drop, cells, times, bl, openm, mb, ob, m, o, P, ks, ring, ringm,
                                tend, hit, bge, last)

        # ---- the opponent is already inside a blast: is it doomed, or can I doom it by where I stand?
        #      (Walking into its only way out works even if it wants the same cell: then neither moves, and I
        #      simply try again, as long as standing here does not kill me first.)
        if last and danger & ob:
            frO, shO = _plan(openm, bge, last, -1, ks, ring, 1 - P, m)
            if not _fate(ob, 1, last, hit, frO, shO, frO[1] & nmb):  # doomed whatever I do
                kills = opts
            elif self.block and not squeezed:
                kills = []
                stay_ok = opts[0][0] == 0
                for x in (opts if again is None else opts + [again]):
                    if x[2] != 1 or (x[0] and not stay_ok):
                        continue
                    n = x[1]
                    frO, shO = _plan(openm, bge, last, ~(1 << n), ks, ring, 1 - P, n)
                    if not _fate(ob, 1, last, hit, frO, shO, frO[1] & nmb):
                        kills.append(x)
            else:
                kills = []
            if kills:
                if dbg is not None:
                    dbg["why"] = "hold"
                    dbg["kills"] = [x[0] for x in kills]
                tmin = min(x[2] for x in kills)
                pool = [x for x in kills if x[2] == tmin]
                for x in pool:
                    if x[0] == 0:
                        return 0
                return random.choice(pool)[0]

        # ---- a bomb now?
        if can_drop:
            hit2, bge2, last2, tf2 = _schedule(bl, cells + [m], times + [_FUSE], ks, ringm, tend)
            if tf2[-1]:  # it will really go off (not on a ring that closes first)
                later2 = _later(hit2, last2)
                fr2, sh2 = _plan(openm, bge2, last2, -1 if squeezed else nob, ks, ring, P, o)
                F1 = _wait(mb, 1, hit2, sh2)
                bt = _escape(F1, 2, last2, hit2, later2, fr2, sh2) if F1 else 0
                if dbg is not None:
                    dbg["alive"][5] = bt
                if bt:
                    G2 = -1
                    if later2[0] & ob:
                        frO, shO = _plan(openm, bge2, last2, nmb, ks, ring, 1 - P, m)
                        G2 = _fate(ob, 1, last2, hit2, frO, shO)
                        if not G2:
                            if dbg is not None:
                                dbg["why"] = "kill"
                            return 5
                    if self.comfort:
                        F2 = _wait(F1, 2, hit2, sh2)
                        ok = F2 and _escape(F2, 3, last2, hit2, later2, fr2, sh2)
                    else:
                        ok = True
                    if ok and not squeezed:
                        if bl[m] & ob:
                            if self.press >= 1.0 or random.random() < self.press:
                                if dbg is not None:
                                    dbg["why"] = "press"
                                return 5
                        elif self.cut and danger & ob and G2 > 0:
                            frO, shO = _plan(openm, bge, last2, nmb, ks, ring, 1 - P, m)
                            n1 = _fate(ob, 1, last2, hit, frO, shO).bit_count()
                            n2 = G2.bit_count()
                            if n2 <= 2 or 2 * n2 <= n1:
                                if dbg is not None:
                                    dbg["why"] = "cut"
                                return 5

        # ---- movement
        if danger & mb:  # I am inside a blast zone
            pool = None
            mode = self.flee
            if mode == 3:
                pool = []
                for x in opts:
                    if x[2] == 1:
                        pool.append(x)
                    else:
                        F2 = _wait(x[3], 2, hit, sh)
                        if F2 and _escape(F2, 3, last, hit, later, fr, sh):
                            pool.append(x)
            if not pool:
                tmin = min(x[2] for x in opts)
                pool = [x for x in opts if x[2] == tmin]
            if len(pool) == 1:
                return pool[0][0]
            if mode == 0:
                return random.choice(pool)[0]
            return self._toward(pool, ob, o, openm, allb, None, None, None, -1 if mode == 2 else 1)

        pool = [x for x in opts if x[2] == 1]
        if not pool:
            pool = opts
        elif self.eps and random.random() < self.eps:
            return random.choice(pool)[0]
        if self.enter and len(pool) < len(opts):
            for x in opts:
                if x[2] > 1:
                    F2 = _wait(x[3], 2, hit, sh)
                    if F2 and _escape(F2, 3, last, hit, later, fr, sh):
                        pool.append(x)

        # the ring is about to close: get inside
        inward = None
        if ring is not None and left <= 8 and ringm & mb:
            want = mb
            for x in pool:
                want |= 1 << x[1]
            free = openm & ~allb
            din = _dists(ring[1] & free, free, want)
            if left <= din.get(m, 9) + 2:
                inward = din

        # could I bomb from there on the next step, if the opponent stays where it is? (2: it would have no way
        # out, 1: it would be inside the blast and I could get away)
        ready = None
        if self.hunt and ks > 7 and tend > 7 and len(pool) > 1 and not squeezed:
            mine2 = 0
            for i in range(len(cells)):
                if own[i] and tf[i] >= 2:
                    mine2 += 1
            if mine2 < _MAXB and not hit[1] & ob:
                for x in pool:
                    n = x[1]
                    reach = bl[n] & ob
                    if not reach and not danger & ob:
                        continue
                    nbit = 1 << n
                    c3 = [n]
                    t3 = [_FUSE + 1]
                    for i in range(len(cells)):
                        if tf[i] >= 2:
                            if cells[i] == n:
                                c3 = None
                                break
                            c3.append(cells[i])
                            t3.append(tf[i])
                    if c3 is None:
                        continue
                    hit3, bge3, last3, _ = _schedule(bl, c3, t3, 99, 0, 99)
                    if (hit3[1] | hit3[2]) & nbit:
                        continue
                    later3 = _later(hit3, last3)
                    fr3 = [openm & ~bge3[k] & nob for k in range(last3 + 2)]
                    if not _escape(nbit, 3, last3, hit3, later3, fr3, None):
                        continue
                    val = 0
                    if later3[1] & ob:  # the opponent (on its cell at time 1) moves from step 2 on, I stand on n
                        blkn = ~nbit
                        frO = [openm & ~bge3[k] & blkn for k in range(last3 + 2)]
                        if not _fate(ob, 2, last3, hit3, frO, None):
                            val = 2
                    if not val and reach:
                        F3 = nbit & ~hit3[3]
                        if F3 and _escape(F3, 4, last3, hit3, later3, fr3, None):
                            val = 1
                    if val:
                        if ready is None:
                            ready = {}
                        ready[n] = val
        return self._toward(pool, ob, o, openm, allb, ready, inward,
                            ring if (ring is not None and left <= 3) else None, 1)

    # ----------------------------------------------------------------------------------------------------------
    def _toward(self, pool, ob, o, openm, allb, ready, inward, ring, sign):
        """The option of `pool` that brings me closest to the opponent (sign = -1: farthest from it); random
        among equals."""
        want = 0
        for x in pool:
            want |= 1 << x[1]
        free = openm & ~allb
        src = ob
        if ring is not None:  # the ring closes in a moment: plan on the board as it will be
            free &= ring[1]
            if ring[0] & ob:
                src = 1 << _tele(ring[2], o, allb)
        dist = _dists(src, free, want)
        if not dist:
            dist = _dists(src, openm if ring is None else ring[1], want)
        best = None
        acts = []
        for x in pool:
            a = x[0]
            n = x[1]
            key = sign * dist.get(n, 60)
            if ready is not None:
                key -= 100 * ready.get(n, 0)
            if inward is not None:
                key += 1000 * inward.get(n, 9)
            if not dist and a == 0:
                key -= 1  # nowhere to go: rather stand still
            if best is None or key < best:
                best = key
                acts = [a]
            elif key == best:
                acts.append(a)
        return acts[0] if len(acts) == 1 else random.choice(acts)

    def _doomed(self, legal, can_drop, cells, times, bl, openm, mb, ob, m, o, P, ks, ring, ringm, tend, hit, bge,
                last):
        """As far as I can see every action ends in a blast: take the opponent along if a bomb can do that,
        otherwise stay alive as long as possible (the opponent may still change something)."""
        if can_drop:
            hit2, bge2, last2, tf2 = _schedule(bl, cells + [m], times + [_FUSE], ks, ringm, tend)
            if tf2[-1] and _later(hit2, last2)[0] & ob:
                frO, shO = _plan(openm, bge2, last2, ~mb, ks, ring, 1 - P, m)
                if not _fate(ob, 1, last2, hit2, frO, shO):
                    return 5
        fr, sh = _plan(openm, bge, last, -1, ks, ring, P, o)
        best = -1
        acts = [0]
        for a, n in legal:
            F = _wait(1 << n, 1, hit, sh)
            life = 0
            k = 2
            while F:
                life = k - 1
                if k > last:
                    break
                F = _fate(F, k, k, hit, fr, sh)
                k += 1
            if life > best:
                best = life
                acts = [a]
            elif life == best:
                acts.append(a)
        return acts[0] if len(acts) == 1 else random.choice(acts)


def make_policy(arg):
    """`zoo.ta_smart_b:eps=0.1,press=0.5,comfort=0` -> a factory of Policy objects with these settings."""
    kw = {}
    for part in str(arg).split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if not hasattr(Policy, k):
            raise ValueError(f"unknown setting {k!r}")
        kw[k] = float(v) if "." in v else int(v)

    def factory():
        return Policy(**kw)

    return factory
