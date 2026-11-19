"""val_hunter: a bitboard look-ahead hunter (used only to VALIDATE the network, never to train it).

Design (own style, no code shared with the other zoo bots):

* The board is an 81-bit integer (bit = row * 9 + col), so a flood fill over time is a handful of shifts.
* Every step it simulates ALL pairs (my action, opponent action) with the exact engine rules
  (drop -> move with blocking / same-target rule -> tick, chains -> hits -> sudden-death shrink with teleport).
* Every resulting position is judged with an exact "who can still outrun the bombs" analysis:
  a backward pass over the explosion schedule gives, for every cell at once, whether a player standing
  there can survive (bombs block the way until they explode, chains go off early, the shrink teleports).
  On top of it, one more ply for the two bombs that matter most: "what if I drop a bomb next step"
  (is the opponent then lost?) and "what if the opponent drops one next step" (am I lost?).
* The action is chosen by a risk-averse mix: worst case over the opponent's replies + the average over
  its plausible replies; forced kills and forced deaths dominate, then pressure / territory / distance.

Timing convention: "time k" = the end of the k-th step from the position at hand; an obs fuse f explodes at
time f; a bomb dropped in step 1 explodes at time RULES.fuse.
"""

import gc
import random

from bombcompetition import RULES

FUSE = RULES.fuse
RAD = RULES.radius
MAXB = RULES.max_bombs
SHRINK = RULES.shrink_every or 0
MAXSTEPS = RULES.max_steps

DIDX = (0, -9, 9, -1, 1, 0)
RINGOF = tuple(min(i // 9, i % 9, 8 - i // 9, 8 - i % 9) for i in range(81))

WIN = 1000.0
LOSS = -1000.0
DRAW = -350.0

# evaluation weights
W_DIST = 3.0
W_TERR = 0.8
THREAT = 450.0
LAM = 0.6        # weight of the worst case (the rest: average over plausible replies)
NOISE = 1.5
SHRINK_LOOK = 12  # this many steps before a shrink, positions are judged on the board after it

IN_ME = (0.0, -33.0, -18.0, -11.0, -8.0, -8.0)    # I stand inside a coming blast, by number of safe first moves
IN_OP = (0.0, 35.0, 20.0, 13.0, 10.0, 10.0)
MOB_ME = (0.0, -45.0, -16.0, -5.0, 0.0, 0.0)      # my safe first moves if the opponent bombs next step
MOB_OP = (0.0, 40.0, 14.0, 4.0, 0.0, 0.0)

BLOCK_ME = 400.0       # the opponent can shut my way out with its body
BLOCK_OP = 250.0       # I can shut the opponent's way out with my body
BLOCK_ME_SOFT = 120.0
BLOCK_OP_SOFT = 40.0
W_IRRATIONAL = 0.1     # weight of an opponent reply that loses for sure
FULL = (1 << 81) - 1

DEBUG = False
ERRORS = []      # exceptions caught in act() (for testing: must stay empty)


# ----------------------------------------------------------------------------------------------------------------
# geometry (cached per wall layout)
# ----------------------------------------------------------------------------------------------------------------
class _Shrink:
    __slots__ = ("ring", "ring_open", "inner_mask", "geo", "order", "land")


class _Sched:
    """Explosion schedule of a set of bombs + who survives it.
    K: last explosion time; bl[k]: cells exploding at time k; so[k]: bombs in the way during the move of step k;
    B1: cells where a player may stand after the move of step 1 and still survive everything; U: all blasts."""
    __slots__ = ("K", "bl", "so", "B1", "so1", "U", "ks", "sh", "geo", "Bs")


class _Geo:
    __slots__ = ("open", "cells", "blast", "_dist", "_feat", "_shr", "single", "empty")

    def __init__(self, opn):
        self.open = opn
        self.cells = [i for i in range(81) if (opn >> i) & 1]
        blast = [0] * 81
        for i in self.cells:
            mk = 1 << i
            for d in (-9, 9, -1, 1):
                j = i
                for _ in range(RAD):
                    j += d
                    if j < 0 or j > 80 or not (opn >> j) & 1:
                        break
                    mk |= 1 << j
            blast[i] = mk
        self.blast = blast
        self._dist = {}
        self._feat = {}
        self._shr = {}
        self.single = {}
        e = _Sched()
        e.K = 0
        e.bl = (0,)
        e.so = (0, 0)
        e.B1 = opn
        e.so1 = 0
        e.U = 0
        e.ks = 0
        e.sh = None
        e.geo = self
        e.Bs = None
        self.empty = e

    def dists(self, s):
        d = self._dist.get(s)
        if d is None:
            opn = self.open
            d = [99] * 81
            d[s] = 0
            q = [s]
            for x in q:
                dx = d[x] + 1
                for y in (x - 9, x + 9, x - 1, x + 1):
                    if 0 <= y < 81 and (opn >> y) & 1 and d[y] == 99:
                        d[y] = dx
                        q.append(y)
            self._dist[s] = d
        return d

    def feat(self, a, b):
        """(path distance a-b, territory: cells nearer to a minus cells nearer to b)."""
        key = a * 81 + b
        f = self._feat.get(key)
        if f is None:
            da = self.dists(a)
            db = self.dists(b)
            t = 0
            for c in self.cells:
                x = da[c]
                y = db[c]
                if x < y:
                    t += 1
                elif y < x:
                    t -= 1
            f = (da[b], t)
            self._feat[key] = f
        return f

    def shrink(self, k):
        if k in self._shr:
            return self._shr[k]
        ring = 0
        inner = []
        for i in range(81):
            rg = RINGOF[i]
            if rg == k:
                ring |= 1 << i
            elif rg > k and (self.open >> i) & 1:
                inner.append(i)
        s = None
        if inner:
            s = _Shrink()
            s.ring = ring
            s.ring_open = [i for i in self.cells if (ring >> i) & 1]
            im = 0
            for i in inner:
                im |= 1 << i
            s.inner_mask = im
            s.geo = _geo_from_mask(im)
            s.order = {}
            land = list(range(81))
            for c in s.ring_open:
                r0, c0 = divmod(c, 9)
                lst = sorted(inner, key=lambda x: (abs(x // 9 - r0) + abs(x % 9 - c0), x))
                s.order[c] = lst
                land[c] = lst[0]
            s.land = land
        self._shr[k] = s
        return s


_GEO_BY_MASK = {}
_GEO_BY_KEY = {}


def _geo_from_mask(opn):
    g = _GEO_BY_MASK.get(opn)
    if g is None:
        if len(_GEO_BY_MASK) > 4000:
            _GEO_BY_MASK.clear()
            _GEO_BY_KEY.clear()
        g = _GEO_BY_MASK[opn] = _Geo(opn)
    return g


def _geo_from_walls(walls):
    try:
        key = walls.tobytes()
    except AttributeError:
        key = b""
    if len(key) != 81:
        key = bytes(1 if walls[r][c] else 0 for r in range(9) for c in range(9))
    g = _GEO_BY_KEY.get(key)
    if g is None:
        opn = 0
        for i in range(81):
            if not key[i]:
                opn |= 1 << i
        g = _GEO_BY_KEY[key] = _geo_from_mask(opn)
    return g


# ----------------------------------------------------------------------------------------------------------------
# explosion schedule + survival analysis
# ----------------------------------------------------------------------------------------------------------------
def _dest(sh, c, bm):
    """Where a player on ring cell c lands at the shrink (bombs avoided; the other player is ignored here)."""
    lst = sh.order[c]
    for x in lst:
        if not (bm >> x) & 1:
            return x
    return lst[0]


def _make_sched(geo, bombs, ks, sh):
    """bombs: {cell: time it explodes by its own fuse (>= 1)}. ks: the shrink `sh` happens at the end of step ks
    (0 = no shrink in sight)."""
    n = len(bombs)
    if n == 0:
        return geo.empty
    blast = geo.blast
    ring = sh.ring if ks else 0
    t = bombs
    if n > 1:
        t = dict(bombs)
        changed = True
        while changed:
            changed = False
            for c in t:
                tc = t[c]
                if ks and tc > ks and (ring >> c) & 1:
                    continue  # vanishes at the shrink before it can explode
                bc = blast[c]
                for d in t:
                    if t[d] > tc and (bc >> d) & 1:
                        t[d] = tc
                        changed = True
    K = 0
    for c, tc in t.items():
        if ks and tc > ks and (ring >> c) & 1:
            continue
        if tc > K:
            K = tc
    s = _Sched()
    s.geo = geo
    s.K = K
    if ks > K:
        ks = 0  # the shrink comes after the last explosion: irrelevant for survival
    s.ks = ks
    s.sh = sh if ks else None
    bl = [0] * (K + 2)
    so = [0] * (max(K, max(t.values())) + 3)
    U = 0
    for c, tc in t.items():
        bit = 1 << c
        if ks and tc > ks:
            if (ring >> c) & 1:
                for k in range(1, ks + 1):
                    so[k] |= bit
                continue
            b = blast[c] & ~ring
        else:
            b = blast[c]
        if tc <= K:
            bl[tc] |= b
            U |= b
        for k in range(1, tc + 1):
            so[k] |= bit
    s.bl = bl
    s.so = so
    s.so1 = so[1]
    s.U = U
    s.Bs = Bs = [0] * (K + 1)
    if K == 0:
        s.B1 = geo.open
        return s
    opn_pre = geo.open
    if ks:
        opn_post = sh.inner_mask
        A = opn_post
    else:
        opn_post = opn_pre
        A = opn_pre
    Bk = A
    for k in range(K, 0, -1):
        if k == ks:
            Bk = A
            bm = so[k + 1]
            for c in sh.ring_open:
                if (A >> _dest(sh, c, bm)) & 1:
                    Bk |= 1 << c
            Bk &= ~bl[k]
        else:
            Bk = A & ~bl[k]
        Bs[k] = Bk
        X = Bk & ~so[k]
        A = (Bk | (X << 1) | (X >> 1) | (X << 9) | (X >> 9)) & (opn_post if (ks and k > ks) else opn_pre)
    s.B1 = Bk
    return s


def _flood(s, start, block1):
    """Forward flood of one player from `start`: (cells it can stand on alive at time K, time of death or 99).
    block1: cells it cannot enter in step 1 (the other player's cell)."""
    F = 1 << start
    K = s.K
    if K == 0:
        return F, 99
    bl = s.bl
    so = s.so
    ks = s.ks
    opn = s.geo.open
    for k in range(1, K + 1):
        mv = ((F << 1) | (F >> 1) | (F << 9) | (F >> 9)) & opn & ~so[k]
        if k == 1:
            mv &= ~block1
        F = (F | mv) & ~bl[k]
        if not F:
            return 0, k
        if k == ks:
            sh = s.sh
            rf = F & sh.ring
            if rf:
                F &= ~sh.ring
                bm = so[k + 1]
                for c in sh.ring_open:
                    if (rf >> c) & 1:
                        F |= 1 << _dest(sh, c, bm)
            opn = sh.inner_mask
    return F, 99


def _blockable(s, x0, y0):
    """Can the player on y0 (the blocker) doom the player on x0 (the victim) with its body: by walking to one cell
    of the victim's way out and holding it (a cell both want is entered by neither)?
    0 = no, 1 = only at the blocker's own peril, 2 = yes and the blocker stays safe."""
    if s.ks:
        return 0
    K = s.K
    so = s.so
    Bs = s.Bs
    opn = s.geo.open
    xb = 1 << x0
    yb = 1 << y0
    FB = yb
    FV = xb
    FBl = [yb]
    UV = 0
    UB = yb
    for k in range(1, K + 1):
        nso = opn & ~so[k]
        m1 = ((FB << 1) | (FB >> 1) | (FB << 9) | (FB >> 9)) & nso
        m2 = ((FV << 1) | (FV >> 1) | (FV << 9) | (FV >> 9)) & nso
        if k == 1:
            m1 &= ~xb
            m2 &= ~yb
        FB = (FB | m1) & Bs[k]
        FV = (FV | m2) & Bs[k]
        FBl.append(FB)
        UV |= FV
        UB |= FB
    cand = UV & UB & ~xb
    best = 0
    while cand:
        X = cand & -cand
        cand ^= X
        a = 0
        while not FBl[a] & X:
            a += 1
        k0 = a if a else 1
        e = k0 - 1
        while e < K and Bs[e + 1] & X:
            e += 1
        if e < k0:
            continue  # the blocker cannot stand there at all
        e += 1
        F = xb
        for k in range(1, K + 1):
            mv = ((F << 1) | (F >> 1) | (F << 9) | (F >> 9)) & opn & ~so[k]
            if k == 1:
                mv &= ~yb
            if k0 <= k <= e:
                mv &= ~X
            F = (F | mv) & Bs[k]
            if not F:
                break
        if not F:
            if a == 0:
                return 2
            if ((X << 1) | (X >> 1) | (X << 9) | (X >> 9)) & FBl[a - 1] & Bs[a]:
                return 2  # bounced back by a contest, the blocker is still fine
            best = 1
    return best


class _Combo:
    """The board after this step for one pair of drop decisions (me drops?, opponent drops?)."""
    __slots__ = ("geo", "blast1", "rem", "rmask", "nm", "no", "sched", "t1", "t2", "ks", "sh", "shrunk",
                 "vals", "land", "egeo")


# ----------------------------------------------------------------------------------------------------------------
class Policy:
    def __init__(self):
        self.info = None
        # A full garbage collection that is already due (typical right after a fork of a process that has torch
        # loaded) would otherwise start inside some act() and cost tens of ms there: pay for it here.
        try:
            if gc.isenabled() and gc.get_count()[2] >= gc.get_threshold()[2] > 0:
                gc.collect()
        except Exception:
            pass

    def act(self, obs):
        try:
            return int(self._act(obs))
        except Exception as e:  # never fault: stay put
            if DEBUG:
                raise
            if len(ERRORS) < 20:
                ERRORS.append(repr(e))
            return 0

    # ------------------------------------------------------------------------------------------------------------
    def _combo(self, geo, bombs, mine, m, o, dm, do, ks, sh):
        bs = bombs
        if dm or do:
            bs = dict(bombs)
            if dm:
                bs[m] = FUSE
            if do:
                bs[o] = FUSE
        C = _Combo()
        blast1 = 0
        gone = None
        for c, f in bs.items():
            if f <= 1:
                if gone is None:
                    gone = set()
                    stack = []
                stack.append(c)
        if gone is not None:
            blast = geo.blast
            while stack:
                c = stack.pop()
                if c in gone:
                    continue
                gone.add(c)
                bc = blast[c]
                blast1 |= bc
                for d in bs:
                    if d not in gone and (bc >> d) & 1:
                        stack.append(d)
            rem = {c: f - 1 for c, f in bs.items() if c not in gone}
        else:
            rem = {c: f - 1 for c, f in bs.items()}
        C.blast1 = blast1
        if ks == 1:  # sudden death at the end of this step
            ring = sh.ring
            rem = {c: f for c, f in rem.items() if not (ring >> c) & 1}
            C.geo = g2 = sh.geo
            C.ks = 0
            C.sh = None
            C.shrunk = True
            C.egeo = g2
            C.land = None
        else:
            C.geo = g2 = geo
            C.shrunk = False
            if ks:
                C.ks = ks - 1
                C.sh = sh
            else:
                C.ks = 0
                C.sh = None
            if ks and ks - 1 <= SHRINK_LOOK:
                C.egeo = sh.geo
                C.land = sh.land
            else:
                C.egeo = g2
                C.land = None
        C.rem = rem
        rmask = 0
        nm = no = 0
        for c in rem:
            rmask |= 1 << c
            if c in mine or (dm and c == m):
                nm += 1
            else:
                no += 1
        C.rmask = rmask
        C.nm = nm
        C.no = no
        C.sched = _make_sched(g2, rem, C.ks, C.sh)
        C.t1 = {}
        C.t2 = {}
        C.vals = {}
        return C

    def _threat_sched(self, C, cache, x):
        s = cache.get(x)
        if s is None:
            if not C.rem and (C.ks == 0 or C.ks > FUSE):
                single = C.geo.single
                s = single.get(x)
                if s is None:
                    s = single[x] = _make_sched(C.geo, {x: FUSE}, 0, None)
            else:
                nb = dict(C.rem)
                nb[x] = FUSE
                s = _make_sched(C.geo, nb, C.ks, C.sh)
            cache[x] = s
        return s

    def _leaf(self, C, m2, o2):
        """(value for me, kind) of the position after one simulated step: I stand on m2, the opponent on o2.
        kind: 0 = open, 1 = the opponent is surely lost, 2 = we both are (a draw)."""
        geo = C.geo
        sc = C.sched
        mb = 1 << m2
        ob = 1 << o2
        v = 0.0
        nbm = (mb << 1) | (mb >> 1) | (mb << 9) | (mb >> 9)
        nbo = (ob << 1) | (ob >> 1) | (ob << 9) | (ob >> 9)
        if sc.K:
            so1 = sc.so1
            B1 = sc.B1
            M1 = (mb | (nbm & ~so1)) & B1 & ~ob   # cells I may step on now and still outrun everything
            O1 = (ob | (nbo & ~so1)) & B1 & ~mb
            if not M1 or not O1:
                tm = to = 99
                if not M1:
                    tm = _flood(sc, m2, ob)[1]
                if not O1:
                    to = _flood(sc, o2, mb)[1]
                if tm < to:
                    return (-900.0 + 10.0 * tm, 0)
                if to < tm:
                    # it is lost, but a lost opponent may still drop a bomb next step to take me along
                    if C.no < MAXB and not (C.rmask >> o2) & 1:
                        s2 = self._threat_sched(C, C.t2, o2)
                        if not (mb | (nbm & ~s2.so1)) & s2.B1 & ~ob:
                            tm2 = _flood(s2, m2, ob)[1]
                            to2 = _flood(s2, o2, mb)[1]
                            if tm2 < to2:
                                return (-700.0, 0)
                            if tm2 == to2:
                                return (DRAW - 20.0, 2)
                    return (900.0 - 10.0 * to, 1)
                if tm < 99:
                    return (DRAW + 30.0, 2)
            cm = M1.bit_count()
            co = O1.bit_count()
            if cm == 1 and M1 == O1:
                # one way out for each of us and it is the same cell: neither gets in, who burns first loses
                tm = _flood(sc, m2, FULL)[1]
                to = _flood(sc, o2, FULL)[1]
                if tm < to:
                    return (-850.0, 0)
                if to < tm:
                    return (850.0, 1)
                return (DRAW + 30.0, 2)
            U = sc.U
            m_in = (U >> m2) & 1
            o_in = (U >> o2) & 1
            if m_in:
                v += IN_ME[cm]
            if o_in:
                v += IN_OP[co]
            # body blocks: a single cell of the way out, taken and held by the other player
            bm = _blockable(sc, m2, o2) if m_in else 0
            bo = _blockable(sc, o2, m2) if o_in else 0
            if bm == 2:
                v -= BLOCK_ME if bo != 2 else 200.0
            elif bo == 2:
                v += BLOCK_OP
            if bm == 1:
                v -= BLOCK_ME_SOFT
            if bo == 1:
                v += BLOCK_OP_SOFT
        else:
            m_in = o_in = 0
            cm = co = 5
        # one more ply for the two bombs that matter: mine next step, the opponent's next step
        thr1 = thr2 = False
        blast = geo.blast
        if C.nm < MAXB and not (C.rmask >> m2) & 1 and (o_in or (blast[m2] >> o2) & 1):
            s1 = self._threat_sched(C, C.t1, m2)
            if (s1.B1 >> m2) & 1:  # I can get away from my own bomb
                O1b = (ob | (nbo & ~s1.so1)) & s1.B1 & ~mb
                co = O1b.bit_count()
                if not co:
                    thr1 = True
                else:
                    v += MOB_OP[co]
        if C.no < MAXB and not (C.rmask >> o2) & 1 and (m_in or (blast[o2] >> m2) & 1):
            s2 = self._threat_sched(C, C.t2, o2)
            M1b = (mb | (nbm & ~s2.so1)) & s2.B1 & ~ob
            cm = M1b.bit_count()
            if (s2.B1 >> o2) & 1:
                if not cm:
                    thr2 = True
                else:
                    v += MOB_ME[cm]
            elif not cm:
                v -= 60.0  # it would die with me: only a reckless opponent does that
        if thr1:
            v += THREAT if not thr2 else -60.0
        elif thr2:
            v -= THREAT
        land = C.land
        if land is not None:
            d, terr = C.egeo.feat(land[m2], land[o2])
        else:
            d, terr = C.egeo.feat(m2, o2)
        if d > 20:
            d = 20
        return (v - W_DIST * d + W_TERR * terr, 0)

    # ------------------------------------------------------------------------------------------------------------
    def _act(self, obs):
        p = obs["player"]
        geo = _geo_from_walls(obs["walls"])
        me = obs["me"]
        opp = obs["opp"]
        m = me[0] * 9 + me[1]
        o = opp[0] * 9 + opp[1]
        step = obs["step"]
        bombs = {}
        mine = set()
        nmine = nopp = 0
        bmask = 0
        for b in obs["bombs"]:
            pos = b["pos"]
            i = pos[0] * 9 + pos[1]
            bombs[i] = int(b["fuse"])
            bmask |= 1 << i
            if b["owner"] == p:
                mine.add(i)
                nmine += 1
            else:
                nopp += 1

        ks = 0
        sh = None
        if SHRINK:
            kk = step // SHRINK + 1
            if kk * SHRINK < MAXSTEPS:
                sh = geo.shrink(kk)
                if sh is not None:
                    ks = kk * SHRINK - step
        last_step = step + 1 >= MAXSTEPS

        free = geo.open & ~bmask
        my_opts = [(0, m, 0)]
        op_opts = [(0, o, 0)]
        for a in (1, 2, 3, 4):
            t = m + DIDX[a]
            if (free >> t) & 1 and t != o:
                my_opts.append((a, t, 0))
            t = o + DIDX[a]
            if (free >> t) & 1 and t != m:
                op_opts.append((a, t, 0))
        if nmine < MAXB and not (bmask >> m) & 1:
            my_opts.append((5, m, 1))
        op_bomb = -1
        if nopp < MAXB and not (bmask >> o) & 1:
            op_bomb = len(op_opts)
            op_opts.append((5, o, 1))

        combos = [None, None, None, None]
        V = []
        nb = len(op_opts)
        bad = [True] * nb     # reply j: the opponent dies (or we both do) whatever I do
        comfy = [True] * nb   # reply j: the opponent is in no mortal danger whatever I do
        for (a, tm, dm) in my_opts:
            row = []
            j = 0
            for (b, to, do) in op_opts:
                ci = dm * 2 + do
                C = combos[ci]
                if C is None:
                    C = combos[ci] = self._combo(geo, bombs, mine, m, o, dm, do, ks, sh)
                if tm == to:
                    pm = m
                    po = o
                else:
                    pm = tm
                    po = to
                bl1 = C.blast1
                if bl1 and ((bl1 >> pm) & 1 or (bl1 >> po) & 1):
                    if (bl1 >> po) & 1:
                        v = DRAW if (bl1 >> pm) & 1 else WIN
                        comfy[j] = False
                    else:
                        v = LOSS
                        bad[j] = False
                elif last_step:
                    v = DRAW
                    bad[j] = False
                else:
                    if C.shrunk:
                        pm, po = _teleport(sh, C.rmask, pm, po, p)
                    key = pm * 81 + po
                    r = C.vals.get(key)
                    if r is None:
                        r = C.vals[key] = self._leaf(C, pm, po)
                    v = r[0]
                    if r[1]:
                        comfy[j] = False
                    else:
                        bad[j] = False
                        if v > 300.0:
                            comfy[j] = False
                row.append(v)
                j += 1
            V.append(row)

        # Which replies to expect. An opponent that has a comfortable reply will not blow itself up; an opponent
        # in trouble everywhere may well do it (to take me along), so then every reply counts in the worst case.
        if True in comfy:
            w = [W_IRRATIONAL if x else 1.0 for x in bad]
        else:
            w = [1.0] * nb
            bad = [False] * nb
        if op_bomb >= 0 and not bad[op_bomb] and (geo.blast[o] >> m) & 1:
            w[op_bomb] *= 2.0
        tot = sum(w)

        best = None
        best_s = -1e18
        scores = []
        for i, row in enumerate(V):
            mn = 1e18
            ex = 0.0
            for j in range(nb):
                x = row[j]
                ex += w[j] * x
                if x < mn and not bad[j]:
                    mn = x
            s = LAM * mn + (1.0 - LAM) * ex / tot + random.random() * NOISE
            scores.append(s)
            if s > best_s:
                best_s = s
                best = my_opts[i][0]
        if DEBUG:
            self.info = (my_opts, op_opts, V, scores)
        return best


def _teleport(sh, rmask, pm, po, p):
    """The engine's shrink teleport (player 0 first), for me = player p on pm and the opponent on po."""
    pos = [0, 0]
    pos[p] = pm
    pos[1 - p] = po
    ring = sh.ring
    for q in (0, 1):
        c = pos[q]
        if (ring >> c) & 1:
            other = pos[1 - q]
            lst = sh.order[c]
            dest = lst[0]
            for x in lst:
                if not (rmask >> x) & 1 and x != other:
                    dest = x
                    break
            pos[q] = dest
    return pos[p], pos[1 - p]
