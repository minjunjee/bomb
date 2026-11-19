"""test_boss: held-out TEST opponent, the "final TA bot" persona (rule based + exact search, no learning).

Idea
----
* The board is a bitboard (cell (r, c) = bit r * 9 + c of a python int), so "where can a player still be standing
  after k steps without being hit" is a handful of shifts and masks per step (exact timing: obs fuse f = explodes at
  the end of the f-th step from now, chains included, bombs block until they explode, the shrink is modelled).
* Every step: look at all joint actions (my action x the opponent's action), play each one with an exact copy of the
  engine rules and score the position that comes out of it:
    - somebody hit -> win / loss / draw,
    - exact survival analysis for both players (who can still outrun every bomb, who dies first),
    - one more ply for bombs: "if X drops a bomb right now, can the other one still get away, and can X?",
    - desperation bombs of a doomed player (chains -> draw),
    - pocket races: precomputed "a bomb on y kills whoever stands on x" tables + who reaches the mouth first,
    - mobility under fire and a small positional term (close in, be inside before the board shrinks).
  The action with the best worst case is played (plus a little credit for the average over the opponent's replies).
"""

import os
import random

import numpy as np

from bombcompetition import RULES

FUSE = RULES.fuse
RAD = RULES.radius
MAXB = RULES.max_bombs
SHRINK = RULES.shrink_every or 0
MAXSTEPS = RULES.max_steps
N = 9
NN = N * N
DI = (0, -N, N, -1, 1, 0)  # change of the cell index per action
INF = 99

WIN, LOSS, DRAW = 1.0, -1.0, -0.3
LAM = 0.15  # weight of the mean over the opponent's replies (the worst case has weight 1)
_DEBUG = bool(os.environ.get("TEST_BOSS_DEBUG"))

_CTX = {}


class Ctx:
    """Everything that only depends on the walls."""

    def __init__(self, key, rows):
        self.key = key
        self.rows = rows
        openm = 0
        for r in range(1, N - 1):
            row = rows[r]
            for c in range(1, N - 1):
                if not row[c]:
                    openm |= 1 << (r * N + c)
        self.open = openm
        self.cells = [i for i in range(NN) if (openm >> i) & 1]
        blast = [0] * NN
        for i in self.cells:
            m = 1 << i
            for d in (-N, N, -1, 1):
                j = i
                for _ in range(RAD):
                    j += d
                    if not (openm >> j) & 1:
                        break
                    m |= 1 << j
            blast[i] = m
        self.blast = blast
        self._dist = {}
        self._nxt = {}
        self._trap = None

    def dist(self, src):
        """Walking distances from src (walls only)."""
        d = self._dist.get(src)
        if d is None:
            d = [INF] * NN
            d[src] = 0
            openm = self.open
            cur, k = [src], 0
            while cur:
                k += 1
                nx = []
                for i in cur:
                    for dd in (-N, N, -1, 1):
                        j = i + dd
                        if d[j] == INF and (openm >> j) & 1:
                            d[j] = k
                            nx.append(j)
                cur = nx
            self._dist[src] = d
        return d

    def nxt(self, k):
        """The board after ring k has turned into walls: (ctx, teleport map, inner cells, steps to the inside),
        or None if that shrink does not happen."""
        if k in self._nxt:
            return self._nxt[k]
        rows = [list(r) for r in self.rows]
        inner = []
        for r in range(N):
            for c in range(N):
                ring = min(r, c, N - 1 - r, N - 1 - c)
                if ring == k:
                    rows[r][c] = True
                elif ring > k and not self.rows[r][c]:
                    inner.append((r, c))
        res = None
        if inner:
            nx = _ctx_from_rows(rows)
            tele = {}
            for i in self.cells:
                r, c = divmod(i, N)
                if min(r, c, N - 1 - r, N - 1 - c) == k:
                    b = min(inner, key=lambda x: (abs(x[0] - r) + abs(x[1] - c), x))
                    tele[i] = b[0] * N + b[1]
            din = [INF] * NN
            g = nx.open
            seen = g
            for i in nx.cells:
                din[i] = 0
            k2 = 0
            while g:
                k2 += 1
                g = (g << 1 | g >> 1 | g << N | g >> N) & self.open & ~seen
                seen |= g
                x = g
                while x:
                    low = x & -x
                    din[low.bit_length() - 1] = k2
                    x ^= low
            res = (nx, tele, inner, din)
        self._nxt[k] = res
        return res

    def trap(self):
        """T[y] = cells x such that a bomb dropped on y (the victim on x moves in the same step, y is blocked) kills
        the victim although the bomber can walk away from it; by[x] = the list of such y. No other bombs."""
        if self._trap is None:
            T = [0] * NN
            by = [[] for _ in range(NN)]
            openm = self.open
            for y in self.cells:
                bl = self.blast[y]
                yb = 1 << y
                free = openm & ~yb
                f = yb
                for _ in range(FUSE - 1):
                    f |= (f << 1 | f >> 1 | f << N | f >> N) & free
                if not f & ~bl:
                    continue
                m = 0
                x = bl & ~yb
                while x:
                    low = x & -x
                    x ^= low
                    g = low
                    for _ in range(FUSE):
                        g |= (g << 1 | g >> 1 | g << N | g >> N) & free
                    if not g & ~bl:
                        m |= low
                        by[low.bit_length() - 1].append(y)
                T[y] = m
            self._trap = (T, by)
        return self._trap


def _ctx_from_rows(rows):
    key = bytes(bytearray(1 if rows[r][c] else 0 for r in range(N) for c in range(N)))
    ctx = _CTX.get(key)
    if ctx is None:
        if len(_CTX) > 4000:
            _CTX.clear()
        ctx = _CTX[key] = Ctx(key, rows)
    return ctx


def _get_ctx(walls):
    w = np.asarray(walls, dtype=bool)
    key = w.tobytes()
    ctx = _CTX.get(key)
    if ctx is None:
        ctx = _ctx_from_rows(w.tolist())
    return ctx


def _flood(g, tl, block):
    """g: the player's cell (bit), tl: a timeline, block: cells that cannot be entered in the first step.
    Returns (time of death or INF, cells the player can stand on after the last explosion)."""
    last, free, safe, _, ks, nopen, tele = tl
    g = (g | ((g << 1 | g >> 1 | g << N | g >> N) & free[1] & ~block)) & safe[1]
    if not g:
        return 1, 0
    if ks < 0:
        for k in range(2, last + 1):
            g = (g | ((g << 1 | g >> 1 | g << N | g >> N) & free[k])) & safe[k]
            if not g:
                return k, 0
        return INF, g
    k = 1
    while True:
        if k == ks:  # the board shrinks after this step: players on the ring are moved inside
            out = g & ~nopen
            if out:
                g &= nopen
                while out:
                    low = out & -out
                    out ^= low
                    g |= 1 << tele[low.bit_length() - 1]
        k += 1
        if k > last:
            return INF, g
        g = (g | ((g << 1 | g >> 1 | g << N | g >> N) & free[k])) & safe[k]
        if not g:
            return k, 0


def _flood_c(g, r, tl):
    """Like _flood (boards without a shrink in sight), but pessimistic about the other player (r): a cell that the
    other one can stand on or walk to in the same step (while staying alive itself) cannot be entered.
    Returns the time of death or INF."""
    last, free, safe = tl[0], tl[1], tl[2]
    for k in range(1, last + 1):
        fk, sk = free[k], safe[k]
        r = (r | ((r << 1 | r >> 1 | r << N | r >> N) & fk)) & sk
        g = (g | ((g << 1 | g >> 1 | g << N | g >> N) & fk & ~r)) & sk
        if not g:
            return k
    return INF


class Policy:
    def __init__(self):
        self.p = 0
        self._tlc = {}

    # ------------------------------------------------------------------------------------------------------------
    def act(self, obs):
        if _DEBUG:
            return int(self._act(obs))
        try:
            a = int(self._act(obs))
            return a if 0 <= a <= 5 else 0
        except Exception:  # noqa: BLE001  (never raise)
            return 0

    # ------------------------------------------------------------------------------------------------------------
    def _timeline(self, ctx, t, bombs):
        """When and where the bombs go off (chains, the shrink and the end of the game included).
        -> (last, free[k], safe[k], danger, ks, new open mask, teleport map) or None when nothing explodes.
        free[k]: cells that can be entered in step k; safe[k]: cells not hit at the end of step k."""
        key = (t, ctx.key, bombs)
        tlc = self._tlc
        if key in tlc:
            return tlc[key]
        n = len(bombs)
        lim = MAXSTEPS - t
        if n == 0 or lim <= 0:
            tlc[key] = None
            return None
        BL = ctx.blast
        cells = [x[0] for x in bombs]
        times = [x[1] for x in bombs]
        if n > 1:
            ch = True
            while ch:
                ch = False
                for i in range(n):
                    ti, bi = times[i], BL[cells[i]]
                    for j in range(n):
                        if times[j] > ti and (bi >> cells[j]) & 1:
                            times[j] = ti
                            ch = True
        last = max(times)
        openm = ctx.open
        sh = None
        ks = -1
        if SHRINK and (t // SHRINK + 1) * SHRINK < MAXSTEPS:
            ku = SHRINK - t % SHRINK
            if ku <= max(x[1] for x in bombs):
                sh = ctx.nxt(t // SHRINK + 1)
                if sh is not None:
                    ks = ku
        if ks < 0:
            hit = [0] * (last + 1)
            cb = [0] * (last + 1)
            for i in range(n):
                ti = times[i]
                hit[ti] |= BL[cells[i]]
                cb[ti] |= 1 << cells[i]
            free = [0] * (last + 1)
            safe = [0] * (last + 1)
            acc = danger = 0
            for k in range(last, 0, -1):
                acc |= cb[k]
                free[k] = openm & ~acc
                safe[k] = ~hit[k]
                danger |= hit[k]
            if last > lim:
                last = lim
            tl = (last, free, safe, danger, -1, 0, None)
        else:
            nx, tele = sh[0], sh[1]
            nopen, NBL = nx.open, nx.blast
            L = max(max(x[1] for x in bombs), ks)
            hit = [0] * (L + 1)
            cb = [0] * (L + 1)
            lc, lt = [], []
            for i in range(n):
                ti, c = times[i], cells[i]
                if ti <= ks:
                    hit[ti] |= BL[c]
                    cb[ti] |= 1 << c
                elif not (nopen >> c) & 1:
                    cb[ks] |= 1 << c  # a bomb on the ring vanishes with it
                else:
                    lc.append(c)
                    lt.append(bombs[i][1])
            m = len(lc)
            if m > 1:
                ch = True
                while ch:
                    ch = False
                    for i in range(m):
                        ti, bi = lt[i], NBL[lc[i]]
                        for j in range(m):
                            if lt[j] > ti and (bi >> lc[j]) & 1:
                                lt[j] = ti
                                ch = True
            for i in range(m):
                hit[lt[i]] |= NBL[lc[i]]
                cb[lt[i]] |= 1 << lc[i]
            free = [0] * (L + 1)
            safe = [0] * (L + 1)
            acc = 0
            for k in range(L, 0, -1):
                acc |= cb[k]
                free[k] = (openm if k <= ks else nopen) & ~acc
                safe[k] = ~hit[k]
            last = min(L, lim)
            tl = (last, free, safe, -1, ks, nopen, tele)
        tlc[key] = tl
        return tl

    # ------------------------------------------------------------------------------------------------------------
    def _sim(self, ctx, t, me, opp, bombs, a, b):
        """One exact step of the engine. bombs: tuple of (cell, fuse, owner) with owner 0 = me, 1 = the opponent.
        -> (hit, me, opp, bombs, ctx); hit: 0 nobody, 1 me, 2 the opponent, 3 both."""
        occ = 0
        nm = no = 0
        for x in bombs:
            occ |= 1 << x[0]
            if x[2]:
                no += 1
            else:
                nm += 1
        if a == 5 and nm < MAXB and not (occ >> me) & 1:
            bombs = bombs + ((me, FUSE, 0),)
            occ |= 1 << me
        if b == 5 and no < MAXB and not (occ >> opp) & 1:
            bombs = bombs + ((opp, FUSE, 1),)
            occ |= 1 << opp
        openm = ctx.open
        tm = me + DI[a]
        if tm != me and (tm == opp or not (openm >> tm) & 1 or (occ >> tm) & 1):
            tm = me
        to = opp + DI[b]
        if to != opp and (to == me or not (openm >> to) & 1 or (occ >> to) & 1):
            to = opp
        if tm != to:
            me, opp = tm, to
        hit = 0
        if bombs:
            boom = False
            nb = []
            for c, f, o in bombs:
                if f <= 1:
                    boom = True
                nb.append((c, f - 1, o))
            if boom:
                BL = ctx.blast
                blast = 0
                ch = True
                while ch:
                    ch = False
                    keep = []
                    for x in nb:
                        if x[1] <= 0 or (blast >> x[0]) & 1:
                            blast |= BL[x[0]]
                            ch = True
                        else:
                            keep.append(x)
                    nb = keep
                hit = ((blast >> me) & 1) | (((blast >> opp) & 1) << 1)
            bombs = tuple(nb)
        t += 1
        if hit == 0 and SHRINK and t % SHRINK == 0 and t < MAXSTEPS:
            sh = ctx.nxt(t // SHRINK)
            if sh is not None:
                nx, inner = sh[0], sh[2]
                nopen = nx.open
                bombs = tuple(x for x in bombs if (nopen >> x[0]) & 1)
                occ = 0
                for x in bombs:
                    occ |= 1 << x[0]
                pos = [me, opp] if self.p == 0 else [opp, me]
                for q in (0, 1):
                    cell = pos[q]
                    if not (nopen >> cell) & 1:
                        r0, c0 = divmod(cell, N)
                        other = pos[1 - q]
                        fr = [x for x in inner if not (occ >> (x[0] * N + x[1])) & 1 and x[0] * N + x[1] != other]
                        b2 = min(fr or inner, key=lambda x: (abs(x[0] - r0) + abs(x[1] - c0), x))
                        pos[q] = b2[0] * N + b2[1]
                me, opp = (pos[0], pos[1]) if self.p == 0 else (pos[1], pos[0])
                ctx = nx
        return hit, me, opp, bombs, ctx

    # ------------------------------------------------------------------------------------------------------------
    def _race(self, ctx, x, q):
        """The victim stands on x, the bomber on q (no bombs considered). -> (margin, steps of the bomber):
        margin >= 1: the bomber reaches a cell y whose bomb kills the victim before the victim can leave that pocket."""
        T, by = ctx.trap()
        ys = by[x]
        if not ys:
            return -INF, 0
        dq = ctx.dist(q)
        openm = ctx.open
        best, bd = -INF, 0
        for y in ys:
            d = dq[y]
            if d > 6:
                continue
            bad = T[y] | (1 << y)
            g = 1 << x
            r = 1 << q
            esc = d + 2
            for j in range(1, d + 1):  # the victim may not enter a cell the bomber can reach first
                r |= (r << 1 | r >> 1 | r << N | r >> N) & openm
                g |= (g << 1 | g >> 1 | g << N | g >> N) & openm & ~r
                if g & ~bad:
                    esc = j
                    break
            if esc - d > best:
                best, bd = esc - d, d
        return best, bd

    # ------------------------------------------------------------------------------------------------------------
    def _eval(self, ctx, t, me, opp, bombs):
        """The value of a position (both players alive, both to move) for me, in [-1, 1]."""
        mb, ob = 1 << me, 1 << opp
        tl = self._timeline(ctx, t, bombs)
        md = od = INF
        mcnt = ocnt = 9
        danger = 0
        cont = 0.0
        if tl is not None:
            danger = tl[3]
            if mb & danger:
                md, g = _flood(mb, tl, ob)
                mcnt = g.bit_count()
                if md == INF and tl[4] < 0 and _flood_c(mb, ob, tl) < INF:
                    cont -= 0.25  # my way out can be blocked by the opponent's body
            if ob & danger:
                od, g = _flood(ob, tl, mb)
                ocnt = g.bit_count()
                if od == INF and tl[4] < 0 and _flood_c(ob, mb, tl) < INF:
                    cont += 0.15
        occ = 0
        nm = no = 0
        for x in bombs:
            occ |= 1 << x[0]
            if x[2]:
                no += 1
            else:
                nm += 1
        can_m = nm < MAXB and not occ & mb
        can_o = no < MAXB and not occ & ob
        if md < INF or od < INF:
            if md < INF and od < INF:
                if md < od:
                    return LOSS + 0.01 * md
                if od < md:
                    return WIN - 0.01 * od
                return DRAW
            if od < INF:  # the opponent is lost; its last bomb may still take me along
                v = WIN - 0.01 * od
                if can_o:
                    tl2 = self._timeline(ctx, t, bombs + ((opp, FUSE, 1),))
                    d1, _ = _flood(mb, tl2, ob)
                    if d1 < INF:
                        d2, _ = _flood(ob, tl2, mb)
                        if d1 < d2:
                            v = LOSS + 0.01 * d1
                        elif d1 == d2:
                            v = DRAW
                return v
            v = LOSS + 0.01 * md  # I am lost; can my last bomb take the opponent along?
            if can_m:
                tl2 = self._timeline(ctx, t, bombs + ((me, FUSE, 0),))
                d1, _ = _flood(ob, tl2, mb)
                if d1 < INF:
                    d2, _ = _flood(mb, tl2, ob)
                    if d1 < d2:
                        v = WIN - 0.01 * d1
                    elif d1 == d2:
                        v = DRAW
            return v
        # nobody is lost yet: what does one more bomb, dropped right now, do?
        BL = ctx.blast
        othr = mthr = False
        if can_o and mb & (BL[opp] | danger):
            tl2 = self._timeline(ctx, t, bombs + ((opp, FUSE, 1),))
            d1, _ = _flood(mb, tl2, ob)
            if d1 < INF:
                d2, _ = _flood(ob, tl2, mb)
                if d2 > d1:
                    othr = True
        if can_m and ob & (BL[me] | danger):
            tl2 = self._timeline(ctx, t, bombs + ((me, FUSE, 0),))
            d1, _ = _flood(ob, tl2, mb)
            if d1 < INF:
                d2, _ = _flood(mb, tl2, ob)
                if d2 > d1:
                    mthr = True
        if othr:
            return -0.45 if mthr else -0.8
        if mthr:
            return 0.8
        v = 0.02 * (min(mcnt, 6) - min(ocnt, 6)) + cont
        return v + self._strategic(ctx, me, opp)

    def _strategic(self, ctx, me, opp):
        """Pocket races (bombs ignored): who can be sealed in a dead end."""
        _, by = ctx.trap()
        v = 0.0
        if by[me]:
            m, d = self._race(ctx, me, opp)
            if m >= 1:
                v -= 0.6
            elif m == 0:
                v -= 0.08
        if by[opp]:
            m, d = self._race(ctx, opp, me)
            if m >= 1:
                v += 0.5 - 0.03 * d
            elif m == 0:
                v += 0.05
        return v

    def _v2(self, ctx, t, me, opp, bombs, hi):
        """One more exact step: the best worst case over my actions (max over mine of the min over the opponent's).
        Stops as soon as the value reaches `hi` (the caller only wants to know whether it is below that)."""
        openm = ctx.open
        occ = 0
        nm = no = 0
        for x in bombs:
            occ |= 1 << x[0]
            if x[2]:
                no += 1
            else:
                nm += 1
        A2 = []
        B2 = [5] if no < MAXB and not (occ >> opp) & 1 else []
        for a in (1, 2, 3, 4):
            j = me + DI[a]
            if (openm >> j) & 1 and not (occ >> j) & 1 and j != opp:
                A2.append(a)
            j = opp + DI[a]
            if (openm >> j) & 1 and not (occ >> j) & 1 and j != me:
                B2.append(a)
        A2.append(0)
        B2.append(0)
        if nm < MAXB and not (occ >> me) & 1:
            A2.append(5)
        sim, ev = self._sim, self._eval
        t1 = t + 1
        best = -9.0
        for a2 in A2:
            worst = 9.0
            for b2 in B2:
                hit, m2, o2, bb, c2 = sim(ctx, t, me, opp, bombs, a2, b2)
                if hit:
                    v = DRAW if hit == 3 else (LOSS if hit == 1 else WIN)
                elif t1 >= MAXSTEPS:
                    v = DRAW
                else:
                    v = ev(c2, t1, m2, o2, bb)
                if v < worst:
                    worst = v
                    if worst <= best:
                        break
            if worst > best:
                best = worst
                if best >= hi:
                    break
        return best

    # ------------------------------------------------------------------------------------------------------------
    def _act(self, obs):
        ctx = _get_ctx(obs["walls"])
        p = self.p = int(obs["player"])
        me = obs["me"][0] * N + obs["me"][1]
        opp = obs["opp"][0] * N + obs["opp"][1]
        t = int(obs["step"])
        bl = []
        occ = 0
        nm = no = 0
        for b in obs["bombs"]:
            c = b["pos"][0] * N + b["pos"][1]
            o = 0 if b["owner"] == p else 1
            bl.append((c, int(b["fuse"]), o))
            occ |= 1 << c
            if o:
                no += 1
            else:
                nm += 1
        bombs = tuple(bl)
        self._tlc = {}
        openm = ctx.open

        A = [0]
        B = [0]
        for a in (1, 2, 3, 4):
            j = me + DI[a]
            if (openm >> j) & 1 and not (occ >> j) & 1 and j != opp:
                A.append(a)
            j = opp + DI[a]
            if (openm >> j) & 1 and not (occ >> j) & 1 and j != me:
                B.append(a)
        if nm < MAXB and not (occ >> me) & 1:
            A.append(5)
        if no < MAXB and not (occ >> opp) & 1:
            B.append(5)

        sh = None
        ksh = 999
        if SHRINK and (t // SHRINK + 1) * SHRINK < MAXSTEPS:
            sh = ctx.nxt(t // SHRINK + 1)
            ksh = SHRINK - t % SHRINK
        dopp = ctx.dist(opp)

        def positional(cell):
            d = dopp[cell]
            s = -0.012 * (d if d > 2 else 2)
            if sh is not None and not (sh[0].open >> cell) & 1:
                slack = ksh - sh[3][cell]
                if slack <= 1:
                    s -= 0.5
                elif slack <= 6:
                    s -= 0.12
                elif ksh <= 40:
                    s -= 0.02
            return s

        best_a, best_s = 0, -1e9
        if not bombs and dopp[me] >= 6:  # nothing tactical can happen yet: walk
            for a in A:
                if a == 5:
                    continue
                cell = me + DI[a]
                s = positional(cell) + self._strategic(ctx, cell, opp) + random.random() * 0.004
                if s > best_s:
                    best_a, best_s = a, s
            return best_a

        sim, ev, v2 = self._sim, self._eval, self._v2
        nB = len(B)
        t1 = t + 1
        rows = []
        for a in A:  # first pass: one exact step + the static value of what comes out
            worst = 9.0
            tot = 0.0
            row = []
            for b in B:
                hit, m2, o2, b2, c2 = sim(ctx, t, me, opp, bombs, a, b)
                if hit:
                    v = DRAW if hit == 3 else (LOSS if hit == 1 else WIN)
                    st = None
                elif t1 >= MAXSTEPS:
                    v = DRAW
                    st = None
                else:
                    v = ev(c2, t1, m2, o2, b2)
                    st = (c2, t1, m2, o2, b2)
                if v < worst:
                    worst = v
                tot += v
                row.append((v, st))
            mean = tot / nB
            bonus = LAM * mean + positional(me + DI[a]) + random.random() * 0.004
            rows.append((worst + bonus, a, bonus, mean, row))
        rows.sort(reverse=True)
        deep = []
        for _, a, bonus, mean, row in rows:  # second pass: one more exact step behind every reply (alpha-beta)
            worst = 9.0
            lo = best_s - bonus
            for v, st in sorted(row, key=lambda x: x[0]):
                if st is not None:
                    v = v2(st[0], st[1], st[2], st[3], st[4], worst)
                if v < worst:
                    worst = v
                    if worst <= lo:
                        break
            s = worst + bonus
            deep.append((worst, mean, a))
            if s > best_s:
                best_a, best_s = a, s
        if best_s < -0.85:  # every action can be punished: do not be predictable
            best_s = -1e9
            for worst, mean, a in deep:
                s = mean + 0.1 * worst + random.random() * 0.08
                if s > best_s:
                    best_a, best_s = a, s
        return best_a
