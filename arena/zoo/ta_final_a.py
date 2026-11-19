"""ta_final_a: a "final TA bot" boss. Exact time-expanded reasoning on bitboards, one to two plies of look-ahead.

Board cells are bits of a python int (bit = row * 9 + col), so "every cell a player can be on" is one integer and a
whole time layer of the (cell, time) graph is a handful of shifts.

Timing convention: "time k" = the end of the k-th step from now (the step being decided is k = 1). A bomb whose obs
fuse is f explodes at time f, a bomb dropped in step j explodes at time j + 4, chains pull explosions earlier.

Every decision:
1. For each pair (my action, opponent action) the first step is resolved with the real rules (drop, blocked moves,
   two players into one cell). For each resulting bomb set the exact explosion schedule is computed (chains, the
   sudden-death shrink: ring bombs vanish, later blasts stop at the new walls, players on the ring are teleported).
2. A backward pass over the time layers gives, for every cell, "standing here at time k, can a player still avoid
   every explosion?" (bombs block the way until they explode). This answers both "am I safe" and "can the opponent
   get away" for all cells at once.
3. My safety is checked on three levels: against every opponent action of this step (also a bomb dropped right now),
   against a bomb the opponent drops next step from wherever it moves, and against the opponent's body blocking my
   escape route (I only count on cells the opponent cannot reach before me).
4. Attack: a bomb (or simply standing still as a blocker) is a forced kill when no opponent action leaves it an
   escape path; one ply deeper, a move is rated by how few opponent replies survive my bomb of the next step.
5. Otherwise: walk toward the opponent, get off the closing ring before a shrink, a little noise against loops.
"""

import random

import numpy as np

from bombcompetition import RULES

N = 9
NN = N * N
FUSE = RULES.fuse
RAD = RULES.radius
MAXB = RULES.max_bombs
EVERY = RULES.shrink_every or 0
MAXSTEPS = RULES.max_steps
STEP = (0, -N, N, -1, 1, 0)  # index change per action

_RING = [min(i // N, i % N, N - 1 - i // N, N - 1 - i % N) for i in range(NN)]
_RINGM = [sum(1 << i for i in range(NN) if _RING[i] == k) for k in range(N)]
_INNERM = [sum(1 << i for i in range(NN) if _RING[i] > k) for k in range(N)]


class _Map:
    """One wall layout: the free cells and the blast of a bomb on every cell."""

    __slots__ = ("free", "blast", "after", "nfree")

    def __init__(self, free):
        self.free = free
        self.nfree = free.bit_count()
        self.after = {}  # ring -> (the _Map after that ring has closed, inner cells, teleports on an empty board)
        blast = [0] * NN
        for i in range(NN):
            if free >> i & 1:
                m = 1 << i
                for d in (-N, N, -1, 1):
                    j = i
                    for _ in range(RAD):
                        j += d
                        if j < 0 or j >= NN or not free >> j & 1:
                            break
                        m |= 1 << j
                blast[i] = m
        self.blast = blast


_MAPS = {}  # free mask -> _Map
_WALLS = {}  # walls bytes -> _Map
_NONE = (0, None, 0, (), {})


def _map_of(walls):
    wb = walls.tobytes()
    mp = _WALLS.get(wb)
    if mp is None:
        free = 0
        for i, w in enumerate(np.asarray(walls, dtype=bool).ravel().tolist()):
            if not w:
                free |= 1 << i
        mp = _MAPS.get(free)
        if mp is None:
            mp = _MAPS[free] = _Map(free)
        if len(_WALLS) > 5000:
            _WALLS.clear()
        _WALLS[wb] = mp
    return mp


def _shrink_info(mp, step):
    """(k, map after, ring mask, inner cells, teleports): the next shrink follows the explosions of step k from now
    (k = 0 and map None: there is none)."""
    if not EVERY:
        return _NONE
    left = EVERY - step % EVERY
    t = step + left
    if t >= MAXSTEPS:
        return _NONE
    ring = t // EVERY
    if ring >= N:
        return _NONE
    info = mp.after.get(ring)
    if info is None:
        inner = tuple(i for i in range(NN) if mp.free >> i & 1 and _RING[i] > ring)
        if not inner:
            info = (None, 0, (), {})  # nothing inside: no shrink
        else:
            ringm = _RINGM[ring]
            f2 = mp.free & ~ringm
            nxt = _MAPS.get(f2)
            if nxt is None:
                nxt = _MAPS[f2] = _Map(f2)
            info = (nxt, ringm, inner, _teleports(mp, ringm, inner, ()))
        mp.after[ring] = info
    if info[0] is None:
        return _NONE
    return (left,) + info


def _teleports(mp, ringm, inner, rem):
    """{open ring cell: the inner cell a player standing there is moved to} (the other player is ignored)."""
    cand = [i for i in inner if i not in rem] or inner
    tele = {}
    x = mp.free & ringm
    while x:
        low = x & -x
        x ^= low
        i0 = low.bit_length() - 1
        r0, c0 = divmod(i0, N)
        tele[i0] = min(cand, key=lambda i: (abs(i // N - r0) + abs(i % N - c0), i))
    return tele


def _teleport_pair(inner, ringm, first, second, bm):
    """The engine's shrink for both players: `first` = player 0's cell, `second` = player 1's; bm = bombs left."""
    out = [first, second]
    for p in (0, 1):
        x = out[p]
        if ringm >> x & 1:
            other = out[1 - p]
            cand = [i for i in inner if not bm >> i & 1 and i != other] or inner
            r0, c0 = divmod(x, N)
            out[p] = min(cand, key=lambda i: (abs(i // N - r0) + abs(i % N - c0), i))
    return out


def _schedule(mp, bombs, shr, late):
    """Exact explosion schedule. bombs: {cell: time}; late: (cell, time) = a bomb that appears in step 2.
    Returns (hit, pres, fre, K, tele, tshr, ringm): for k = 1..K, hit[k] = cells exploding at time k, pres[k] = bombs
    in the way during step k, fre[k] = open cells during step k; tele = teleports of the shrink after step tshr."""
    tshr, mp2, ringm, inner, _ = shr
    rem = dict(bombs)
    K = 1
    for f in rem.values():
        if f > K:
            K = f
    if late is not None and late[1] > K:
        K = late[1]
    free = mp.free
    blast = mp.blast
    hit = [0] * (K + 2)
    pres = [0] * (K + 2)
    fre = [free] * (K + 2)
    tele = None
    for k in range(1, K + 1):
        if k == 2 and late is not None:
            rem[late[0]] = late[1]
        fre[k] = free
        if rem:
            pm = 0
            q = []
            for i, f in rem.items():
                pm |= 1 << i
                if f <= k:
                    q.append(i)
            pres[k] = pm
            if q:
                h = 0
                while q:
                    i = q.pop()
                    if i in rem:
                        del rem[i]
                        b = blast[i]
                        h |= b
                        for j in rem:
                            if b >> j & 1:
                                q.append(j)
                hit[k] = h
        if k == tshr and mp2 is not None:
            for i in [i for i in rem if ringm >> i & 1]:
                del rem[i]
            if k < K:
                tele = _teleports(mp, ringm, inner, rem)
            free = mp2.free
            blast = mp2.blast
    fre[K + 1] = free
    return hit, pres, fre, K, tele, tshr, ringm


def _back(sch, block, perm):
    """S[k] = cells where a player standing there at time k (alive) can still avoid every explosion.
    block[k]: cells that cannot be entered during step k; perm: cells that can never be entered.
    S[0] = the same for the start of step 2, i.e. after the teleport when the shrink follows step 1 (else = S[1])."""
    hit, pres, fre, K, tele, tshr, ringm = sch
    S = [0] * (K + 2)
    cur = fre[K] & ~hit[K]
    S[K] = S[K + 1] = cur
    post = None
    if K == 1 and tshr == 1:
        post = fre[2]
    k = K
    while k > 1:
        e = cur & ~(pres[k] | perm)
        if block is not None:
            e &= ~block[k]
        cur |= (e << N) | (e >> N) | (e << 1) | (e >> 1)
        k -= 1
        if k == tshr and tele is not None:
            if k == 1:
                post = cur & fre[2]
            g = cur & ~ringm
            for x, d in tele.items():
                if cur >> d & 1:
                    g |= 1 << x
            cur = g
        cur &= fre[k] & ~hit[k]
        S[k] = cur
    S[0] = S[1] if post is None else post
    return S


def _reach(sch, y, lim, y2):
    """B[k] = cells a player may stand on or step into during step k, while staying inside lim[k] (= cells from
    which it can still save itself). It is on y at time 1 (on y2 after the teleport, if the shrink follows step 1)."""
    hit, pres, fre, K, tele, tshr, ringm = sch
    B = [0] * (K + 2)
    o = (1 << y) & lim[1]
    k = 2
    while k <= K and o:
        if tele is not None and k - 1 == tshr:
            if k == 2:
                o = 1 << y2
            else:
                g = o & ~ringm
                for x, d in tele.items():
                    if o >> x & 1:
                        g |= 1 << d
                o = g
        mv = ((o << N) | (o >> N) | (o << 1) | (o >> 1)) & fre[k] & ~pres[k]
        B[k] = o | mv
        o = (o | mv) & lim[k]
        k += 1
    return B


def _alive_time(sch, c):
    """The last time a player standing on c at time 1 can still be alive (K + 1 = survives everything)."""
    hit, pres, fre, K, tele, tshr, ringm = sch
    f = (1 << c) & ~hit[1]
    if not f:
        return 0
    k = 2
    while k <= K:
        if tele is not None and k - 1 == tshr:
            g = f & ~ringm
            for x, d in tele.items():
                if f >> x & 1:
                    g |= 1 << d
            f = g
        f = (f | (((f << N) | (f >> N) | (f << 1) | (f >> 1)) & ~pres[k])) & fre[k] & ~hit[k]
        if not f:
            return k - 1
        k += 1
    return K + 1


def _dists(free, src, cells):
    """{cell: walking distance from the cells of the mask src} for the given cells (99 = unreachable)."""
    out = {}
    want = set(cells)
    seen = f = src & free
    d = 0
    while f and want:
        for c in tuple(want):
            if f >> c & 1:
                out[c] = d
                want.discard(c)
        f = ((f << N) | (f >> N) | (f << 1) | (f >> 1)) & free & ~seen
        seen |= f
        d += 1
    for c in want:
        out[c] = 99
    return out


def _voronoi(free, c, y):
    """Cells a player on c reaches strictly before a player on y (both walk at the same speed)."""
    if c == y:
        return 0
    mine = fm = 1 << c
    theirs = fo = 1 << y
    while fm:
        taken = mine | theirs
        no = ((fo << N) | (fo >> N) | (fo << 1) | (fo >> 1)) & free & ~taken
        nm = ((fm << N) | (fm >> N) | (fm << 1) | (fm >> 1)) & free & ~taken & ~no
        mine |= nm
        theirs |= no
        fm = nm
        fo = no
    return mine


class Policy:
    def __init__(self):
        self.trace = None  # a list here collects one row per candidate action (debugging)

    def act(self, obs):
        try:
            return int(self._act(obs))
        except Exception:  # noqa: BLE001  (never raise: stay put)
            return 0

    def _act(self, obs):
        mp = _map_of(obs["walls"])
        free = mp.free
        me = obs["me"][0] * N + obs["me"][1]
        opp = obs["opp"][0] * N + obs["opp"][1]
        p = obs["player"]
        step = obs["step"]
        bombs = {}
        mine = theirs = 0
        bmask = 0
        for b in obs["bombs"]:
            i = b["pos"][0] * N + b["pos"][1]
            bombs[i] = int(b["fuse"])
            bmask |= 1 << i
            if b["owner"] == p:
                mine += 1
            else:
                theirs += 1
        shr = _shrink_info(mp, step)
        tshr, mp2, ringm, inner, tele0 = shr
        now = tshr == 1 and mp2 is not None  # the shrink follows this very step
        ix = 0 if now else 1  # which safe set describes "after this step" (see _back)

        # what each player can do in this step: (action, target cell, drops a bomb)
        acts = [(0, me, False)]
        oacts = [(0, opp, False)]
        for a in (1, 2, 3, 4):
            t = me + STEP[a]
            if free >> t & 1 and not bmask >> t & 1 and t != opp:
                acts.append((a, t, False))
            t = opp + STEP[a]
            if free >> t & 1 and not bmask >> t & 1 and t != me:
                oacts.append((a, t, False))
        if mine < MAXB and not bmask >> me & 1:
            acts.append((5, me, True))
        opp_can_bomb = theirs < MAXB
        if opp_can_bomb and not bmask >> opp & 1:
            oacts.append((5, opp, True))

        memo = {}

        def base(mb, ob, late=None):
            key = (mb, ob, late)
            r = memo.get(key)
            if r is None:
                bb = bombs
                if mb or ob:
                    bb = dict(bombs)
                    if mb:
                        bb[me] = FUSE
                    if ob:
                        bb[opp] = FUSE
                sch = _schedule(mp, bb, shr, late)
                r = memo[key] = (sch, _back(sch, None, 0))
            return r

        def par(mb, ob, y, y2):
            """(cells where I am safe after this step even if the opponent's body blocks my way,
            the same if it drops a bomb in the next step, does it survive that bomb itself)."""
            key = (mb, ob, y2, 0)
            r = memo.get(key)
            if r is None:
                sch, S = base(mb, ob)
                wb = S[ix]
                if sch[3] > 1:
                    wb = _back(sch, _reach(sch, y, S, y2), 0)[ix]
                wl = -1
                alive = False
                if opp_can_bomb and not (ob and y == opp) and not sch[1][2] >> y2 & 1:
                    S2 = base(mb, ob, (y2, FUSE + 1))[1]
                    wl = S2[ix]
                    alive = S2[2] >> y2 & 1
                r = memo[key] = (wb, wl, alive)
            return r

        def blk(ob, c):
            """Cells where the opponent can still save itself after this step if I stand on c for good."""
            key = (ob, c, 1)
            r = memo.get(key)
            if r is None:
                sch, S = base(False, ob)
                anyhit = 0
                for h in sch[0]:
                    anyhit |= h
                r = memo[key] = -1 if anyhit >> c & 1 else _back(sch, None, 1 << c)[ix]
            return r

        cells = {t for _, t, _ in acts}
        dist = _dists(free, 1 << opp, cells)
        ring_pen = {}
        tfree = free
        proj = None
        if mp2 is not None and tshr <= 14:
            din = _dists(free, free & ~ringm, cells)
            for c in cells:
                ring_pen[c] = (15 - tshr) * 1.5 * din[c] if din[c] else 0.0
                if din[c] >= tshr:  # cannot be inside in time: the teleport decides where I land
                    ring_pen[c] += 20.0
            if tshr <= 8:  # territory on the board as it will be
                tfree = mp2.free
                proj = tele0
        terr_need = min(6, tfree.bit_count() // 4)
        exposed = {}
        for mb in (False, True):
            m = 0
            for h in base(mb, False)[0][0]:
                m |= h
            exposed[mb] = m

        nopp = len(oacts)
        best_a, best_sc = 0, -1e18
        nesc_stay = nopp
        for a, tm, mb in acts:
            nloss = ndraw = npb = npl = npk = nesc = n2 = nb = 0
            stay_dead = False
            for bi, (b, to, ob) in enumerate(oacts):
                if tm == to:
                    c, y = me, opp
                else:
                    c, y = tm, to
                sch, S = base(mb, ob)
                h1 = sch[0][1]
                if now:  # both players are moved by the shrink before the next step
                    if p == 0:
                        c2, y2 = _teleport_pair(inner, ringm, c, y, sch[1][2])
                    else:
                        y2, c2 = _teleport_pair(inner, ringm, y, c, sch[1][2])
                else:
                    c2, y2 = c, y
                s1 = S[ix]
                o_ok = s1 >> y2 & 1 and not h1 >> y & 1
                nesc += o_ok
                if h1 >> c & 1 or not s1 >> c2 & 1:
                    if o_ok:
                        nloss += 1
                    else:
                        ndraw += 1
                    if bi == 0:
                        stay_dead = True
                else:
                    wb, wl, alive = par(mb, ob, y, y2)
                    if not wb >> c2 & 1:
                        npb += 1
                    elif not wl >> c2 & 1:
                        if alive:
                            npl += 1
                        else:
                            npk += 1
                if not mb:
                    # one ply deeper: I drop a bomb on my cell in the next step
                    if mine < MAXB and not sch[1][2] >> c2 & 1:
                        S2 = base(False, ob, (c2, FUSE + 1))[1]
                        if (S2[ix] >> y2 & 1 and not h1 >> y & 1) or not S2[2] >> c2 & 1:
                            n2 += 1
                    else:
                        n2 += 1
                    # or I simply stay there as a blocker
                    if not (bombs or ob) or (blk(ob, c2) >> y2 & 1 and not h1 >> y & 1):
                        nb += 1
            if a == 0:
                nesc_stay = nesc
            c0 = tm
            sure = nloss == 0 and ndraw == 0
            npar = npb + npl + npk
            sc = -(1e6 * nloss + 3e5 * ndraw + 2000.0 * (npb + npl) + 300.0 * npk)
            if stay_dead:
                sc += -5e7 + 1e5 * _alive_time(base(mb, False)[0], c0)
            if nesc == 0 and sure:
                sc += 1e5  # forced kill
            elif mb:
                if npar == 0 and sure and mp.blast[me] >> opp & 1 and (mine == 0 or nesc < nesc_stay):
                    sc += 6.0 + 2.0 * (nesc_stay - nesc)
                else:
                    sc -= 500.0
            if not mb:
                if sure and (n2 == 0 or nb == 0):
                    sc += 300.0
                else:
                    sc += 1.5 * (nopp - n2)
            sc -= 1.0 * dist[c0]
            sc -= ring_pen.get(c0, 0.0)
            if exposed[mb] >> c0 & 1:
                sc -= 6.0  # inside a coming blast: get out early
            if proj is None:
                pc, po = c0, opp
            else:
                pc, po = proj.get(c0, c0), proj.get(opp, opp)
            terr = _voronoi(tfree, pc, po).bit_count()
            if terr < terr_need:
                sc -= 8.0 * (terr_need - terr)
            oterr = _voronoi(tfree, po, pc).bit_count()
            if oterr < terr_need:
                sc += 2.0 * (terr_need - oterr)
            sc += 0.3 * random.random()
            if self.trace is not None:
                self.trace.append((a, "loss", nloss, "draw", ndraw, "par", npb, npl, npk, "esc", nesc, n2, nb,
                                   "terr", terr, oterr, round(sc, 2)))
            if sc > best_sc:
                best_a, best_sc = a, sc
        return best_a
