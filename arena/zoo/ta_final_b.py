"""ta_final_b -- a positional "territory and cut-off" bot.

Idea: think in regions, not in chases.
  * Bombs are temporary walls: a bomb blocks its cell until it explodes and then burns a cross. The bot looks for
    the moment when one more bomb leaves the opponent no way to outrun the explosions (a sealed pocket, corridor
    or dead end), and for moves that set this up one or two steps ahead.
  * Between fights it maximises territory (cells it reaches first) and squeezes the opponent's region; near a
    sudden-death shrink only the cells that survive the shrink count, so it takes the centre early.
  * It never takes an action unless it keeps an escape, also if the opponent drops a bomb now or after its next
    move, and it prefers escapes that the opponent cannot body-block.

Implementation: the 9x9 board is an 81-bit integer (bit r*9+c); explosion timing is exact (chains, bombs block,
the shrink removes ring bombs and teleports players). Time k = the end of the k-th step from now (the step being
decided is k = 1): a bomb with obs fuse f explodes at time f, a bomb dropped now at time 5.
One step is searched exactly for every (my action, opponent action) pair; the leaves are judged with a backward
"can still outrun everything" set, plus the bomb each side could drop on the step after.
"""

import random

from bombcompetition import RULES

_FUSE = RULES.fuse
_RAD = RULES.radius
_MAXB = RULES.max_bombs
_EVERY = RULES.shrink_every or 0
_MAXT = RULES.max_steps
_ALL = (1 << 81) - 1
_CACHE = {}

_WIN = 1000.0
_LOSS = -1000.0
_BOTH = -400.0


class _Tab:
    __slots__ = ("O", "BL", "shrink", "RING", "O2", "BL2", "TP", "DIN")


def _blasts(wl):
    out = [0] * 81
    for i in range(81):
        if wl[i]:
            continue
        msk = 1 << i
        for d in (-9, 9, -1, 1):
            j = i
            for _ in range(_RAD):
                j += d
                if j < 0 or j > 80 or wl[j]:
                    break
                msk |= 1 << j
        out[i] = msk
    return out


def _tables(walls, kn):
    key = (walls.tobytes(), kn)
    T = _CACHE.get(key)
    if T is not None:
        return T
    if len(_CACHE) > 3000:
        _CACHE.clear()
    wl = [bool(x) for x in walls.ravel().tolist()]
    T = _Tab()
    O = 0
    for i in range(81):
        if not wl[i]:
            O |= 1 << i
    T.O = O
    T.BL = _blasts(wl)
    T.shrink = False
    T.RING = 0
    T.O2 = O
    T.BL2 = T.BL
    T.TP = {}
    T.DIN = None
    if kn:
        ring, inner = [], []
        for i in range(81):
            r, c = divmod(i, 9)
            ro = min(r, c, 8 - r, 8 - c)
            if ro == kn:
                ring.append(i)
            elif ro > kn and not wl[i]:
                inner.append(i)
        if inner:
            T.shrink = True
            wl2 = list(wl)
            rm = 0
            for i in ring:
                wl2[i] = True
                if not wl[i]:
                    rm |= 1 << i
            T.RING = rm
            o2 = 0
            for i in inner:
                o2 |= 1 << i
            T.O2 = o2
            T.BL2 = _blasts(wl2)
            tp = {}
            for i in ring:
                if wl[i]:
                    continue
                r, c = divmod(i, 9)
                tp[i] = sorted(inner, key=lambda y: (abs(y // 9 - r) + abs(y % 9 - c), y))
            T.TP = tp
            din = [99] * 81
            for i in inner:
                din[i] = 0
            F = seen = o2
            d = 0
            while F:
                d += 1
                F = ((F << 9) | (F >> 9) | (F << 1) | (F >> 1)) & O & ~seen
                seen |= F
                x = F
                while x:
                    lb = x & -x
                    x ^= lb
                    din[lb.bit_length() - 1] = d
            T.DIN = din
    _CACHE[key] = T
    return T


def _tl(T, ks, bl):
    """Timeline of a bomb list [(cell, time, birth step)] -> (K, S, nP, ks, sm, U) or None (no explosion).
    S[k] = cells that are open and not exploding at time k, nP[k] = ~(bombs in the way during the move of step k),
    U = every cell that explodes, ks != 0: the shrink happens inside the horizon (then S is post-shrink for k > ks)."""
    n = len(bl)
    if not n:
        return None
    BL = T.BL
    tm = [x[1] for x in bl]
    if n > 1:
        ch = True
        while ch:
            ch = False
            for a in range(n):
                ta = tm[a]
                msk = BL[bl[a][0]]
                for c in range(n):
                    if tm[c] > ta and (msk >> bl[c][0]) & 1 and bl[c][2] <= ta:
                        tm[c] = ta
                        ch = True
    K = max(tm)
    if ks and K > ks:
        return _tl_shrink(T, ks, bl, tm)
    O = T.O
    H = [0] * (K + 1)
    P = [0] * (K + 1)
    U = 0
    for a in range(n):
        i, _, k = bl[a]
        t = tm[a]
        h = BL[i]
        H[t] |= h
        U |= h
        bit = 1 << i
        while k <= t:
            P[k] |= bit
            k += 1
    return (K, [O & ~h for h in H], [~p for p in P], 0, 0, U)


def _tl_shrink(T, ks, bl, tm):
    """The same when the shrink (at the end of step ks) comes before the last explosion: bombs on the ring that
    have not exploded by then vanish, the others are re-chained with the new walls."""
    n = len(bl)
    BL, BL2, RING, O, O2 = T.BL, T.BL2, T.RING, T.O, T.O2
    surv = [a for a in range(n) if tm[a] > ks and not (RING >> bl[a][0]) & 1]
    tm2 = list(tm)
    if surv:
        for a in surv:
            tm2[a] = bl[a][1]
        ch = True
        while ch:
            ch = False
            for a in surv:
                ta = tm2[a]
                msk = BL2[bl[a][0]]
                for c in surv:
                    if tm2[c] > ta and (msk >> bl[c][0]) & 1 and bl[c][2] <= ta:
                        tm2[c] = ta
                        ch = True
        K = max(tm2[a] for a in surv)
    else:
        K = 0
        for a in range(n):
            if K < tm[a] <= ks:
                K = tm[a]
        if not K:
            return None
    H = [0] * (K + 1)
    P = [0] * (K + 1)
    U = 0
    sm = 0
    ss = set(surv)
    for a in range(n):
        i, _, k = bl[a]
        bit = 1 << i
        t = tm[a]
        if t <= ks:
            H[t] |= BL[i]
            U |= BL[i]
            while k <= t:
                P[k] |= bit
                k += 1
        elif a in ss:
            t2 = tm2[a]
            H[t2] |= BL2[i]
            U |= BL2[i]
            if k <= ks:
                sm |= bit
            while k <= t2:
                P[k] |= bit
                k += 1
        else:
            while k <= ks and k <= K:
                P[k] |= bit
                k += 1
    S = [((O if k <= ks else O2) & ~H[k]) for k in range(K + 1)]
    return (K, S, [~p for p in P], (ks if surv else 0), sm, U)


def _safe(T, tl, k0=1):
    """Backward induction: the cells where a player standing after step k0 can still outrun every explosion."""
    if tl is None:
        return T.O
    K = tl[0]
    S = tl[1]
    nP = tl[2]
    ks = tl[3]
    if k0 >= K:
        return S[K] if k0 == K else T.O2
    W = S[K]
    k = K
    if ks and ks >= k0:
        while k > ks + 1:
            X = W & nP[k]
            k -= 1
            W = (W | (X << 9) | (X >> 9) | (X << 1) | (X >> 1)) & S[k]
        X = W & nP[k]
        Wp = (W | (X << 9) | (X >> 9) | (X << 1) | (X >> 1)) & T.O2  # right after the shrink
        sm = tl[4]
        Wq = Wp
        for x, order in T.TP.items():  # a player on the ring is teleported
            y = order[0]
            if (sm >> y) & 1:
                for y2 in order:
                    if not (sm >> y2) & 1:
                        y = y2
                        break
            if (Wp >> y) & 1:
                Wq |= 1 << x
        k = ks
        W = Wq & S[k]
    while k > k0:
        X = W & nP[k]
        k -= 1
        W = (W | (X << 9) | (X >> 9) | (X << 1) | (X >> 1)) & S[k]
    return W


def _tele(T, F, sm):
    """Positions after the shrink: players on the ring are moved to the nearest open inner cell without a bomb."""
    R = F & T.RING
    out = F & T.O2
    while R:
        lb = R & -R
        R ^= lb
        order = T.TP[lb.bit_length() - 1]
        y = order[0]
        if (sm >> y) & 1:
            for y2 in order:
                if not (sm >> y2) & 1:
                    y = y2
                    break
        out |= 1 << y
    return out


def _block_ok(T, tl, pb, qb, mb):
    """Robust escape: I stand on pb after step 1. Can I still outrun the explosions if the opponent walks to any
    single cell and stays there (a body block)? The shrink (teleports included) is followed exactly."""
    if tl is None:
        return True
    K = tl[0]
    if K < 2:
        return True
    S = tl[1]
    nP = tl[2]
    ks = tl[3]
    sm = tl[4]
    O = T.O
    # cells I may use
    F = pb
    if ks == 1:
        F = _tele(T, F, sm)
    reach = F
    k = 2
    while k <= K:
        F = (F | (((F << 9) | (F >> 9) | (F << 1) | (F >> 1)) & nP[k])) & S[k]
        if not F:
            return False
        if k == ks:
            F = _tele(T, F, sm)
        reach |= F
        k += 1
    # (cell, d): the opponent can stand on the cell from the end of step d on
    pairs = [(qb, 0)]
    G = seen = qb
    d = 0
    if ks == 1:
        G = seen = _tele(T, qb, sm)
        if G != qb:
            pairs = [(G, 0)]
    while True:
        d += 1
        if d > K:
            break
        G = ((G << 9) | (G >> 9) | (G << 1) | (G >> 1)) & nP[d] & (O if (not ks or d <= ks) else T.O2) & ~seen
        if d == 1:
            G &= ~mb
            if ks == 1:
                G = _tele(T, G, sm) & ~seen
        seen |= G
        x = G & reach
        while x:
            cb = x & -x
            x ^= cb
            pairs.append((cb, d))
        if d == ks and d > 1:
            ns = _tele(T, seen, sm)
            x = ns & ~seen & reach
            while x:
                cb = x & -x
                x ^= cb
                pairs.append((cb, d + 1))
            G = seen = ns
        if not G:
            break
    p1 = _tele(T, pb, sm) if ks == 1 else pb
    for cb, d in pairs:
        if cb == p1:
            continue
        ncb = ~cb
        F = p1
        k = 2
        while k <= K:
            e = ((F << 9) | (F >> 9) | (F << 1) | (F >> 1)) & nP[k]
            if k >= d:
                e &= ncb
            F = (F | e) & S[k]
            if k == ks:
                F = _tele(T, F, sm)
                if d <= k + 1:
                    F &= ncb
            if not F:
                return False
            k += 1
    return True


def _hit_time(tl, pb):
    S = tl[1]
    for k in range(1, tl[0] + 1):
        if not (S[k] & pb):
            return k
    return 0


def _voro(free, pb, qb, cm):
    """Territory: (cells of cm I reach strictly first, cells the opponent reaches strictly first)."""
    free &= ~(pb | qb)
    A = pb
    B = qb
    ca = cb = 0
    while True:
        na = ((A << 9) | (A >> 9) | (A << 1) | (A >> 1)) & free
        nb = ((B << 9) | (B >> 9) | (B << 1) | (B >> 1)) & free
        u = na | nb
        if not u:
            break
        both = na & nb
        ca += ((na ^ both) & cm).bit_count()
        cb += ((nb ^ both) & cm).bit_count()
        free &= ~u
        A = na
        B = nb
    return ca, cb


class Policy:
    def __init__(self):
        self.prev = -1
        self.dbg = None

    def act(self, obs):
        try:
            a = int(self._act(obs))
            if 0 <= a <= 5:
                return a
        except Exception:  # noqa: BLE001  (never fault)
            pass
        return 0

    def _act(self, obs):
        walls = obs["walls"]
        step = obs["step"]
        me = obs["player"]
        mr, mc = obs["me"]
        qr, qc = obs["opp"]
        m = mr * 9 + mc
        q = qr * 9 + qc
        kn = step // _EVERY + 1 if _EVERY else 0
        if kn * _EVERY >= _MAXT:
            kn = 0
        T = _tables(walls, kn)
        O = T.O
        BL = T.BL
        RING = T.RING
        ks = kn * _EVERY - step if T.shrink else 0
        bl0 = []
        bmask = 0
        mine = theirs = 0
        for b in obs["bombs"]:
            r, c = b["pos"]
            i = r * 9 + c
            bl0.append((i, b["fuse"], 1))
            bmask |= 1 << i
            if b["owner"] == me:
                mine += 1
            else:
                theirs += 1
        mb = 1 << m
        qb = 1 << q
        free = O & ~bmask
        md = abs(mr - qr) + abs(mc - qc)

        nbm = ((mb << 9) | (mb >> 9) | (mb << 1) | (mb >> 1)) & free & ~qb
        cand = [(0, m)]
        if (nbm >> (m - 9)) & 1:
            cand.append((1, m - 9))
        if (nbm >> (m + 9)) & 1:
            cand.append((2, m + 9))
        if (nbm >> (m - 1)) & 1:
            cand.append((3, m - 1))
        if (nbm >> (m + 1)) & 1:
            cand.append((4, m + 1))
        can_bomb = mine < _MAXB and not (bmask & mb)
        qn = ((qb << 9) | (qb >> 9) | (qb << 1) | (qb >> 1)) & free & ~mb  # cells the opponent can walk to
        orep = qb | qn  # where the opponent can be after this step
        near = md <= 6
        opp_now = near and theirs < _MAXB and not (bmask & qb)

        # bomb sets after this step: v = (I drop a bomb) + 2 * (the opponent drops one)
        blv = [bl0, None, None, None]
        if can_bomb:
            blv[1] = bl0 + [(m, _FUSE, 1)]
        if opp_now:
            blv[2] = bl0 + [(q, _FUSE, 1)]
            if can_bomb:
                blv[3] = blv[1] + [(q, _FUSE, 1)]
        tlv = [None] * 4
        Wv = [O] * 4
        for v in range(4):
            if blv[v]:
                tlv[v] = _tl(T, ks, blv[v])
                Wv[v] = _safe(T, tlv[v])
        W0 = Wv[0]

        # the opponent's actions: (cell after the step, drops a bomb)
        oacts = [(q, 0)]
        if near:
            x = qn
            while x:
                lb = x & -x
                x ^= lb
                oacts.append((lb.bit_length() - 1, 0))
            if opp_now:
                oacts.append((q, 1))

        c2 = {}
        TP = T.TP

        def w2(v, cell, cell2=-1):
            """Safe set (after this step) if a bomb is dropped on `cell` (and `cell2`) in the next step."""
            key = (v, cell, cell2)
            w = c2.get(key)
            if w is None:
                if ks == 1 and (RING >> cell) & 1:
                    cell = TP[cell][0]
                bl = blv[v] + [(cell, _FUSE + 1, 2)]
                if cell2 >= 0:
                    if ks == 1 and (RING >> cell2) & 1:
                        cell2 = TP[cell2][0]
                    bl.append((cell2, _FUSE + 1, 2))
                w = c2[key] = _safe(T, _tl(T, ks, bl))
            return w

        leaves = {}

        def leaf(pme, popp, ib, ob):
            v = ib + 2 * ob
            key = (pme, popp, v)
            val = leaves.get(key)
            if val is not None:
                return val
            W = Wv[v]
            mbit = 1 << pme
            obit = 1 << popp
            if not (W & mbit):
                val = _LOSS if (W & obit) else _BOTH
            else:
                to = 0  # the opponent's bomb on the next step leaves me no escape (2 = it dies with me)
                o_can = near and not ob and theirs < _MAXB and not (bmask & obit)
                if o_can:
                    Wo = w2(v, popp)
                    if not (Wo & mbit):
                        to = 1 if (Wo & obit) else 2
                if not (W & obit):
                    val = 600.0 if to else _WIN
                else:
                    tm = False  # my bomb on the next step leaves the opponent no escape
                    if near and not ib and mine < _MAXB and not (bmask & mbit):
                        Wx = w2(v, pme)
                        if (Wx & mbit) and not (Wx & obit):
                            tm = True
                            if o_can and not (w2(v, pme, popp) & mbit):
                                tm = False
                    if to:
                        val = -150.0 if tm else (-500.0 if to == 1 else -250.0)
                    else:
                        val = 500.0 if tm else 0.0
            leaves[key] = val
            return val

        late = ks and ks <= 10
        din = T.DIN
        q2 = qb
        free2 = 0
        if late:  # judge territory on the board as it will be after the shrink
            O2 = T.O2
            free2 = O2 & ~bmask
            if not (qb & O2):
                q2 = 1 << TP[q][0]
        tl0 = tlv[0]
        U0 = tl0[5] if tl0 else 0
        doomed = not (orep & W0)  # the opponent cannot outrun the bombs that are already there
        scores = {}
        pos_stay = 0.0
        nact = len(oacts)
        for a, p in cand:
            pb = 1 << p
            mn = 1e9
            tot = 0.0
            for oc, ob in oacts:
                if oc == p:  # both walk into the same cell: nobody moves
                    val = leaf(m, q, 0, 0)
                else:
                    val = leaf(p, oc, 0, ob)
                tot += val
                if val < mn:
                    mn = val
            sc = mn + 0.3 * tot / nact
            # territory
            if late:
                p2 = pb
                if not (pb & O2):
                    order = TP[p]
                    p2 = 1 << order[0]
                    if p2 == q2 and len(order) > 1:
                        p2 = 1 << order[1]
                if p2 == q2:
                    ca = cb = 0
                else:
                    ca, cb = _voro(free2, p2, q2, O2)
            else:
                ca, cb = _voro(free, pb, qb, O)
            pos = float(ca - cb)
            if ca < 6:
                pos -= 2.5 * (6 - ca)
                if ca < 2:
                    pos -= 6.0
            if cb < 6:
                pos += 1.5 * (6 - cb)
            pr, pc = divmod(p, 9)
            pos -= 0.05 * (abs(pr - qr) + abs(pc - qc))
            if ks and (RING >> p) & 1:
                slack = ks - 1 - din[p]
                if slack < 0:
                    pos -= 12.0
                elif slack < 3:
                    pos -= 6.0
                elif ks <= 15:
                    pos -= 0.3 * (15 - ks)
            if p == self.prev and not bl0:
                pos -= 0.3
            if a == 0:
                pos_stay = pos
            if U0 & pb and mn > -300:
                th = _hit_time(tl0, pb)
                if th:
                    pos -= 2.0 + 1.0 * (6 - th)  # do not linger inside a blast
                if not _block_ok(T, tl0, pb, qb, mb):
                    sc -= 200.0
            scores[a] = sc + pos + random.random() * 0.01

        if can_bomb:
            mn = 1e9
            tot = 0.0
            for oc, ob in oacts:
                val = leaf(m, oc, 1, ob)
                tot += val
                if val < mn:
                    mn = val
            sc = mn + 0.3 * tot / nact
            bonus = -4.0
            if mn >= 0.0 and not doomed:
                robust = _block_ok(T, tlv[1], mb, qb, mb)
                if not robust:
                    sc -= 200.0
                elif mn < 900.0 and md <= 5:
                    nb = (orep & Wv[1]).bit_count()
                    if BL[m] & qb:
                        bonus += 5.0
                    if nb <= 2:
                        bonus += 3.0
                    if mine >= 2:
                        bonus -= 4.0
                    # two bombs: this one, a step aside, one more
                    if mine + 1 < _MAXB:
                        x = nbm
                        while x:
                            lb = x & -x
                            x ^= lb
                            i = lb.bit_length() - 1
                            tl3 = _tl(T, ks, blv[1] + [(i, _FUSE + 2, 3)])
                            if not (orep & _safe(T, tl3)) and (_safe(T, tl3, 3) & lb):
                                bonus += 150.0
                                break
            else:
                bonus -= 6.0
            scores[5] = sc + pos_stay + bonus + random.random() * 0.01

        # two quiet steps, then a bomb that seals the opponent whatever it does meanwhile
        if near and md <= 5 and mine < _MAXB and not doomed:
            seen3 = {}
            for a, p in cand:
                if scores[a] < -100.0:
                    continue
                pb = 1 << p
                x = (pb | (((pb << 9) | (pb >> 9) | (pb << 1) | (pb >> 1)) & free & ~qb)) & ~bmask
                hit = False
                while x and not hit:
                    lb = x & -x
                    x ^= lb
                    i = lb.bit_length() - 1
                    r3 = seen3.get(i)
                    if r3 is None:
                        tl3 = _tl(T, ks, bl0 + [(i, _FUSE + 2, 3)])
                        r3 = seen3[i] = (not (orep & _safe(T, tl3))) and bool(_safe(T, tl3, 3) & lb)
                    hit = r3
                if hit:
                    scores[a] += 150.0

        best = max(scores, key=scores.get)
        self.prev = m
        self.dbg = scores
        return best
