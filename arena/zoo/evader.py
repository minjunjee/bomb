"""evader -- a survivor (rule-based, bitboards).

Priority 1 is never to be hit; it bombs only when that is completely safe and the opponent stands in the blast.

How it decides (step 1 = the step being decided; a bomb with obs fuse f explodes at the end of step f):

1. HARD SAFETY = a small exact game search on bitboards.
   `_back` computes by backward induction, for a set of bombs, the cells on which one may stand at the end of
   step k and still outrun every explosion (chain reactions, bombs blocking the way, bombs that vanish and escape
   routes that close at a sudden-death shrink, teleport included).
   `_dp` then solves the simultaneous-move game for the next DEPTH steps: W_j(o) = my cells at the end of step j,
   with the opponent on o, from which I survive WHATEVER the opponent does in the following steps: stay, move
   (its body blocks me; if we both go for the same cell nobody moves), or drop a bomb on its cell (after its bomb
   it is assumed passive).  An action is "safe" when it is safe against every reply.  If no action is safe the
   search is repeated with a smaller depth, and at depth 0 it takes the action that survives the most replies
   (if it is surely dead it drops a bomb: a chain that also reaches the opponent turns a loss into a draw).
2. SAFETY MARGIN (soft score among the safe actions):
     - L: for how many steps ahead it can follow a path that is safe against a fresh bomb dropped from ANY cell
       the opponent could have reached by then (static table F[c] = cells from which a bomb on c can be escaped):
       this keeps it out of dead ends and corridors when the opponent is near,
     - the number of cells it reaches strictly before the opponent (territory), with a bonus for cells that no
       single bomb can trap,
     - the walking distance to the opponent, not standing in the opponent's blast lines.
3. SUDDEN DEATH.  From 20 steps before a shrink it keeps a reserve of steps to walk inside; during the last 8
   steps the soft score is taken on the board as it will be after the shrink.
4. BOMB only if the opponent is inside the blast AND the bomb is safe in the search of point 1.
"""

import random

from bombcompetition import RULES

FUSE = RULES.fuse
RAD = RULES.radius
MAXB = RULES.max_bombs
EVERY = RULES.shrink_every or 0
MAXSTEPS = RULES.max_steps
N = 9
NN = N * N
ALL = (1 << NN) - 1
RING = [min(i // N, i % N, N - 1 - i // N, N - 1 - i % N) for i in range(NN)]
MOVES4 = ((1, -N), (2, N), (3, -1), (4, 1))  # action, index step: UP, DOWN, LEFT, RIGHT
DEPTH = 3  # steps of the exact game search
DQ = 6  # look-ahead of the "any fresh bomb" path test

_TABS = {}  # walls bytes -> _Tab (static tables of one wall layout)


class _Tab:
    __slots__ = ("wl", "open", "bl", "nb", "cells", "F", "U", "shr", "qc", "bc")


def _build(wl):
    """Static tables of a wall layout (wl: 9x9 nested list, True = wall)."""
    t = _Tab()
    t.wl = wl
    opn = 0
    for r in range(N):
        row = wl[r]
        for c in range(N):
            if not row[c]:
                opn |= 1 << (r * N + c)
    t.open = opn
    bl = [0] * NN
    nb = [0] * NN
    cells = []
    for i in range(NN):
        if (opn >> i) & 1:
            cells.append(i)
            m = 1 << i
            for d in (-N, N, -1, 1):
                j = i
                for _ in range(RAD):
                    j += d
                    if j < 0 or j >= NN or not (opn >> j) & 1:
                        break
                    m |= 1 << j
            bl[i] = m
            b = 1 << i
            nb[i] = (b << 1 | b >> 1 | b << N | b >> N) & opn
    t.bl = bl
    t.nb = nb
    t.cells = cells
    # F[c]: cells on which one may stand at the end of the step in which a bomb is dropped on c (alone on the
    # board) and still get out of its blast in the 4 moves that are left.  U: cells no single bomb can trap.
    F = [0] * NN
    U = opn
    for c in cells:
        cb = 1 << c
        S = opn & ~bl[c]
        for _ in range(FUSE - 1):
            E = S & ~cb
            S = (S | E << 1 | E >> 1 | E << N | E >> N) & opn
        F[c] = S
        U &= S | cb
    t.F = F
    t.U = U
    t.shr = {}
    t.qc = {}
    t.bc = {}
    return t


def _shrunk(tab, k):
    """What the shrink of ring k does to this layout: (tab after, ring mask, teleports, {ring cell: target}) or None."""
    v = tab.shr.get(k, 0)
    if v != 0:
        return v
    inner = [i for i in tab.cells if RING[i] > k]
    if not inner:
        v = None
    else:
        wl2 = [row[:] for row in tab.wl]
        for i in range(NN):
            if RING[i] == k:
                wl2[i // N][i % N] = True
        key = bytes(bytearray(1 if x else 0 for row in wl2 for x in row))
        t2 = _TABS.get(key)
        if t2 is None:
            t2 = _TABS[key] = _build(wl2)
        rm = 0
        tele = []
        dest = {}
        for x in tab.cells:
            if RING[x] != k:
                continue
            rm |= 1 << x
            r0, c0 = divmod(x, N)
            order = sorted(inner, key=lambda i: (abs(i // N - r0) + abs(i % N - c0), i))[:4]
            tele.append((1 << x, tuple(order)))
            dest[x] = order[0]
        v = (t2, rm, tuple(tele), dest)
    tab.shr[k] = v
    return v


def _balls(tab, c):
    """b[j] = cells within j steps of c (walls only)."""
    b = tab.bc.get(c)
    if b is None:
        opn = tab.open
        f = 1 << c
        b = [f]
        for _ in range(9):
            f = (f | f << 1 | f >> 1 | f << N | f >> N) & opn
            b.append(f)
        tab.bc[c] = b
    return b


def _chain(bl, cells, tm):
    """Chain reactions, in place: a bomb inside an earlier blast goes off with it."""
    n = len(cells)
    if n > 1:
        ch = True
        while ch:
            ch = False
            for a in range(n):
                ta = tm[a]
                ba = bl[cells[a]]
                for b in range(n):
                    if tm[b] > ta and (ba >> cells[b]) & 1:
                        tm[b] = ta
                        ch = True
    return tm


def _prep(tab, shr, ks, cells, times):
    """(cells, explosion steps) of the bombs that really explode: chains, and bombs on a closing ring vanish."""
    bl = tab.bl
    tm = _chain(bl, cells, list(times))
    if shr is not None:
        rm = shr[1]
        while True:
            keep = [a for a in range(len(cells)) if not (tm[a] > ks and (rm >> cells[a]) & 1)]
            if len(keep) == len(cells):
                break
            cells = [cells[a] for a in keep]
            times = [times[a] for a in keep]
            tm = _chain(bl, cells, list(times))
    return cells, tm


def _back(tab, shr, ks, cells, tm, kstop, out=None):
    """S[kstop]: the cells on which one may stand at the end of step kstop and still survive all these bombs
    (cells/tm: bomb cell, step at whose end it explodes).  A bomb blocks the way until it has exploded.
    shr/ks: a shrink happens at the end of step ks (S[ks] = cells before the teleport).  out[k] = S[k], k >= kstop."""
    opn = tab.open
    n = len(cells)
    if n == 0:
        return opn
    K = max(tm)
    if K < kstop:
        return opn
    bl = tab.bl
    H = [0] * (K + 1)
    D = [0] * (K + 1)
    for a in range(n):
        t = tm[a]
        c = cells[a]
        H[t] |= bl[c]
        D[t] |= 1 << c
    if shr is not None and K > ks:
        oa = shr[0].open
    else:
        oa = opn
        ks = 0
    S = oa & ~H[K]
    P = D[K]
    k = K
    if out is not None:
        out[K] = S
    while k > kstop:
        E = S & ~P  # cells that can be entered in step k
        S |= E << 1 | E >> 1 | E << N | E >> N
        k -= 1
        if k > ks:
            S &= oa & ~H[k]
        elif k == ks:  # S = where I may be after the shrink; go back to the cells before the teleport
            S &= oa
            pre = S
            for xb, order in shr[2]:
                for d in order:
                    if not (P >> d) & 1:
                        break
                if (S >> d) & 1:
                    pre |= xb
            S = pre & ~H[k]
        else:
            S &= opn & ~H[k]
        P |= D[k]
        if out is not None:
            out[k] = S
    return S


def _sdrop(tab, shr, ks, cells, tm, o, j):
    """Safe cells at the end of step j+1 if a bomb is dropped on o in step j+1 (it explodes at the end of j+FUSE)."""
    j1 = j + 1
    c2 = []
    t2 = []
    for a in range(len(cells)):
        if tm[a] >= j1:  # earlier bombs are gone by then
            c2.append(cells[a])
            t2.append(tm[a])
    if c2:
        t0 = list(t2)
        c2.append(o)
        t2.append(j + FUSE)
        _chain(tab.bl, c2, t2)
        if shr is not None and t2[-1] > ks and (shr[1] >> o) & 1:  # dropped on the closing ring: it vanishes
            c2.pop()
            t2 = t0
    else:
        if shr is not None and j + FUSE > ks and (shr[1] >> o) & 1:
            return tab.open
        c2.append(o)
        t2.append(j + FUSE)
    return _back(tab, shr, ks, c2, t2, j1)


def _dp(tab, shr, ks, cells, tm, m0, o0, canb, can1, D):
    """The game search.  Returns (Sb1, Sd0, W1):
    Sb1: my safe cells at the end of step 1 against a passive opponent;
    Sd0: the same if the opponent drops a bomb now;
    W1[c]: my safe cells at the end of step 1 if the opponent then stands on c and nothing new was dropped,
           safe against everything it can do in the steps 2..D (a missing c means Sb1)."""
    opn = tab.open
    bl = tab.bl
    nb = tab.nb
    n = len(cells)
    K = max(tm) if n else 0
    top = (K if K > D else D) + 2
    Sb = [opn] * top
    Pb = [0] * top
    if n:
        _back(tab, shr, ks, cells, tm, 1, Sb)
        for a in range(n):
            bit = 1 << cells[a]
            for k in range(1, tm[a] + 1):
                Pb[k] |= bit
    Wn = {}
    if D > 1:
        mball = _balls(tab, m0)
        hball = _balls(tab, o0)
        r = 4 if n else 1
        for j in range(D - 1, 0, -1):
            Wj = {}
            pj = Pb[j + 1]
            hdom = opn & ~pj  # cells that can be entered in step j+1
            sbj = Sb[j]
            sbn = Sb[j + 1]
            mreach = mball[j + r]
            wm = 0
            for c in Wn:
                wm |= 1 << c
            g = hball[j]
            while g:
                lb = g & -g
                g ^= lb
                o = lb.bit_length() - 1
                nbo = nb[o] & hdom  # cells the opponent can move to
                need = canb and not pj & lb and bl[o] & mreach
                if not need and not wm & (lb | nbo):
                    continue  # the opponent on o cannot matter: W_j(o) = Sb[j]
                Sd = _sdrop(tab, shr, ks, cells, tm, o, j) if need else ALL
                Wo = Wn.get(o, sbn)
                T0 = T = Sd & Wo  # my cell after step j+1 must be safe if it bombs, and if it stays
                terms = []
                h = nbo
                while h:
                    l2 = h & -h
                    h ^= l2
                    t2 = Wn.get(l2.bit_length() - 1, sbn) | l2  # ... and if it moves to l2 (I am not on l2 then)
                    terms.append((l2, t2))
                    T &= t2
                E = hdom & ~lb  # cells I can enter in step j+1
                mv = T & E
                X = mv & ~nbo
                Y = mv & nbo  # cells it can also go for: then nobody moves, and I must be safe where I am
                W = T | X << 1 | X >> 1 | X << N | X >> N
                if Y:
                    W |= (Y << 1 | Y >> 1 | Y << N | Y >> N) & Wo
                for l2, _ in terms:  # my cells next to it: it cannot step onto my cell
                    Tm = T0
                    for l3, t3 in terms:
                        if l3 != l2:
                            Tm &= t3
                    mvm = Tm & E & nb[l2.bit_length() - 1]
                    if Tm & l2 or mvm & ~nbo or (mvm & nbo and Wo & l2):
                        W |= l2
                    else:
                        W &= ~l2
                Wj[o] = W & sbj & ~lb
            Wn = Wj
    Sd0 = _sdrop(tab, shr, ks, cells, tm, o0, 0) if can1 else ALL
    return Sb[1], Sd0, Wn


def _qlist(tab, o):
    """q[j]: cells that are safe to stand on at the end of step j+1 if the opponent (now on o) drops a fresh bomb
    in that step from any cell it can have reached by then."""
    q = tab.qc.get(o)
    if q is None:
        F = tab.F
        opn = tab.open
        acc = F[o]
        q = [acc]
        ball = f = 1 << o
        for _ in range(DQ - 1):
            f = (f << 1 | f >> 1 | f << N | f >> N) & opn & ~ball
            ball |= f
            g = f
            while g:
                lb = g & -g
                acc &= F[lb.bit_length() - 1] | lb
                g ^= lb
            q.append(acc)
        tab.qc[o] = q
    return q


def _vor(dom, yb, ob):
    """Cells I (on yb) reach strictly before the opponent (on ob)."""
    fa = mine = yb
    fb = ob
    seen = yb | ob
    while fa:
        fb = (fb << 1 | fb >> 1 | fb << N | fb >> N) & dom & ~seen
        seen |= fb
        fa = (fa << 1 | fa >> 1 | fa << N | fa >> N) & dom & ~seen
        seen |= fa
        mine |= fa
    return mine


def _safe(y, m0, o0, hopts, can1, Sb1, Sd0, W1):
    """Is ending this step on y (my target; m0 = where I am) safe against every reply of the opponent?"""
    if can1 and not (Sd0 >> y) & 1:
        return False
    w0 = W1.get(o0, Sb1)
    if not (w0 >> y) & 1:
        return False
    for c in hopts:
        if c == o0:
            continue
        if c == y:  # it goes for the same cell: nobody moves
            if not (w0 >> m0) & 1:
                return False
        elif not (W1.get(c, Sb1) >> y) & 1:
            return False
    return True


class Policy:
    def __init__(self):
        pass

    def act(self, obs):
        try:
            a = self._act(obs)
            return a if a in (0, 1, 2, 3, 4, 5) else 0
        except Exception:  # noqa: BLE001  (never raise: stay put)
            return 0

    def _act(self, obs):
        walls = obs["walls"]
        try:
            key = walls.tobytes()
        except AttributeError:
            import numpy as np

            walls = np.asarray(walls, dtype=bool)
            key = walls.tobytes()
        tab = _TABS.get(key)
        if tab is None:
            if len(_TABS) > 600:
                _TABS.clear()
            tab = _TABS[key] = _build(walls.tolist())
        me, op = obs["me"], obs["opp"]
        m0 = me[0] * N + me[1]
        o0 = op[0] * N + op[1]
        mb = 1 << m0
        pl = obs["player"]
        step = obs["step"]
        bc, bt = [], []
        bmask = 0
        mine = his = 0
        for b in obs["bombs"]:
            p = b["pos"]
            i = p[0] * N + p[1]
            bc.append(i)
            bt.append(b["fuse"])
            bmask |= 1 << i
            if b["owner"] == pl:
                mine += 1
            else:
                his += 1
        opn = tab.open
        dom = opn & ~bmask
        shr = None
        ks = 0
        depth = DEPTH
        if EVERY:
            left = EVERY - step % EVERY
            if step + left < MAXSTEPS:
                shr = _shrunk(tab, step // EVERY + 1)
                if shr is not None:
                    ks = left
                    if ks < depth:
                        depth = ks  # the search does not look across a shrink (the bomb timing does)

        # my moves and the cells the opponent can stand on after this step
        cand = [(0, m0)]
        for a, d in MOVES4:
            y = m0 + d
            if (dom >> y) & 1 and y != o0:
                cand.append((a, y))
        hopts = [o0]
        for _, d in MOVES4:
            c = o0 + d
            if (dom >> c) & 1 and c != m0:
                hopts.append(c)
        canb = his < MAXB
        can1 = canb and not (bmask >> o0) & 1
        bl = tab.bl

        # ---- hard safety -------------------------------------------------------------------------------------
        cells, tm = _prep(tab, shr, ks, bc, bt) if bc else (bc, bt)
        d = depth
        while True:
            Sb1, Sd0, W1 = _dp(tab, shr, ks, cells, tm, m0, o0, canb, can1, d)
            top = [x for x in cand if _safe(x[1], m0, o0, hopts, can1, Sb1, Sd0, W1)]
            if top or d <= 1:
                break
            d -= 1
        full = bool(top) and d == depth
        if not top:
            # no action survives every reply: take the one that survives the most (a bomb now counts triple)
            best = -1
            for a, y in cand:
                n = 2 * ((Sb1 >> y) & 1)
                if can1:
                    n += 3 * ((Sd0 >> y) & 1)
                for c in hopts[1:]:
                    n += (Sb1 >> (m0 if c == y else y)) & 1
                if n > best:
                    best = n
                    top = [(a, y)]
                elif n == best:
                    top.append((a, y))
            if best == 0 and mine < MAXB and not (bmask >> m0) & 1:
                return 5  # surely hit: a chained bomb of mine may reach the opponent too (draw instead of loss)

        # ---- bomb: only when completely safe and the opponent is in the blast ---------------------------------
        if full and mine < MAXB and not (bmask >> m0) & 1 and (bl[m0] >> o0) & 1:
            c2, t2 = _prep(tab, shr, ks, bc + [m0], bt + [FUSE])
            SbB, SdB, WB = _dp(tab, shr, ks, c2, t2, m0, o0, canb, can1, depth)
            if _safe(m0, m0, o0, hopts, can1, SbB, SdB, WB):
                q = 0  # replies of the opponent that survive my bomb
                for c in hopts:
                    q += (SbB >> c) & 1
                if q <= 2 or random.random() < 0.4:
                    return 5

        if len(top) == 1:
            return top[0][0]

        # ---- safety margin -----------------------------------------------------------------------------------
        use2 = shr is not None and ks <= 8
        if use2:  # judge positions on the board as it will be after the shrink
            t2 = shr[0]
            dest = shr[3]
            oS = dest.get(o0, o0)
        else:
            t2 = tab
            dest = None
            oS = o0
        opn2 = t2.open
        domS = opn2 & ~bmask
        osb = 1 << oS
        q = _qlist(t2, oS)
        lv = [osb]
        seen = f = osb
        while True:
            f = (f << 1 | f >> 1 | f << N | f >> N) & domS & ~seen
            if not f:
                break
            seen |= f
            lv.append(f)
        blo = t2.bl[oS]
        U = t2.U
        lin = None
        if shr is not None and ks <= 20:  # walking distance to the area that stays open
            inner = shr[0].open
            lin = [inner]
            seen = f = inner
            while True:
                f = (f << 1 | f >> 1 | f << N | f >> N) & dom & ~seen
                if not f:
                    break
                seen |= f
                lin.append(f)
        bs = -1e9
        ba = 0
        for a, y in top:
            yS = dest.get(y, y) if use2 else y
            yb = 1 << yS
            if yb == osb:
                s = -20.0
            else:
                f = yb
                L = 0
                for qq in q:
                    f &= qq
                    if not f:
                        break
                    L += 1
                    f = (f | f << 1 | f >> 1 | f << N | f >> N) & opn2
                terr = _vor(domS, yb, osb)
                hd = 9
                for j in range(len(lv)):
                    if lv[j] & yb:
                        hd = j
                        break
                s = 5.0 * L + terr.bit_count() + 0.5 * (terr & U).bit_count() + 3.0 * (hd if hd < 6 else 6)
                if blo & yb:
                    s -= 4.0
            if lin is not None and not (lin[0] >> y) & 1:
                din = 99
                for j in range(1, len(lin)):
                    if (lin[j] >> y) & 1:
                        din = j
                        break
                slack = ks - 1 - din
                if slack < 3:
                    s -= 25.0 * (3 - slack) if din < 99 else 30.0
            s += 0.5 * random.random()
            if s > bs:
                bs = s
                ba = a
        return ba
