"""x_matrix: an EXPERT (teacher) bot. Simultaneous-move game-tree search with matrix-game backups.

Stateless: act(obs) depends only on obs. The only things kept between calls are per-map tables (keyed by the walls)
and scratch buffers; the transposition table is invalidated on every call (a fresh salt), so the answer for an obs
never depends on earlier calls. The search budget is a NODE count (not a wall-clock time), so it is deterministic.

How it works
------------
* Forward model (numba, verified against bombcompetition.Game): the 7x7 interior is 49 cells, sets of cells are
  49-bit masks in an int64. A state is (p0, p1, f1, f2, f3, f4, own0, t): the two positions, the bombs by fuse
  (mask of the bombs whose obs fuse is 1..4), the mask of player 0's bombs and the step counter. Shrinks at 100/200
  (ring -> walls, bombs on the ring vanish, teleport) are part of the model.
* Search: depth-limited over JOINT actions. Equivalent actions are merged (a blocked move and an impossible BOMB are
  STAY). Every node builds the payoff matrix M[my action][opponent action] of the child values and backs up the
  mixed-strategy value of the zero-sum matrix game (saddle point if there is one, else fictitious play), blended
  with a little of "best row against an opponent that plays uniformly over its non-suicidal actions".
  Iterative deepening (depth 1, 2, ...) inside a node budget, transposition table keyed by (state, depth).
* Leaf evaluation: an exact time-expanded survival analysis of both players for the bombs on the board (chain
  reactions, bombs as walls until they explode, a shrink inside the horizon), each in two variants (the other
  player absent / the other player standing still as a wall), then territory (simultaneous flood fill), distance,
  bomb pressure (how many safe cells are left).
* Root: the matrix of the deepest completed iteration. A row that wins against every column is taken; a row that
  loses against some column is avoided if another row does not; otherwise the row with the best value against a
  blend of the opponent's equilibrium mix and a uniform non-suicidal opponent, minus a risk term toward its worst
  case. Ties are broken in a fixed order in the player's own (turned) frame.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# ------------------------------------------------------------------------------------------------------------------
# tunables
# ------------------------------------------------------------------------------------------------------------------
NODE_BUDGET = 250000  # search budget per act() in nodes (about 0.5 us each); the one knob for speed vs strength
MAX_DEPTH = 12        # iterative deepening stops here at the latest

DRAW_VALUE = -0.2     # both hit / 300 steps: worth nothing, but better than losing
NULL_VALUE = -0.2     # a joint action that changes nothing (no bombs, nobody moved): it repeats for ever
LAMBDA_NODE = 0.15    # inner nodes: weight of the "uniform non-suicidal opponent" value
LAMBDA_ROOT = 0.30    # root: the same weight for the choice of the action
RISK_ROOT = 0.25      # root: pull of a row's value toward its worst case
W_TERR = 0.30         # territory (cells I reach first minus cells the opponent reaches first, normalised)
W_DIST = 0.012        # per step of walking distance to the opponent (closer is better: we are the hunter)
W_FRONT = 0.15        # bomb pressure: 1 / (safe cells left) of the opponent minus mine
W_BLOCK = 0.60        # the opponent dies if I just stand still (and I can afford to) / the reverse
SHRINK_NEAR = 12      # this many steps before a shrink only cells that stay open count as territory

# ------------------------------------------------------------------------------------------------------------------
# constants of the compiled code
# ------------------------------------------------------------------------------------------------------------------
WIN_T = 0.8           # a value >= WIN_T is a (practically) forced win, <= LOSS_T a forced loss
LOSS_T = -0.8
FP_ITERS = 48         # fictitious-play iterations for a node without a saddle point
FP_ROOT = 600

T_SHRINK = 0          # table layout (one int64 array per (walls, phase))
RING = 1
OPEN = 2              # [2]: open mask now, open mask after the next shrink
BLAST = 8             # [2 * 49]
NBR = BLAST + 98      # [2 * 49 * 4]: target cell of UP, DOWN, LEFT, RIGHT or -1
TELE = NBR + 392      # [49 * 26]: inner cells nearest first (the engine's order), -1 terminated
DIST = TELE + 49 * 26  # [2 * 49 * 49]: walking distance, 99 = unreachable
TAB_SIZE = DIST + 2 * 2401

P_DRAW, P_LAM, P_WTERR, P_WDIST, P_WFRONT, P_LROOT, P_RHO, P_NEAR, P_WBLOCK, P_NULL = range(10)

NC0 = 0
NC6 = 0
for _i in range(49):
    if _i % 7 != 0:
        NC0 |= 1 << _i
    if _i % 7 != 6:
        NC6 |= 1 << _i
DR = (-1, 1, 0, 0)
DC = (0, 0, -1, 1)
FLIP = (0, 2, 1, 4, 3, 5)
TT_BITS = 18
TT_MASK = (1 << TT_BITS) - 1
NO_SHRINK = 1 << 40


@njit(cache=True)
def _pc(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return x & 0x7F


@njit(cache=True)
def _ctz(b):
    """Index of the single set bit of b."""
    c = 0
    if b >= 4294967296:
        c += 32
        b >>= 32
    if b >= 65536:
        c += 16
        b >>= 16
    if b >= 256:
        c += 8
        b >>= 8
    if b >= 16:
        c += 4
        b >>= 4
    if b >= 4:
        c += 2
        b >>= 2
    if b >= 2:
        c += 1
    return c


@njit(cache=True)
def _exp(f):
    """The 4-neighbours of a set of cells (not yet masked with the open cells)."""
    return (f << 7) | (f >> 7) | ((f << 1) & NC0) | ((f >> 1) & NC6)


# ------------------------------------------------------------------------------------------------------------------
# per-map tables
# ------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _build(w, step):
    """w: int64[81], 1 = wall (the 9x9 board row by row). Tables for the walls now and after the next shrink."""
    tab = np.zeros(TAB_SIZE, np.int64)
    wa = np.zeros(49, np.int64)
    for r in range(7):
        for c in range(7):
            wa[r * 7 + c] = w[(r + 1) * 9 + c + 1]
    wb = wa.copy()
    k = step // 100 + 1
    ring = 0
    ts = NO_SHRINK
    if k <= 2:
        inner = 0
        for r in range(7):
            for c in range(7):
                rg = min(min(r, c), min(6 - r, 6 - c)) + 1
                if rg > k and wa[r * 7 + c] == 0:
                    inner += 1
        if inner > 0:
            ts = k * 100
            for r in range(7):
                for c in range(7):
                    rg = min(min(r, c), min(6 - r, 6 - c)) + 1
                    if rg == k:
                        ring |= 1 << (r * 7 + c)
                        wb[r * 7 + c] = 1
    tab[T_SHRINK] = ts
    tab[RING] = ring
    queue = np.zeros(64, np.int64)
    for ph in range(2):
        wl = wa
        if ph == 1:
            wl = wb
        opn = 0
        for i in range(49):
            if wl[i] == 0:
                opn |= 1 << i
        tab[OPEN + ph] = opn
        for r in range(7):
            for c in range(7):
                i = r * 7 + c
                bl = 0
                for d in range(4):
                    rr = r + DR[d]
                    cc = c + DC[d]
                    tg = -1
                    if wl[i] == 0 and rr >= 0 and rr < 7 and cc >= 0 and cc < 7 and wl[rr * 7 + cc] == 0:
                        tg = rr * 7 + cc
                    tab[NBR + ph * 196 + i * 4 + d] = tg
                if wl[i] == 0:
                    bl = 1 << i
                    for d in range(4):
                        for q in range(1, 3):
                            rr = r + DR[d] * q
                            cc = c + DC[d] * q
                            if rr < 0 or rr >= 7 or cc < 0 or cc >= 7 or wl[rr * 7 + cc] != 0:
                                break
                            bl |= 1 << (rr * 7 + cc)
                tab[BLAST + ph * 49 + i] = bl
        for src in range(49):
            base = DIST + ph * 2401 + src * 49
            for x in range(49):
                tab[base + x] = 99
            if wl[src] != 0:
                continue
            tab[base + src] = 0
            queue[0] = src
            qh = 0
            qt = 1
            while qh < qt:
                x = queue[qh]
                qh += 1
                for d in range(4):
                    y = tab[NBR + ph * 196 + x * 4 + d]
                    if y >= 0 and tab[base + y] == 99:
                        tab[base + y] = tab[base + x] + 1
                        queue[qt] = y
                        qt += 1
    # teleport order of the next shrink: the inner open cells by (Manhattan distance, row, col)
    for i in range(49 * 26):
        tab[TELE + i] = -1
    if ts != NO_SHRINK:
        for r in range(7):
            for c in range(7):
                i = r * 7 + c
                n = 0
                for x in range(49):
                    if wb[x] == 0:
                        key = (abs(x // 7 - r) + abs(x % 7 - c)) * 64 + x
                        # insertion sort on (distance, cell)
                        pos = n
                        while pos > 0:
                            y = tab[TELE + i * 26 + pos - 1]
                            ky = (abs(y // 7 - r) + abs(y % 7 - c)) * 64 + y
                            if ky <= key:
                                break
                            tab[TELE + i * 26 + pos] = y
                            pos -= 1
                        tab[TELE + i * 26 + pos] = x
                        n += 1
    return tab


# ------------------------------------------------------------------------------------------------------------------
# forward model
# ------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _tele(c, other, ab, tab):
    base = TELE + c * 26
    for k in range(26):
        x = tab[base + k]
        if x < 0:
            break
        if x != other and ((ab >> x) & 1) == 0:
            return x
    return tab[base]


@njit(cache=True)
def _step(p0, p1, f1, f2, f3, f4, own0, t, a0, a1, tab):
    """One engine step. Returns (res, p0, p1, f1, f2, f3, f4, own0, t) with
    res: 0 = goes on, 1 = player 0 wins, 2 = player 1 wins, 3 = both hit (draw), 4 = step limit (draw)."""
    ts = tab[T_SHRINK]
    ph = 0
    if t >= ts:
        ph = 1
    allb = f1 | f2 | f3 | f4
    f5 = 0
    b0 = 1 << p0
    b1 = 1 << p1
    if a0 == 5 and (allb & b0) == 0 and _pc(own0) < 3:
        f5 |= b0
    if a1 == 5 and ((allb | f5) & b1) == 0 and _pc(allb & ~own0) < 3:
        f5 |= b1
    if (f5 & b0) != 0:
        own0 |= b0
    ab = allb | f5
    n0 = p0
    n1 = p1
    if a0 >= 1 and a0 <= 4:
        tg = tab[NBR + ph * 196 + p0 * 4 + a0 - 1]
        if tg >= 0 and tg != p1 and ((ab >> tg) & 1) == 0:
            n0 = tg
    if a1 >= 1 and a1 <= 4:
        tg = tab[NBR + ph * 196 + p1 * 4 + a1 - 1]
        if tg >= 0 and tg != p0 and ((ab >> tg) & 1) == 0:
            n1 = tg
    if n0 == n1:
        n0 = p0
        n1 = p1
    res = 0
    if f1 != 0:
        ex = f1
        todo = f1
        blast = 0
        while todo != 0:
            lsb = todo & (-todo)
            todo ^= lsb
            bl = tab[BLAST + ph * 49 + _ctz(lsb)]
            blast |= bl
            nw = bl & ab & ~ex
            ex |= nw
            todo |= nw
        h0 = (blast >> n0) & 1
        h1 = (blast >> n1) & 1
        if h0 != 0 and h1 != 0:
            res = 3
        elif h1 != 0:
            res = 1
        elif h0 != 0:
            res = 2
        keep = ~ex
        f2 &= keep
        f3 &= keep
        f4 &= keep
        f5 &= keep
        own0 &= keep
    t += 1
    f1 = f2
    f2 = f3
    f3 = f4
    f4 = f5
    if res == 0:
        if t >= 300:
            res = 4
        elif t == ts:
            ring = tab[RING]
            keep = ~ring
            f1 &= keep
            f2 &= keep
            f3 &= keep
            f4 &= keep
            own0 &= keep
            ab2 = f1 | f2 | f3 | f4
            if ((ring >> n0) & 1) != 0:
                n0 = _tele(n0, n1, ab2, tab)
            if ((ring >> n1) & 1) != 0:
                n1 = _tele(n1, n0, ab2, tab)
    return res, n0, n1, f1, f2, f3, f4, own0, t


# ------------------------------------------------------------------------------------------------------------------
# leaf evaluation
# ------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _tele_set(f, ring, tab):
    on = f & ring
    f &= ~ring
    while on != 0:
        lsb = on & (-on)
        on ^= lsb
        x = tab[TELE + _ctz(lsb) * 26]
        if x >= 0:
            f |= 1 << x
    return f


@njit(cache=True)
def _eval(p0, p1, f1, f2, f3, f4, own0, t, me, tr, tab, par):
    """Value of a position for player `me` (the game is not over)."""
    ts = tab[T_SHRINK]
    ph = 0
    if t >= ts:
        ph = 1
    opn0 = tab[OPEN + ph]
    if me == 0:
        pm = p0
        po = p1
    else:
        pm = p1
        po = p0
    bm = 1 << pm
    bo = 1 << po
    allb = f1 | f2 | f3 | f4
    s = ts - t
    dt = t - tr

    dm_a = 0  # the step in which I am hit for sure (0 = I can get away), the opponent ignored
    dm_b = 0  # the same with the opponent standing still as a wall
    do_a = 0
    do_b = 0
    sm = 0    # the step in which my present cell explodes (0 = never)
    so = 0
    fm = bm
    gm = bm
    fo = bo
    go = bo
    if allb != 0:
        rem = allb
        opn = opn0
        cph = ph
        for k in range(1, 5):
            if rem == 0:
                break
            if k == 1:
                fk = f1
            elif k == 2:
                fk = f2
            elif k == 3:
                fk = f3
            else:
                fk = f4
            seed = fk & rem
            bl = 0
            ex = 0
            if seed != 0:
                ex = seed
                todo = seed
                while todo != 0:
                    lsb = todo & (-todo)
                    todo ^= lsb
                    b = tab[BLAST + cph * 49 + _ctz(lsb)]
                    bl |= b
                    nw = b & rem & ~ex
                    ex |= nw
                    todo |= nw
            walk = opn & ~rem
            nbl = ~bl
            if dm_a == 0:
                fm = (fm | (_exp(fm) & walk)) & nbl
                if fm == 0:
                    dm_a = k
            if dm_b == 0:
                gm = (gm | (_exp(gm) & walk & ~bo)) & nbl
                if gm == 0:
                    dm_b = k
            if do_a == 0:
                fo = (fo | (_exp(fo) & walk)) & nbl
                if fo == 0:
                    do_a = k
            if do_b == 0:
                go = (go | (_exp(go) & walk & ~bm)) & nbl
                if go == 0:
                    do_b = k
            if sm == 0 and (bl & bm) != 0:
                sm = k
            if so == 0 and (bl & bo) != 0:
                so = k
            rem &= ~ex
            if k == s:
                ring = tab[RING]
                rem &= ~ring
                opn = tab[OPEN + 1]
                cph = 1
                fm = _tele_set(fm, ring, tab)
                gm = _tele_set(gm, ring, tab)
                fo = _tele_set(fo, ring, tab)
                go = _tele_set(go, ring, tab)

    if do_a > 0:  # the opponent cannot get away
        if dm_a == 0:
            if dm_b == 0 or dm_b > do_a:
                return 0.92 - 0.002 * (dt + do_a)
            return 0.35
        if dm_a > do_a:
            return 0.85 - 0.002 * (dt + do_a)
        if dm_a == do_a:
            return par[P_DRAW]
        return -0.9 + 0.002 * (dt + dm_a)
    if dm_a > 0:  # I cannot get away, the opponent can
        if do_b > 0 and do_b < dm_a:
            return -0.3
        if do_b > 0 and do_b == dm_a:
            return -0.6
        return -0.92 + 0.002 * (dt + dm_a)

    v = 0.0
    if do_b > 0 and (sm == 0 or sm > do_b):
        v += par[P_WBLOCK]
    if dm_b > 0 and (so == 0 or so > dm_b):
        v -= par[P_WBLOCK]

    # territory: simultaneous flood fill, bombs are walls
    free = opn0 & ~allb & ~bm & ~bo
    if s > 0 and s <= par[P_NEAR]:
        free &= ~tab[RING]
    xm = bm
    xo = bo
    cm = 0
    co = 0
    while (xm | xo) != 0:
        nm = _exp(xm) & free
        no = _exp(xo) & free
        both = nm & no
        cm |= nm & ~both
        co |= no & ~both
        free &= ~(nm | no)
        xm = nm
        xo = no
    tm = _pc(cm)
    to = _pc(co)
    h = par[P_WTERR] * (tm - to) / (tm + to + 2.0)
    d = tab[DIST + ph * 2401 + pm * 49 + po]
    if d >= 99:
        h = 0.3 * h
        if ts == NO_SHRINK or ph == 1:
            h += 0.5 * par[P_DRAW]
    else:
        h -= par[P_WDIST] * d
    if allb != 0:
        h += par[P_WFRONT] * (1.0 / _pc(fo) - 1.0 / _pc(fm))
    v += h
    if v > 0.78:
        v = 0.78
    elif v < -0.78:
        v = -0.78
    return v


# ------------------------------------------------------------------------------------------------------------------
# matrix game
# ------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _solve(M, n, m, lam, sc):
    """Value of the zero-sum matrix game M[0:n, 0:m] (row player maximises), blended with lam * (best row against
    a uniform non-suicidal column player) unless the game is decided. sc: float scratch (18)."""
    lo = -1e18
    bi = 0
    for i in range(n):
        r = M[i, 0]
        for j in range(1, m):
            x = M[i, j]
            if x < r:
                r = x
        if r > lo:
            lo = r
            bi = i
    hi = 1e18
    bj = 0
    for j in range(m):
        c = M[0, j]
        for i in range(1, n):
            x = M[i, j]
            if x > c:
                c = x
        if c < hi:
            hi = c
            bj = j
    v = lo
    if hi - lo > 1e-7:
        for i in range(n):
            sc[i] = 0.0
        for j in range(m):
            sc[6 + j] = 0.0
        i = bi
        j = bj
        best = 0.0
        worst = 0.0
        for it in range(FP_ITERS):
            best = -1e18
            ni = 0
            for ii in range(n):
                sc[ii] += M[ii, j]
                if sc[ii] > best:
                    best = sc[ii]
                    ni = ii
            worst = 1e18
            nj = 0
            for jj in range(m):
                sc[6 + jj] += M[i, jj]
                if sc[6 + jj] < worst:
                    worst = sc[6 + jj]
                    nj = jj
            i = ni
            j = nj
        v = 0.5 * (best + worst) / FP_ITERS
        if v < lo:
            v = lo
        if v > hi:
            v = hi
    if lam > 0.0 and lo < WIN_T and hi > LOSS_T:
        cnt = 0
        for i in range(n):
            sc[12 + i] = 0.0
        for j in range(m):
            mn = M[0, j]
            for i in range(1, n):
                if M[i, j] < mn:
                    mn = M[i, j]
            if mn < WIN_T:
                cnt += 1
                for i in range(n):
                    sc[12 + i] += M[i, j]
        if cnt > 0:
            b = -1e18
            for i in range(n):
                if sc[12 + i] > b:
                    b = sc[12 + i]
            v += lam * (b / cnt - v)
    return v


# ------------------------------------------------------------------------------------------------------------------
# search
# ------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _hash(p0, p1, f1, f2, f3, f4, own0, t, depth, salt):
    h = salt ^ (p0 | (p1 << 6) | (t << 12) | (depth << 22))
    h = (h ^ f1) * (-7046029254386353131)
    h = (h ^ (h >> 31) ^ f2) * (-4658895280553007687)
    h = (h ^ (h >> 29) ^ f3) * (-7723592293110705685)
    h = (h ^ (h >> 32) ^ f4) * (-7046029254386353131)
    h = (h ^ (h >> 30) ^ own0) * (-4658895280553007687)
    h ^= h >> 31
    return h


@njit(cache=True)
def _legal(pc, po, cnt, allb, ph, tab, flip, ia, off):
    """The distinct actions of the player on cell pc (STAY, the moves that are not blocked, BOMB if it drops one),
    in the order of the player `me`'s turned frame. Returns their number; they are written to ia[off:]."""
    n = 0
    for k in range(6):
        a = k
        if flip != 0:
            a = FLIP[k]
        ok = False
        if a == 0:
            ok = True
        elif a == 5:
            ok = ((allb >> pc) & 1) == 0 and cnt < 3
        else:
            tg = tab[NBR + ph * 196 + pc * 4 + a - 1]
            ok = tg >= 0 and tg != po and ((allb >> tg) & 1) == 0
        if ok:
            ia[off + n] = a
            n += 1
    return n


@njit(cache=True)
def _term(res, me, nt, tr, par):
    if res >= 3:
        return par[P_DRAW]
    if (res == 1 and me == 0) or (res == 2 and me == 1):
        return 1.0 - 0.002 * (nt - tr)
    return -1.0 + 0.002 * (nt - tr)


@njit(cache=True)
def _search(depth, me, tr, tab, par, ttk, ttv, ctl, ms, sc, ia, st):
    """Fill the root matrix ms[0] for a search of `depth` plies. Frame 0 of the stack `st` (the root: state, n, m,
    its action lists in ia[0:12]) is prepared by the caller. An explicit stack instead of recursion.
    st[k] = (p0, p1, f1, f2, f3, f4, own0, t, i, j, n, m, tt key, remaining depth). Returns False if the node
    budget ran out (ms[0] is then incomplete)."""
    lam = par[P_LAM]
    st[0, 8] = 0
    st[0, 9] = 0
    st[0, 13] = depth
    sp = 0
    while True:
        i = st[sp, 8]
        j = st[sp, 9]
        if i >= st[sp, 10]:  # every child of this frame has its value
            if sp == 0:
                return True
            v = _solve(ms[sp], st[sp, 10], st[sp, 11], lam, sc)
            key = st[sp, 12]
            slot = key & TT_MASK
            ttk[slot] = key
            ttv[slot] = v
            sp -= 1
            i = st[sp, 8]
            j = st[sp, 9]
        else:
            am = ia[sp * 12 + i]
            ao = ia[sp * 12 + 6 + j]
            if me == 0:
                res, q0, q1, g1, g2, g3, g4, o0, nt = _step(st[sp, 0], st[sp, 1], st[sp, 2], st[sp, 3], st[sp, 4],
                                                            st[sp, 5], st[sp, 6], st[sp, 7], am, ao, tab)
            else:
                res, q0, q1, g1, g2, g3, g4, o0, nt = _step(st[sp, 0], st[sp, 1], st[sp, 2], st[sp, 3], st[sp, 4],
                                                            st[sp, 5], st[sp, 6], st[sp, 7], ao, am, tab)
            d = st[sp, 13] - 1
            if res != 0:
                v = _term(res, me, nt, tr, par)
            else:
                ctl[0] += 1
                nb = g1 | g2 | g3 | g4
                if (nb == 0 and q0 == st[sp, 0] and q1 == st[sp, 1] and nt != tab[T_SHRINK]
                        and (st[sp, 2] | st[sp, 3] | st[sp, 4] | st[sp, 5]) == 0):
                    # nothing changed: two memoryless players repeat this for ever
                    v = par[P_NULL]
                elif d <= 0:
                    v = _eval(q0, q1, g1, g2, g3, g4, o0, nt, me, tr, tab, par)
                else:
                    if ctl[0] > ctl[1]:
                        ctl[2] = 1
                        return False
                    decided = False
                    if nb != 0:
                        v = _eval(q0, q1, g1, g2, g3, g4, o0, nt, me, tr, tab, par)
                        decided = v >= WIN_T or v <= LOSS_T
                    key = _hash(q0, q1, g1, g2, g3, g4, o0, nt, d, ctl[3])
                    slot = key & TT_MASK
                    if decided:
                        pass
                    elif ttk[slot] == key:
                        v = ttv[slot]
                    else:  # a new frame
                        sp += 1
                        st[sp, 0] = q0
                        st[sp, 1] = q1
                        st[sp, 2] = g1
                        st[sp, 3] = g2
                        st[sp, 4] = g3
                        st[sp, 5] = g4
                        st[sp, 6] = o0
                        st[sp, 7] = nt
                        st[sp, 8] = 0
                        st[sp, 9] = 0
                        st[sp, 12] = key
                        st[sp, 13] = d
                        ph = 0
                        if nt >= tab[T_SHRINK]:
                            ph = 1
                        allb = g1 | g2 | g3 | g4
                        c0 = _pc(o0)
                        c1 = _pc(allb & ~o0)
                        if me == 0:
                            st[sp, 10] = _legal(q0, q1, c0, allb, ph, tab, me, ia, sp * 12)
                            st[sp, 11] = _legal(q1, q0, c1, allb, ph, tab, me, ia, sp * 12 + 6)
                        else:
                            st[sp, 10] = _legal(q1, q0, c1, allb, ph, tab, me, ia, sp * 12)
                            st[sp, 11] = _legal(q0, q1, c0, allb, ph, tab, me, ia, sp * 12 + 6)
                        continue
        # the value v of child (i, j) of frame sp
        ms[sp, i, j] = v
        j += 1
        if j >= st[sp, 11]:
            j = 0
            i += 1
        st[sp, 8] = i
        st[sp, 9] = j


@njit(cache=True)
def _think(p0, p1, f1, f2, f3, f4, own0, t, me, max_depth, tab, par, ttk, ttv, ctl, ms, sc, ia, st, out):
    """Iterative deepening at the root. out[0:6] = the value of each real action, out[6] = the chosen action,
    out[7] = completed depth, out[8] = nodes, out[9] = game value (maximin row)."""
    tr = t
    ph = 0
    if t >= tab[T_SHRINK]:
        ph = 1
    allb = f1 | f2 | f3 | f4
    c0 = _pc(own0)
    c1 = _pc(allb & ~own0)
    if me == 0:
        n = _legal(p0, p1, c0, allb, ph, tab, me, ia, 0)
        m = _legal(p1, p0, c1, allb, ph, tab, me, ia, 6)
    else:
        n = _legal(p1, p0, c1, allb, ph, tab, me, ia, 0)
        m = _legal(p0, p1, c0, allb, ph, tab, me, ia, 6)
    st[0, 0] = p0
    st[0, 1] = p1
    st[0, 2] = f1
    st[0, 3] = f2
    st[0, 4] = f3
    st[0, 5] = f4
    st[0, 6] = own0
    st[0, 7] = t
    st[0, 10] = n
    st[0, 11] = m
    M = ms[0]
    B = ms[max_depth + 1]  # the matrix of the deepest completed iteration
    done_depth = 0
    for depth in range(1, max_depth + 1):
        ctl[2] = 0
        before = ctl[0]
        if not _search(depth, me, tr, tab, par, ttk, ttv, ctl, ms, sc, ia, st):
            break
        for i in range(n):
            for j in range(m):
                B[i, j] = M[i, j]
        done_depth = depth
        lo = -1e18
        for i in range(n):
            r = B[i, 0]
            for j in range(1, m):
                if B[i, j] < r:
                    r = B[i, j]
            if r > lo:
                lo = r
        hi = 1e18
        for j in range(m):
            c = B[0, j]
            for i in range(1, n):
                if B[i, j] > c:
                    c = B[i, j]
            if c < hi:
                hi = c
        if lo >= WIN_T or hi <= LOSS_T:
            break
        used = ctl[0] - before
        if ctl[0] + 4 * used > ctl[1]:
            break

    # ---- the choice at the root ----
    lo = -1e18
    bi = 0
    for i in range(n):
        r = B[i, 0]
        for j in range(1, m):
            if B[i, j] < r:
                r = B[i, j]
        sc[24 + i] = r  # worst case of row i
        if r > lo:
            lo = r
            bi = i
    hi = 1e18
    bj = 0
    for j in range(m):
        c = B[0, j]
        for i in range(1, n):
            if B[i, j] > c:
                c = B[i, j]
        sc[30 + j] = c
        if c < hi:
            hi = c
            bj = j
    # the opponent's equilibrium mix y (sc[36:42])
    for j in range(m):
        sc[36 + j] = 0.0
    if hi - lo <= 1e-7:
        cnt = 0
        for j in range(m):
            if sc[30 + j] <= hi + 1e-7:
                sc[36 + j] = 1.0
                cnt += 1
        for j in range(m):
            sc[36 + j] /= cnt
    else:
        for i in range(n):
            sc[i] = 0.0
        for j in range(m):
            sc[6 + j] = 0.0
        i = bi
        j = bj
        for it in range(FP_ROOT):
            sc[36 + j] += 1.0
            best = -1e18
            ni = 0
            for ii in range(n):
                sc[ii] += B[ii, j]
                if sc[ii] > best:
                    best = sc[ii]
                    ni = ii
            worst = 1e18
            nj = 0
            for jj in range(m):
                sc[6 + jj] += B[i, jj]
                if sc[6 + jj] < worst:
                    worst = sc[6 + jj]
                    nj = jj
            i = ni
            j = nj
        for j in range(m):
            sc[36 + j] /= FP_ROOT
    # the non-suicidal columns
    npl = 0
    for j in range(m):
        mn = B[0, j]
        for i in range(1, n):
            if B[i, j] < mn:
                mn = B[i, j]
        if mn < WIN_T:
            sc[42 + j] = 1.0
            npl += 1
        else:
            sc[42 + j] = 0.0
    if npl == 0:
        for j in range(m):
            sc[42 + j] = 1.0
        npl = m
    any_safe = lo > LOSS_T
    any_win = lo >= WIN_T
    lr = par[P_LROOT]
    rho = par[P_RHO]
    for i in range(n):
        e_eq = 0.0
        e_un = 0.0
        for j in range(m):
            e_eq += sc[36 + j] * B[i, j]
            e_un += sc[42 + j] * B[i, j]
        e_un /= npl
        w = sc[24 + i]
        sv = (1.0 - lr) * e_eq + lr * e_un
        sv -= rho * (sv - w)
        if any_win:
            if w >= WIN_T:
                sv = w
            elif sv > WIN_T - 0.02:
                sv = WIN_T - 0.02
        elif any_safe and w <= LOSS_T:
            sv = w - 0.1
        sc[48 + i] = sv
    # values of the 6 real actions; an action that does nothing is worth a little less than STAY
    stay = sc[48]  # row 0 is always STAY
    for a in range(6):
        out[a] = stay - 0.001
    for i in range(n):
        out[ia[i]] = sc[48 + i]
    # fixed tie-break in my turned frame: moves (UP, DOWN, LEFT, RIGHT), STAY, BOMB
    best_a = 0
    best_v = -1e18
    for k in range(6):
        kk = k + 1
        if k == 4:
            kk = 0
        elif k == 5:
            kk = 5
        a = kk
        if me != 0:
            a = FLIP[kk]
        if out[a] > best_v + 1e-9:
            best_v = out[a]
            best_a = a
    out[6] = best_a
    out[7] = done_depth
    out[8] = ctl[0]
    out[9] = lo
    return best_a


# ------------------------------------------------------------------------------------------------------------------
# python side
# ------------------------------------------------------------------------------------------------------------------
_TABS: dict = {}
_TTK = np.zeros(1 << TT_BITS, np.int64)
_TTV = np.zeros(1 << TT_BITS, np.float64)
_CTL = np.zeros(8, np.int64)
_MS = np.zeros((MAX_DEPTH + 2, 6, 6), np.float64)
_SC = np.zeros(64, np.float64)
_IA = np.zeros((MAX_DEPTH + 2) * 12, np.int64)
_ST = np.zeros((MAX_DEPTH + 2, 16), np.int64)
_OUT = np.zeros(16, np.float64)


def default_params() -> np.ndarray:
    par = np.zeros(16, np.float64)
    par[P_DRAW] = DRAW_VALUE
    par[P_LAM] = LAMBDA_NODE
    par[P_WTERR] = W_TERR
    par[P_WDIST] = W_DIST
    par[P_WFRONT] = W_FRONT
    par[P_LROOT] = LAMBDA_ROOT
    par[P_RHO] = RISK_ROOT
    par[P_NEAR] = SHRINK_NEAR
    par[P_WBLOCK] = W_BLOCK
    par[P_NULL] = NULL_VALUE
    return par


def table_for(walls, step: int) -> np.ndarray:
    walls = np.asarray(walls)
    key = (walls.tobytes(), step // 100)
    tab = _TABS.get(key)
    if tab is None:
        if len(_TABS) >= 256:
            _TABS.clear()
        tab = _build(walls.astype(np.int64).ravel(), step)
        _TABS[key] = tab
    return tab


def encode(obs):
    """obs -> (p0, p1, f1, f2, f3, f4, own0, t, me, tab) of the compiled model."""
    step = int(obs["step"])
    tab = table_for(obs["walls"], step)
    me = int(obs["player"])
    r, c = obs["me"]
    pm = (int(r) - 1) * 7 + int(c) - 1
    r, c = obs["opp"]
    po = (int(r) - 1) * 7 + int(c) - 1
    f = [0, 0, 0, 0, 0]
    own0 = 0
    for b in obs["bombs"]:
        r, c = b["pos"]
        bit = 1 << ((int(r) - 1) * 7 + int(c) - 1)
        fu = int(b["fuse"])
        f[1 if fu < 1 else (4 if fu > 4 else fu)] |= bit
        if b["owner"] == 0:
            own0 |= bit
    if me == 0:
        return pm, po, f[1], f[2], f[3], f[4], own0, step, me, tab
    return po, pm, f[1], f[2], f[3], f[4], own0, step, me, tab


class Policy:
    budget = NODE_BUDGET
    max_depth = MAX_DEPTH

    def __init__(self, budget: int | None = None, max_depth: int | None = None, params=None):
        if budget is not None:
            self.budget = int(budget)
        if max_depth is not None:
            self.max_depth = max(1, min(MAX_DEPTH, int(max_depth)))
        self.par = default_params() if params is None else params
        self.info = (0, 0, 0.0)  # (depth, nodes, value) of the last search

    def _run(self, obs):
        p0, p1, f1, f2, f3, f4, own0, t, me, tab = encode(obs)
        _TTK.fill(0)  # an empty transposition table for every call: the answer depends on obs only
        _CTL[0] = 0
        _CTL[1] = self.budget
        _CTL[2] = 0
        _CTL[3] = 0x1E3779B97F4A7C15
        a = _think(p0, p1, f1, f2, f3, f4, own0, t, me, self.max_depth, tab, self.par,
                   _TTK, _TTV, _CTL, _MS, _SC, _IA, _ST, _OUT)
        self.info = (int(_OUT[7]), int(_OUT[8]), float(_OUT[9]))
        return int(a)

    def act(self, obs) -> int:
        try:
            a = self._run(obs)
            return a if 0 <= a <= 5 else 0
        except Exception:  # noqa: BLE001  (never raise)
            return 0

    def action_values(self, obs) -> list:
        """The search's value of each of my 6 actions (real frame, higher = better); act() is its argmax."""
        try:
            self._run(obs)
            return [float(x) for x in _OUT[:6]]
        except Exception:  # noqa: BLE001
            return [0.0, -1.0, -1.0, -1.0, -1.0, -1.0]


def make_policy(arg: str):
    """zoo.x_matrix:<node budget>  or  zoo.x_matrix:<node budget>d<max depth>"""
    budget, _, depth = str(arg).partition("d")

    def factory():
        return Policy(int(budget) if budget else None, int(depth) if depth else None)

    return factory


def _warm():
    w = np.ones((9, 9), bool)
    w[1:8, 1:8] = False
    obs = {"player": 0, "walls": w, "me": (1, 1), "opp": (7, 7), "bombs": (), "lives": (1, 1), "step": 0}
    Policy(budget=200).act(obs)


try:
    _warm()  # compile / load the cached machine code at import time, not inside the first act()
except Exception:  # noqa: BLE001
    pass
