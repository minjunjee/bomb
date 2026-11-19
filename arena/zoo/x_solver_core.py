"""Numba core of zoo.x_solver: an exact forward model of the bomb game and exact AND-OR searches on it.

Board: only the inner 7x7 can ever be open, so a cell is an integer 0..48 (cell = (row-1)*7 + (col-1)) and a set
of cells is one int64 bit mask.  All bombs (at most 6) are packed into one int64, 10 bits per bomb
(cell | fuse << 6 | owner << 9), sorted by cell, so a game state is (p0, p1, bombs, t): four integers.
Per-map tables for the three sudden-death phases live in one flat int64 array T (see the O_* offsets),
the transposition table and the node counters in one flat int64 array W.

Searches (all exact with respect to the engine's step, chains and both shrinks):
  relaxed_safe  a cheap SOUND test "the runner survives d steps whatever the chaser does" (open-loop path through
                cells the chaser can neither reach, block nor blow up in time); the main pruning device
  survive       exists my move, for all opponent moves: I am not hit for d steps        (runner never bombs)
  kill          exists my move, for all opponent moves: the opponent alone is hit within d steps
  root_eval     everything the policy needs about the 6 x 6 joint actions of the current position
"""

import numpy as np
from numba import njit

# ------------------------------------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------------------------------------
FULL = (1 << 49) - 1
_c0 = 0
_c6 = 0
for _r in range(7):
    _c0 |= 1 << (_r * 7)
    _c6 |= 1 << (_r * 7 + 6)
NC0 = FULL & ~_c0  # cells that are not in column 0 (they can move left)
NC6 = FULL & ~_c6  # cells that are not in column 6 (they can move right)

O_NBR = 0  # T[O_NBR + ph*245 + cell*5 + action]  target cell of action 0..4, -1 = wall
O_BL = 735  # T[O_BL + ph*49 + cell]              blast mask of a bomb on that cell
O_OPEN = 882  # T[O_OPEN + ph]                    open cells
O_SHR = 885  # T[O_SHR + k]   k = 1, 2            1 if the shrink at step 100*k really happens
O_RING = 888  # T[O_RING + k]                     the cells of ring k
O_TEL = 891  # T[O_TEL + (k-1)*1225 + cell*25 + j] teleport candidates of a ring cell, best first, -1 = end
T_SIZE = O_TEL + 2 * 49 * 25

W_NODES = 0
W_LIMIT = 1
W_GEN = 2
W_OVER = 3
W_TT = 8
TT_BITS = 16
TT_MASK = (1 << TT_BITS) - 1
W_SIZE = W_TT + 3 * (1 << TT_BITS)

_DB = 0x03F79D71B4CB0A89
_t = [0] * 64
for _i in range(64):
    _t[(((1 << _i) * _DB) & 0xFFFFFFFFFFFFFFFF) >> 58] = _i
DBT = np.array(_t, dtype=np.int64)  # index of a single set bit: DBT[((b * _DB) >> 58) & 63]

DEC = np.array([sum(64 << (10 * i) for i in range(n)) for n in range(7)], dtype=np.int64)  # fuse -1 for n bombs


def _orders():
    tow = np.zeros(54, np.int64)
    awy = np.zeros(54, np.int64)
    for sr in (-1, 0, 1):
        for sc in (-1, 0, 1):
            code = 3 * (sr + 1) + (sc + 1)
            toward, away = [], []
            if sr < 0:
                toward.append(1), away.append(2)
            if sr > 0:
                toward.append(2), away.append(1)
            if sc < 0:
                toward.append(3), away.append(4)
            if sc > 0:
                toward.append(4), away.append(3)
            rest = [a for a in (1, 2, 3, 4) if a not in toward and a not in away]
            tow[code * 6:code * 6 + 6] = [5] + toward + [0] + rest + away
            awy[code * 6:code * 6 + 6] = away + rest + [0] + toward + [5]
    return tow, awy


ORD_TOW, ORD_AWY = _orders()  # action orders for "go at the other player" / "get away from it" (move ordering only)


# ------------------------------------------------------------------------------------------------------------
# forward model
# ------------------------------------------------------------------------------------------------------------
@njit("i8(i8,i8)", cache=True)
def insert_bomb(bombs, f):
    c = f & 63
    sh = 0
    bb = bombs
    while bb != 0 and (bb & 63) < c:
        bb >>= 10
        sh += 10
    low = bombs & ((1 << sh) - 1)
    return low | (f << sh) | (bb << (sh + 10))


@njit("UniTuple(i8,5)(i8,i8,i8,i8,i8,i8,i8[::1])", cache=True)
def step(p0, p1, bombs, a0, a1, ph, T):
    """One engine step without the shrink. Returns (p0, p1, bombs, hit0, hit1)."""
    occ = 0
    c0 = 0
    c1 = 0
    bb = bombs
    while bb != 0:
        f = bb & 1023
        occ |= 1 << (f & 63)
        if (f >> 9) != 0:
            c1 += 1
        else:
            c0 += 1
        bb >>= 10
    # 1. drops
    if a0 == 5 and c0 < 3 and ((occ >> p0) & 1) == 0:
        bombs = insert_bomb(bombs, p0 | 320)
        occ |= 1 << p0
        c0 += 1
    if a1 == 5 and c1 < 3 and ((occ >> p1) & 1) == 0:
        bombs = insert_bomb(bombs, p1 | 320 | 512)
        occ |= 1 << p1
        c1 += 1
    # 2. moves
    t0 = p0
    if a0 >= 1 and a0 <= 4:
        n = T[O_NBR + ph * 245 + p0 * 5 + a0]
        if n >= 0 and ((occ >> n) & 1) == 0 and n != p1:
            t0 = n
    t1 = p1
    if a1 >= 1 and a1 <= 4:
        n = T[O_NBR + ph * 245 + p1 * 5 + a1]
        if n >= 0 and ((occ >> n) & 1) == 0 and n != p0:
            t1 = n
    if t0 != t1:
        p0 = t0
        p1 = t1
    if bombs == 0:
        return p0, p1, 0, 0, 0
    # 3. tick and explode
    blo = O_BL + ph * 49
    blast = 0
    bb = bombs
    while bb != 0:
        f = bb & 1023
        if ((f >> 6) & 7) == 1:
            blast |= T[blo + (f & 63)]
        bb >>= 10
    if blast == 0:
        return p0, p1, bombs - DEC[c0 + c1], 0, 0
    changed = 1
    while changed != 0:
        changed = 0
        bb = bombs
        while bb != 0:
            c = bb & 63
            if ((blast >> c) & 1) != 0:
                nb = blast | T[blo + c]
                if nb != blast:
                    blast = nb
                    changed = 1
            bb >>= 10
    newb = 0
    sh = 0
    bb = bombs
    while bb != 0:
        f = bb & 1023
        if ((blast >> (f & 63)) & 1) == 0:
            newb |= (f - 64) << sh
            sh += 10
        bb >>= 10
    return p0, p1, newb, (blast >> p0) & 1, (blast >> p1) & 1


@njit("UniTuple(i8,3)(i8,i8,i8,i8,i8[::1])", cache=True)
def shrink(p0, p1, bombs, k, T):
    """Sudden death number k (when the step counter reaches 100*k). Returns (p0, p1, bombs)."""
    if T[O_SHR + k] == 0:
        return p0, p1, bombs
    ring = T[O_RING + k]
    newb = 0
    sh = 0
    occ = 0
    bb = bombs
    while bb != 0:
        f = bb & 1023
        c = f & 63
        if ((ring >> c) & 1) == 0:
            newb |= f << sh
            sh += 10
            occ |= 1 << c
        bb >>= 10
    if ((ring >> p0) & 1) != 0:
        base = O_TEL + (k - 1) * 1225 + p0 * 25
        tgt = T[base]
        for j in range(25):
            x = T[base + j]
            if x < 0:
                break
            if ((occ >> x) & 1) == 0 and x != p1:
                tgt = x
                break
        p0 = tgt
    if ((ring >> p1) & 1) != 0:
        base = O_TEL + (k - 1) * 1225 + p1 * 25
        tgt = T[base]
        for j in range(25):
            x = T[base + j]
            if x < 0:
                break
            if ((occ >> x) & 1) == 0 and x != p0:
                tgt = x
                break
        p1 = tgt
    return p0, p1, newb


# ------------------------------------------------------------------------------------------------------------
# the sound shortcut
# ------------------------------------------------------------------------------------------------------------
@njit("i8(i8,i8,i8,i8,i8,i8,i8[::1])", cache=True)
def relaxed_safe(run, cha, bombs, t, d, mode, T):
    """1 = the runner (on cell `run`) is certainly not hit in the next d steps.
    mode 0: whatever the chaser (on `cha`) does: the runner follows a fixed path that only enters cells the chaser
            cannot have reached, and avoids every cell that any chaser bomb (chains included) could blow up then.
    mode 1: against the bombs on the board only, the chaser standing still (a naive dodger's view).
    0 = not proven."""
    if t + d > 300:
        d = 300 - t
    if d <= 0:
        return 1
    if d > 10:
        return 0
    ph = t // 100
    op = T[O_OPEN + ph]
    R = 1 << run
    Dp = 1 << cha  # cells where the chaser can be at the start of step i
    H0 = Dp
    H1 = 0
    H2 = 0
    H3 = 0
    H4 = 0
    H5 = 0
    for i in range(1, d + 1):
        blo = O_BL + ph * 49
        if mode == 0:
            Dc = (Dp | (Dp >> 7) | (Dp << 7) | ((Dp & NC0) >> 1) | ((Dp & NC6) << 1)) & op
        else:
            Dc = Dp
        seed = 0
        alive = 0
        bb = bombs
        while bb != 0:
            f = bb & 1023
            c = f & 63
            fu = (f >> 6) & 7
            if fu == i:
                seed |= T[blo + c]
            if fu >= i:
                alive |= 1 << c
            bb >>= 10
        Rn = R | ((((R >> 7) | (R << 7) | ((R & NC0) >> 1) | ((R & NC6) << 1)) & op) & ~alive & ~Dc)
        A = seed
        if mode == 0 and i >= 5:
            if i == 5:
                m = H0
            elif i == 6:
                m = H1
            elif i == 7:
                m = H2
            elif i == 8:
                m = H3
            elif i == 9:
                m = H4
            else:
                m = H5
            while m != 0:
                b = m & (-m)
                A |= T[blo + DBT[((b * _DB) >> 58) & 63]]
                m ^= b
        if A != 0:
            P = alive
            if mode == 0:
                P |= Dp
            done = 0
            z = A & P
            while z != 0:
                b = z & (-z)
                A |= T[blo + DBT[((b * _DB) >> 58) & 63]]
                done |= b
                z = A & P & ~done
            Rn &= ~A
            if Rn == 0:
                return 0
        R = Rn
        if i == 1:
            H1 = Dc
        elif i == 2:
            H2 = Dc
        elif i == 3:
            H3 = Dc
        elif i == 4:
            H4 = Dc
        elif i == 5:
            H5 = Dc
        Dp = Dc
        tt = t + i
        if tt % 100 == 0 and tt < 300 and i < d:
            k = tt // 100
            ph = k
            op = T[O_OPEN + ph]
            if T[O_SHR + k] != 0:
                ring = T[O_RING + k]
                R &= ~ring
                if R == 0:
                    return 0
                m = Dp & ring
                Dp &= ~ring
                while m != 0:
                    b = m & (-m)
                    z = DBT[((b * _DB) >> 58) & 63]
                    m ^= b
                    base = O_TEL + (k - 1) * 1225 + z * 25
                    for j in range(8):
                        x = T[base + j]
                        if x < 0:
                            break
                        Dp |= 1 << x
                H0 &= ~ring
                H1 &= ~ring
                H2 &= ~ring
                H3 &= ~ring
                H4 &= ~ring
                H5 &= ~ring
                nb = 0
                sh = 0
                bb = bombs
                while bb != 0:
                    f = bb & 1023
                    if ((ring >> (f & 63)) & 1) == 0:
                        nb |= f << sh
                        sh += 10
                    bb >>= 10
                bombs = nb
    return 1


# ------------------------------------------------------------------------------------------------------------
# exact searches
# ------------------------------------------------------------------------------------------------------------
@njit("i8(i8,i8,i8,i8,i8,i8[::1])", cache=True)
def legal_mask(pos, other, occ, cnt, ph, T):
    """Bit a = action a does something different from STAY (bit 0 is always set)."""
    m = 1
    base = O_NBR + ph * 245 + pos * 5
    for a in range(1, 5):
        n = T[base + a]
        if n >= 0 and ((occ >> n) & 1) == 0 and n != other:
            m |= 1 << a
    if cnt < 3 and ((occ >> pos) & 1) == 0:
        m |= 32
    return m


@njit("i8(i8,i8)", cache=True)
def dir_code(frm, to):
    dr = to // 7 - frm // 7
    dc = to % 7 - frm % 7
    code = 4
    if dr < 0:
        code -= 3
    elif dr > 0:
        code += 3
    if dc < 0:
        code -= 1
    elif dc > 0:
        code += 1
    return code


@njit("i8(i8,i8)", cache=True)
def tt_index(bombs, k2):
    x = bombs * 0x1E3779B97F4A7C15 + k2 * 0x3F58476D1CE4E5B9
    x ^= (x >> 31) & 0x1FFFFFFFF
    x *= 0x14D049BB133111EB
    x ^= (x >> 29) & 0x7FFFFFFFF
    return x & TT_MASK


@njit("i8(i8,i8,i8,i8,i8,i8,i8[::1],i8[::1])", cache=True)
def survive(p0, p1, bombs, t, d, r, T, W):
    """1 = player r can make sure it is not hit during the next d steps (its own moves: STAY and the 4 moves)."""
    if t + d > 300:
        d = 300 - t
    if d <= 0:
        return 1
    if r == 0:
        run = p0
        cha = p1
    else:
        run = p1
        cha = p0
    if relaxed_safe(run, cha, bombs, t, d, 0, T) != 0:
        return 1
    k2 = p0 | (p1 << 6) | (d << 12) | (r << 16) | (t << 18) | (W[W_GEN] << 28)
    e = W_TT + 3 * tt_index(bombs, k2)
    if W[e] == bombs and W[e + 1] == k2:
        return W[e + 2]
    W[W_NODES] += 1
    if W[W_NODES] > W[W_LIMIT]:
        W[W_OVER] = 1
        return 0
    ph = t // 100
    occ = 0
    c0 = 0
    c1 = 0
    bb = bombs
    while bb != 0:
        f = bb & 1023
        occ |= 1 << (f & 63)
        if (f >> 9) != 0:
            c1 += 1
        else:
            c0 += 1
        bb >>= 10
    if r == 0:
        lr = legal_mask(run, cha, occ, c0, ph, T)
        lc = legal_mask(cha, run, occ, c1, ph, T)
    else:
        lr = legal_mask(run, cha, occ, c1, ph, T)
        lc = legal_mask(cha, run, occ, c0, ph, T)
    orr = dir_code(run, cha) * 6
    orc = dir_code(cha, run) * 6
    nt = t + 1
    shr = 0
    if nt % 100 == 0 and nt < 300:
        shr = nt // 100
    res = 0
    for ia in range(5):  # the runner: away first, never a bomb
        ar = ORD_AWY[orr + ia]
        if ((lr >> ar) & 1) == 0:
            continue
        ok = 1
        for ib in range(6):  # the chaser: bomb first, then towards the runner
            ac = ORD_TOW[orc + ib]
            if ((lc >> ac) & 1) == 0:
                continue
            if r == 0:
                q0, q1, nb, h0, h1 = step(p0, p1, bombs, ar, ac, ph, T)
                hr = h0
                hc = h1
            else:
                q0, q1, nb, h0, h1 = step(p0, p1, bombs, ac, ar, ph, T)
                hr = h1
                hc = h0
            if hr != 0:
                ok = 0
                break
            if hc != 0:
                continue  # the game is over and the runner has won
            if d == 1:
                continue
            if shr != 0:
                q0, q1, nb = shrink(q0, q1, nb, shr, T)
            if survive(q0, q1, nb, nt, d - 1, r, T, W) == 0:
                ok = 0
                break
        if ok != 0:
            res = 1
            break
    if W[W_OVER] == 0:
        W[e] = bombs
        W[e + 1] = k2
        W[e + 2] = res
    return res


@njit("i8(i8,i8,i8,i8,i8,i8,i8[::1],i8[::1])", cache=True)
def kill(p0, p1, bombs, t, d, k, T, W):
    """1 = player k can force that the other player is hit within d steps while k itself is not hit before or
    at that step (both hit in one step is a draw and does not count)."""
    if t + d > 300:
        d = 300 - t
    if d <= 0:
        return 0
    if k == 0:
        kil = p0
        vic = p1
    else:
        kil = p1
        vic = p0
    if relaxed_safe(vic, kil, bombs, t, d, 0, T) != 0:
        return 0
    k2 = p0 | (p1 << 6) | (d << 12) | (k << 16) | (1 << 17) | (t << 18) | (W[W_GEN] << 28)
    e = W_TT + 3 * tt_index(bombs, k2)
    if W[e] == bombs and W[e + 1] == k2:
        return W[e + 2]
    W[W_NODES] += 1
    if W[W_NODES] > W[W_LIMIT]:
        W[W_OVER] = 1
        return 0
    ph = t // 100
    occ = 0
    c0 = 0
    c1 = 0
    bb = bombs
    while bb != 0:
        f = bb & 1023
        occ |= 1 << (f & 63)
        if (f >> 9) != 0:
            c1 += 1
        else:
            c0 += 1
        bb >>= 10
    if k == 0:
        lk = legal_mask(kil, vic, occ, c0, ph, T)
        lv = legal_mask(vic, kil, occ, c1, ph, T)
    else:
        lk = legal_mask(kil, vic, occ, c1, ph, T)
        lv = legal_mask(vic, kil, occ, c0, ph, T)
    ork = dir_code(kil, vic) * 6
    orv = dir_code(vic, kil) * 6
    nt = t + 1
    shr = 0
    if nt % 100 == 0 and nt < 300:
        shr = nt // 100
    res = 0
    for ia in range(6):  # the killer: bomb first, then towards the victim
        ak = ORD_TOW[ork + ia]
        if ((lk >> ak) & 1) == 0:
            continue
        ok = 1
        for ib in range(6):  # the victim: away first
            av = ORD_AWY[orv + ib]
            if ((lv >> av) & 1) == 0:
                continue
            if k == 0:
                q0, q1, nb, h0, h1 = step(p0, p1, bombs, ak, av, ph, T)
                hk = h0
                hv = h1
            else:
                q0, q1, nb, h0, h1 = step(p0, p1, bombs, av, ak, ph, T)
                hk = h1
                hv = h0
            if hk != 0:
                ok = 0
                break
            if hv != 0:
                continue
            if d == 1:
                ok = 0
                break
            if shr != 0:
                q0, q1, nb = shrink(q0, q1, nb, shr, T)
            if kill(q0, q1, nb, nt, d - 1, k, T, W) == 0:
                ok = 0
                break
        if ok != 0:
            res = 1
            break
    if W[W_OVER] == 0:
        W[e] = bombs
        W[e + 1] = k2
        W[e + 2] = res
    return res


# ------------------------------------------------------------------------------------------------------------
# the root: everything about the 6 x 6 joint actions (index j = my action * 6 + opponent action)
# ------------------------------------------------------------------------------------------------------------
R_VALID = 0  # both actions are real actions (not a blocked move / an impossible bomb)
R_HITM = 1  # I am hit in this step
R_HITO = 2  # the opponent is hit in this step
R_SVM = 3  # afterwards I can guarantee to survive H-1 more steps (1), cannot (0)
R_KILL = 4  # afterwards I have a forced kill: 0 = the opponent is hit now, DP = within DP more steps, 99 = no
R_SVO = 5  # afterwards the opponent can guarantee to survive H-1 more steps (1), cannot (0), -1 = not computed
R_MISC = 6  # [b] = opponent action b looks safe to a naive dodger (bombs on the board now, I stand still)
R_ROOT = 7  # [a] = depth of my forced win with action a (99 = none); [6+a] = my legal mask; [12+b] = the opponent's
R_P0 = 8  # the state after the joint action
R_P1 = 9
R_BOMBS = 10
R_STAT = 11  # [0..2] nodes of the kill / safety / pressure searches, [3..5] their overflow flags
R_DNG = 12  # afterwards the opponent stands in the blast of a bomb on the board
R_ROWS = 13


@njit("void(i8,i8,i8,i8,i8,i8[::1],i8[::1],i8[::1],i8[:,::1])", cache=True)
def root_eval(p0, p1, bombs, t, me, CFG, T, W, OUT):
    H = CFG[0]
    DK = CFG[1]
    DP = CFG[2]
    ph = t // 100
    for i in range(R_ROWS):
        for j in range(36):
            OUT[i, j] = 0
    occ = 0
    c0 = 0
    c1 = 0
    bb = bombs
    while bb != 0:
        f = bb & 1023
        occ |= 1 << (f & 63)
        if (f >> 9) != 0:
            c1 += 1
        else:
            c0 += 1
        bb >>= 10
    if me == 0:
        lm = legal_mask(p0, p1, occ, c0, ph, T)
        lo = legal_mask(p1, p0, occ, c1, ph, T)
    else:
        lm = legal_mask(p1, p0, occ, c1, ph, T)
        lo = legal_mask(p0, p1, occ, c0, ph, T)
    for a in range(6):
        OUT[R_ROOT, a] = 99
        OUT[R_ROOT, 6 + a] = (lm >> a) & 1
        OUT[R_ROOT, 12 + a] = (lo >> a) & 1
    nt = t + 1
    for a in range(6):
        if ((lm >> a) & 1) == 0:
            continue
        for b in range(6):
            if ((lo >> b) & 1) == 0:
                continue
            j = a * 6 + b
            if me == 0:
                q0, q1, nb, h0, h1 = step(p0, p1, bombs, a, b, ph, T)
                hm = h0
                ho = h1
            else:
                q0, q1, nb, h0, h1 = step(p0, p1, bombs, b, a, ph, T)
                hm = h1
                ho = h0
            if hm == 0 and ho == 0 and nt % 100 == 0 and nt < 300:
                q0, q1, nb = shrink(q0, q1, nb, nt // 100, T)
            OUT[R_VALID, j] = 1
            OUT[R_HITM, j] = hm
            OUT[R_HITO, j] = ho
            OUT[R_P0, j] = q0
            OUT[R_P1, j] = q1
            OUT[R_BOMBS, j] = nb
            OUT[R_SVM, j] = 0
            OUT[R_KILL, j] = 99
            OUT[R_SVO, j] = -1
            if hm == 0 and ho == 0 and nt < 300:
                nph = nt // 100
                dz = 0
                bb = nb
                while bb != 0:
                    dz |= T[O_BL + nph * 49 + (bb & 63)]
                    bb >>= 10
                if me == 0:
                    OUT[R_DNG, j] = (dz >> q1) & 1
                else:
                    OUT[R_DNG, j] = (dz >> q0) & 1

    # ---- (b) forced win, shallowest first
    W[W_NODES] = 0
    W[W_LIMIT] = CFG[3]
    W[W_OVER] = 0
    found = 0
    for D in range(1, DK + 1):
        for a in range(6):
            if ((lm >> a) & 1) == 0:
                continue
            ok = 1
            for b in range(6):
                if ((lo >> b) & 1) == 0:
                    continue
                j = a * 6 + b
                if OUT[R_HITM, j] != 0:
                    ok = 0
                    break
                if OUT[R_HITO, j] != 0:
                    continue
                if D == 1:
                    ok = 0
                    break
                if kill(OUT[R_P0, j], OUT[R_P1, j], OUT[R_BOMBS, j], nt, D - 1, me, T, W) == 0:
                    ok = 0
                    break
            if ok != 0:
                OUT[R_ROOT, a] = D
                found = 1
        if found != 0 or W[W_OVER] != 0:
            break
    OUT[R_STAT, 0] = W[W_NODES]
    OUT[R_STAT, 3] = W[W_OVER]

    # ---- (a) safety of every joint action
    W[W_NODES] = 0
    W[W_LIMIT] = CFG[4]
    W[W_OVER] = 0
    for a in range(6):
        if ((lm >> a) & 1) == 0:
            continue
        for b in range(6):
            if ((lo >> b) & 1) == 0:
                continue
            j = a * 6 + b
            if OUT[R_HITM, j] != 0:
                OUT[R_SVM, j] = 0
            elif OUT[R_HITO, j] != 0:
                OUT[R_SVM, j] = 1
            else:
                OUT[R_SVM, j] = survive(OUT[R_P0, j], OUT[R_P1, j], OUT[R_BOMBS, j], nt, H - 1, me, T, W)
    OUT[R_STAT, 1] = W[W_NODES]
    OUT[R_STAT, 4] = W[W_OVER]

    # ---- how a naive dodger sees its own actions (I stand still, only the bombs on the board)
    for b in range(6):
        if ((lo >> b) & 1) == 0:
            continue
        j = b
        if OUT[R_HITO, j] != 0:
            OUT[R_MISC, b] = 0
        elif OUT[R_HITM, j] != 0:
            OUT[R_MISC, b] = 1
        elif me == 0:
            OUT[R_MISC, b] = relaxed_safe(OUT[R_P1, j], OUT[R_P0, j], OUT[R_BOMBS, j], nt, 5, 1, T)
        else:
            OUT[R_MISC, b] = relaxed_safe(OUT[R_P0, j], OUT[R_P1, j], OUT[R_BOMBS, j], nt, 5, 1, T)

    # ---- (c) pressure: what each opponent reply leads to, for my safe actions
    if found == 0:
        W[W_NODES] = 0
        W[W_LIMIT] = CFG[5]
        W[W_OVER] = 0
        for a in range(6):
            if ((lm >> a) & 1) == 0:
                continue
            safe = 1
            for b in range(6):
                if ((lo >> b) & 1) != 0 and OUT[R_SVM, a * 6 + b] == 0:
                    safe = 0
                    break
            if safe == 0:
                continue
            for b in range(6):
                if ((lo >> b) & 1) == 0:
                    continue
                j = a * 6 + b
                if OUT[R_HITO, j] != 0:
                    OUT[R_KILL, j] = 0
                    OUT[R_SVO, j] = 0
                    continue
                if kill(OUT[R_P0, j], OUT[R_P1, j], OUT[R_BOMBS, j], nt, DP, me, T, W) != 0:
                    OUT[R_KILL, j] = DP
                    OUT[R_SVO, j] = 0
                else:
                    OUT[R_SVO, j] = survive(OUT[R_P0, j], OUT[R_P1, j], OUT[R_BOMBS, j], nt, H - 1, 1 - me, T, W)
        OUT[R_STAT, 2] = W[W_NODES]
        OUT[R_STAT, 5] = W[W_OVER]


@njit("void(i8,i8,i8[::1],i8[::1])", cache=True)
def bfs_dist(srcmask, ph, T, out):
    """out[cell] = steps from the nearest cell of srcmask through open cells (bombs and players ignored), -1 = unreachable."""
    for i in range(49):
        out[i] = -1
    op = T[O_OPEN + ph]
    seen = srcmask & op
    front = seen
    dist = 0
    while front != 0:
        m = front
        while m != 0:
            b = m & (-m)
            out[DBT[((b * _DB) >> 58) & 63]] = dist
            m ^= b
        nxt = ((front >> 7) | (front << 7) | ((front & NC0) >> 1) | ((front & NC6) << 1)) & op & ~seen
        seen |= nxt
        front = nxt
        dist += 1


# ------------------------------------------------------------------------------------------------------------
# tables (plain Python, once per map and phase)
# ------------------------------------------------------------------------------------------------------------
def build_tables(walls9, phase):
    """walls9: the 9x9 bool array of an observation taken in phase `phase` (= step // 100)."""
    T = np.zeros(T_SIZE, np.int64)
    T[O_TEL:] = -1
    cur = np.array(walls9, dtype=bool).copy()
    deltas = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    def fill(ph, w):
        op = 0
        for r in range(1, 8):
            for c in range(1, 8):
                cell = (r - 1) * 7 + (c - 1)
                base = O_NBR + ph * 245 + cell * 5
                if w[r, c]:
                    T[base:base + 5] = -1
                    continue
                op |= 1 << cell
                bl = 1 << cell
                for a, (dr, dc) in enumerate(deltas):
                    rr, cc = r + dr, c + dc
                    T[base + a] = -1 if w[rr, cc] else (rr - 1) * 7 + (cc - 1)
                    if a:
                        for k in (1, 2):
                            rr, cc = r + dr * k, c + dc * k
                            if w[rr, cc]:
                                break
                            bl |= 1 << ((rr - 1) * 7 + (cc - 1))
                T[O_BL + ph * 49 + cell] = bl
        T[O_OPEN + ph] = op

    for ph in range(phase + 1):
        fill(ph, cur)
    for k in range(phase + 1, 3):
        ring = [(r, c) for r in range(9) for c in range(9) if min(r, c, 8 - r, 8 - c) == k]
        inner = [(r, c) for r in range(9) for c in range(9) if min(r, c, 8 - r, 8 - c) > k and not cur[r, c]]
        T[O_RING + k] = sum(1 << ((r - 1) * 7 + (c - 1)) for r, c in ring)
        if inner:
            T[O_SHR + k] = 1
            for (r, c) in ring:
                if cur[r, c]:
                    continue
                order = sorted(inner, key=lambda x: (abs(x[0] - r) + abs(x[1] - c), x))
                base = O_TEL + (k - 1) * 1225 + ((r - 1) * 7 + (c - 1)) * 25
                for j, x in enumerate(order[:25]):
                    T[base + j] = (x[0] - 1) * 7 + (x[1] - 1)
            cur = cur.copy()
            for cell in ring:
                cur[cell] = True
        fill(k, cur)
    return T


def new_work():
    return np.zeros(W_SIZE, np.int64)


def pack_bombs(bombs):
    """[(cell, fuse, owner)] -> the packed int."""
    v = 0
    for i, (c, f, o) in enumerate(sorted(bombs)):
        v |= (c | (f << 6) | (o << 9)) << (10 * i)
    return v


def unpack_bombs(v):
    out = []
    while v:
        f = v & 1023
        out.append((f & 63, (f >> 6) & 7, f >> 9))
        v >>= 10
    return out


def _selftest():
    for i in range(49):
        b = np.int64(1 << i)
        assert int(_bit_index(b)) == i, "bit index self-test failed"


@njit("i8(i8)", cache=True)
def _bit_index(b):
    return DBT[((b * _DB) >> 58) & 63]


_selftest()
