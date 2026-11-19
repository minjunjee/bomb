"""test_camper: a patient camper (held-out TEST opponent, rule-based, never used for training).

Style
-----
* Claims the roomiest, best connected cell that survives the shrinks (near the centre) and waits there.
* Rarely chases. When the opponent comes within reach it lays a bomb as a shield (the bomb blocks the way and its
  blast denies the approach), often a second one inside the first blast (a chain: both go off together), and steps
  back around a corner to the best cell outside every blast.
* Very hard to kill: every action is checked with an exact, chain- and shrink-aware escape search against every
  reply of the opponent in this step (its bomb included, its body blocking the way), plus "it steps and then bombs".
* Punishes: takes every forced kill (my bomb, or just holding my cell so that the opponent cannot get out of a
  blast), and prefers bombs that leave the opponent few surviving replies.

Timing convention: "time k" = the end of the k-th step from now (the step being decided is k = 1). A bomb whose obs
fuse is f explodes at time f; a bomb dropped now explodes at time 5. The board is a bitmask (cell = row * 9 + col).
"""

import random

import numpy as np

_DEBUG = False

_RING = [0] * 5  # all cells of ring k (0 = border)
_INSIDE = [0] * 5  # all cells of the rings > k
for _r in range(9):
    for _c in range(9):
        _RING[min(_r, _c, 8 - _r, 8 - _c)] |= 1 << (_r * 9 + _c)
for _k in range(5):
    for _j in range(_k + 1, 5):
        _INSIDE[_k] |= _RING[_j]

_MOVES = ((1, -9), (2, 9), (3, -1), (4, 1))
_DIRS = (-9, 9, -1, 1)
_CAP = 6
_FUSE = 5
_MAXB = 3


class Policy:
    def __init__(self):
        self.wkey = None
        self.phase = -1
        self._tkey = None
        self._tval = 0

    # ------------------------------------------------------------------------------------------------------------
    def act(self, obs):
        try:
            return int(self._act(obs))
        except Exception:  # noqa: BLE001  (never raise in a game)
            if _DEBUG:
                raise
            return 0

    # ------------------------------------------------------------------------------------------------------------
    # static tables for one wall layout (rebuilt after every shrink)
    # ------------------------------------------------------------------------------------------------------------
    def _roomy(self, G):
        cells = [i for i in self.cells if (G >> i) & 1]
        nb = {i: [i + d for d in _DIRS if (G >> (i + d)) & 1] for i in cells}
        deg = {i: len(nb[i]) for i in cells}
        core = set(cells)  # the 2-core: cells on loops or between loops (no dead ends)
        stack = [i for i in cells if deg[i] <= 1]
        while stack:
            i = stack.pop()
            if i not in core:
                continue
            core.discard(i)
            for j in nb[i]:
                if j in core:
                    deg[j] -= 1
                    if deg[j] <= 1:
                        stack.append(j)
        comp = {}
        for i in cells:
            if i in comp:
                continue
            seen, q = {i}, [i]
            while q:
                x = q.pop()
                for y in nb[x]:
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
            for x in seen:
                comp[x] = len(seen)
        out = {}
        for i in cells:
            ball = {i}
            for j in nb[i]:
                ball.add(j)
                ball.update(nb[j])
            out[i] = len(nb[i]) + 0.5 * len(ball) + (2.0 if i in core else 0.0) + 0.25 * min(comp[i], 12)
        return out

    def _setup(self, walls, key, phase):
        self.wkey, self.phase = key, phase
        self._tkey = None
        openm, cells = 0, []
        for r in range(1, 8):
            row = walls[r]
            for c in range(1, 8):
                if not row[c]:
                    i = r * 9 + c
                    openm |= 1 << i
                    cells.append(i)
        self.openm, self.cells = openm, cells
        BL = [0] * 81
        NB = [()] * 81
        for i in cells:
            m = 1 << i
            nb = []
            for d in _DIRS:
                j = i + d
                if (openm >> j) & 1:
                    nb.append(j)
                    m |= 1 << j
                    j += d
                    if (openm >> j) & 1:
                        m |= 1 << j
            BL[i] = m
            NB[i] = tuple(nb)
        self.BL, self.NB = BL, NB
        D = [None] * 81
        far = [99] * 81
        for i in range(81):
            D[i] = far
        for i in cells:
            dist = [99] * 81
            dist[i] = 0
            q = [i]
            for x in q:
                dx = dist[x] + 1
                for y in NB[x]:
                    if dist[y] > dx:
                        dist[y] = dx
                        q.append(y)
            D[i] = dist
        self.D = D
        # the next shrink
        self.will_shrink = False
        self.ringm, self.innerm, self.tele = 0, openm, {}
        if phase < 2:
            k = phase + 1
            inner = openm & _INSIDE[k]
            if inner:
                self.will_shrink = True
                self.ringm, self.innerm = openm & _RING[k], inner
                ins = [i for i in cells if (inner >> i) & 1]
                for i in cells:
                    if (self.ringm >> i) & 1:
                        r0, c0 = divmod(i, 9)
                        self.tele[i] = min(ins, key=lambda x: (abs(x // 9 - r0) + abs(x % 9 - c0), x))
        # camp value: roomy and well connected now and after the shrinks
        g0 = openm
        g1 = self.innerm
        g2 = g1
        if phase == 0:
            x = g1 & _INSIDE[2]
            if x:
                g2 = x
        V = [0.0] * 81
        for i, v in self._roomy(g0).items():
            V[i] = v
        if g1 != g0:
            for i, v in self._roomy(g1).items():
                V[i] += 4.0 + v
        if g2 != g1:
            for i, v in self._roomy(g2).items():
                V[i] += 4.0 + v
        self.V = V

    # ------------------------------------------------------------------------------------------------------------
    # exact bomb timing
    # ------------------------------------------------------------------------------------------------------------
    def _timeline(self, bl):
        """bl: [(cell, own time)] -> (last explosion time, hits[k] = cells exploding at time k,
        pres[k] = bombs in the way during the move of step k, union of all hits). Chains and the shrink included."""
        hits = [0] * 9
        pres = [0] * 9
        n = len(bl)
        if n == 0:
            return (0, hits, pres, 0)
        BL = self.BL
        ks = self.ks
        ringm = self.ringm
        innerm = self.innerm
        cells = [x[0] for x in bl]
        times = [x[1] for x in bl]
        if n > 1:
            ch = True
            while ch:
                ch = False
                for i in range(n):
                    ti = times[i]
                    ci = cells[i]
                    if ti > ks:
                        if (ringm >> ci) & 1:
                            continue  # it vanishes with the ring before it explodes
                        b = BL[ci] & innerm
                    else:
                        b = BL[ci]
                    for j in range(n):
                        if times[j] > ti and (b >> cells[j]) & 1:
                            times[j] = ti
                            ch = True
        kend = self.kend
        L = 0
        U = 0
        for i in range(n):
            ti = times[i]
            ci = cells[i]
            bit = 1 << ci
            if ti > ks and (ringm >> ci) & 1:
                last = ks
            else:
                last = ti
                if ti <= kend:
                    b = BL[ci] if ti <= ks else (BL[ci] & innerm)
                    hits[ti] |= b
                    U |= b
                    if ti > L:
                        L = ti
            for k in range(1, last + 1):
                pres[k] |= bit
        return (L, hits, pres, U)

    def _tl(self, mb, ob):
        k = mb * 2 + ob
        T = self._tlc[k]
        if T is None:
            bl = self._EB
            if mb:
                bl = bl + [(self._me, _FUSE)]
            if ob:
                bl = bl + [(self._opp, _FUSE)]
            T = self._tlc[k] = self._timeline(bl)
        return T

    def _teleport(self, f):
        r = f & self.ringm
        if r:
            f &= ~r
            tele = self.tele
            while r:
                low = r & -r
                f |= 1 << tele[low.bit_length() - 1]
                r ^= low
        return f

    def _surv(self, start, T, blk, blk2):
        """A player standing on `start` at time 1: the cells where it can be alive at the horizon (0 = it dies).
        blk: a cell that is never free (the other player standing there; the game ends when that one is hit),
        blk2: a cell that is not free in step 2 only."""
        L, hits, pres, _ = T
        f = 1 << start
        if f & hits[1]:
            return 0
        ks = self.ks
        if blk:
            for k in range(1, L + 1):
                if hits[k] & blk:
                    L = k
                    break
        if ks == 1:
            f = self._teleport(f)
        if L < 2:
            return f
        openm = self.openm
        nblk = ~blk
        fr = (self.innerm if ks < 2 else openm) & ~pres[2] & nblk & ~blk2
        f = (f | (((f << 9) | (f >> 9) | (f << 1) | (f >> 1)) & fr)) & ~hits[2]
        if not f:
            return 0
        if ks == 2:
            f = self._teleport(f)
        for k in range(3, L + 1):
            fr = (self.innerm if k > ks else openm) & ~pres[k] & nblk
            f = (f | (((f << 9) | (f >> 9) | (f << 1) | (f >> 1)) & fr)) & ~hits[k]
            if not f:
                return 0
            if k == ks:
                f = self._teleport(f)
        return f

    def _surv2(self, mp, op, T, hold2):
        """Strict version: I stand on `mp` and the opponent on `op` at time 1. The cells where I can be alive at the
        horizon even if the opponent does anything it can to get in my way: at every step I may only enter cells
        the opponent can neither stand on nor walk to in that step (two players targeting one cell both stay put).
        Staying where I am is always possible. The opponent is gone once it is hit. hold2: it does not move in
        step 2 (it is dropping a bomb)."""
        L, hits, pres, _ = T
        f = 1 << mp
        g = 1 << op
        h = hits[1]
        if f & h:
            return 0
        if g & h:
            return f
        ks = self.ks
        if ks == 1:
            f = self._teleport(f)
            g = self._teleport(g)
        openm = self.openm
        for k in range(2, L + 1):
            fr = (self.innerm if k > ks else openm) & ~pres[k]
            if k == 2 and hold2:
                gt = g
            else:
                gt = g | (((g << 9) | (g >> 9) | (g << 1) | (g >> 1)) & fr)
            h = hits[k]
            f = (f | (((f << 9) | (f >> 9) | (f << 1) | (f >> 1)) & fr & ~gt)) & ~h
            if not f:
                return 0
            g = gt & ~h
            if not g:
                return f
            if k == ks:
                f = self._teleport(f)
                g = self._teleport(g)
        return f

    # ------------------------------------------------------------------------------------------------------------
    def _target(self, me, opp, dang, ks):
        """The camp cell to walk to: roomy, not too far, not next to the opponent, outside every blast."""
        key = (me, opp, dang, ks <= 10)
        if key == self._tkey:
            return self._tval
        V = self.V
        Dm = self.D[me]
        Do = self.D[opp]
        BLo = self.BL[opp]
        ringm = self.ringm if ks <= 10 else 0
        best, bv = me, -1e9
        for c in self.cells:
            d = Dm[c]
            if d > 60:
                continue
            v = V[c] - 0.3 * d
            do = Do[c]
            if do < 3:
                v -= (3 - do) * 1.5
                if (BLo >> c) & 1:
                    v -= 1.0
            if (dang >> c) & 1:
                v -= 5.0
            if (ringm >> c) & 1:
                v -= 10.0
            if v > bv:
                bv, best = v, c
        self._tkey, self._tval = key, best
        return best

    # ------------------------------------------------------------------------------------------------------------
    def _act(self, obs):
        walls = np.asarray(obs["walls"], dtype=bool)
        t = obs["step"]
        phase = 2 if t >= 200 else (1 if t >= 100 else 0)
        key = walls.tobytes()
        if key != self.wkey or phase != self.phase:
            self._setup(walls, key, phase)
        p = obs["me"]
        me = p[0] * 9 + p[1]
        p = obs["opp"]
        opp = p[0] * 9 + p[1]
        player = obs["player"]
        EB = []
        bombm = mine = theirs = 0
        for b in obs["bombs"]:
            q = b["pos"]
            i = q[0] * 9 + q[1]
            EB.append((i, b["fuse"]))
            bombm |= 1 << i
            if b["owner"] == player:
                mine += 1
            else:
                theirs += 1
        ks = self.ks = (100 - t % 100) if self.will_shrink else 99
        self.kend = 300 - t
        openm, BL, D = self.openm, self.BL, self.D
        dmo = D[me][opp]
        self._EB, self._me, self._opp = EB, me, opp
        tlc = self._tlc = [None] * 4
        T0 = self._tl(0, 0)
        dang = T0[3]
        soon = T0[1][1] | T0[1][2]

        # what I can do / what the opponent can do in this step: (action, target cell, drops a bomb)
        mcands = [(0, me, 0)]
        for a, d in _MOVES:
            n = me + d
            if (openm >> n) & 1 and not (bombm >> n) & 1 and n != opp:
                mcands.append((a, n, 0))
        if mine < _MAXB and not (bombm >> me) & 1:
            mcands.append((5, me, 1))
        near1 = dmo <= 5 or (bool(EB) and dmo <= 8)
        ocands = [(opp, 0)]
        if near1:
            for d in _DIRS:
                n = opp + d
                if (openm >> n) & 1 and not (bombm >> n) & 1 and n != me:
                    ocands.append((n, 0))
            if theirs < _MAXB and not (bombm >> opp) & 1:
                ocands.append((opp, 1))
        nrep = len(ocands)

        # one step of maximin: my action against every reply, judged by exact escape searches
        info = []
        surv = self._surv
        surv2 = self._surv2
        for a, tm, mb in mcands:
            n_loss = n_draw = n_kill = wloss = n_soft = 0
            margin = _CAP
            kc = near1 and (mb or bool(EB))
            for to, ob in ocands:
                if tm == to:  # both walk into the same cell: nobody moves
                    mp, op = me, opp
                else:
                    mp, op = tm, to
                T = tlc[mb * 2 + ob] or self._tl(mb, ob)
                U = T[3]
                if (U >> mp) & 1:
                    fm = surv2(mp, op, T, 0)
                    if fm:
                        c = fm.bit_count()
                        if c < margin:
                            margin = c
                    else:  # not safe against a blocking opponent; safe if it just stands there?
                        n_soft += 3 if ob else 1
                        margin = 0
                        fm = surv(mp, T, 1 << op, 0)
                else:
                    fm = 1
                if fm:
                    if kc and (U >> op) & 1:
                        if not surv(op, T, 0, 1 << mp):
                            n_kill += 1  # it cannot get out whatever it does
                        elif not (U >> mp) & 1 and not surv(op, T, 1 << mp, 0):
                            n_kill += 1  # it cannot get out as long as I hold my (safe) cell
                elif (U >> op) & 1 and not surv(op, T, 0, 1 << mp):
                    n_draw += 1
                else:
                    n_loss += 1
                    wloss += 3 if ob else 1
            if n_loss == 0:
                tier = 2 if n_draw else (3 if n_soft else 4)
            else:
                tier = 1 if n_loss < nrep else 0
            info.append([a, tm, mb, tier, wloss, n_draw, n_kill, margin, 0, n_soft])

        best_tier = max(e[3] for e in info)
        group = [e for e in info if e[3] == best_tier]
        if best_tier < 4:
            k = 9 if best_tier == 3 else (5 if best_tier == 2 else 4)
            m = min(e[k] for e in group)
            group = [e for e in group if e[k] == m]
        if best_tier >= 3:
            if near1:
                forced = [e for e in group if e[6] == nrep]
                if forced:
                    group = forced
            # two steps ahead: the opponent steps (or waits) and then drops a bomb; can I still get away?
            if dmo <= 4 and theirs < _MAXB and len(group) > 1:
                cache = {}
                for e in group:
                    tm, mb = e[1], e[2]
                    bad = 0
                    for to, ob in ocands:
                        if ob:
                            continue
                        if tm == to:
                            mp, op = me, opp
                        else:
                            mp, op = tm, to
                        if (bombm >> op) & 1:
                            continue
                        k = (mb, op)
                        T = cache.get(k)
                        if T is None:
                            bl = EB + [(op, _FUSE + 1)]
                            if mb:
                                bl.append((me, _FUSE))
                            T = cache[k] = self._timeline(bl)
                        if not (T[3] >> mp) & 1 or surv2(mp, op, T, 1):
                            continue
                        if surv(op, T, 0, 0):  # and it gets away from its own bomb: a real threat
                            bad += 1 if surv(mp, T, 1 << op, 0) else 4
                    e[8] = bad
                m = min(e[8] for e in group)
                group = [e for e in group if e[8] == m]
        if len(group) == 1:
            return group[0][0]

        # the camper's taste among the safe actions
        tgt = self._target(me, opp, dang, ks)
        Dt = D[tgt]
        Dop = D[opp]
        BLo = BL[opp]
        near = dmo <= 4
        ringm = self.ringm if ks <= 8 else 0
        inblast = (BL[me] >> opp) & 1
        best, bs = 0, -1e9
        for a, tm, mb, tier, wloss, n_draw, n_kill, margin, l2, n_soft in group:
            s = random.random() * 0.2 - Dt[tm]
            if (dang >> tm) & 1:
                s -= 3.0 if (soon >> tm) & 1 else 2.0
            if (ringm >> tm) & 1:
                s -= 6.0
            if near:
                dd = Dop[tm]
                if dd < 3:
                    s -= (3 - dd) * 0.8
                if (BLo >> tm) & 1:
                    s -= 1.0
                s += 0.4 * margin
            if n_kill:
                s += 2.0 * n_kill / nrep
            if mb:
                if n_kill == nrep and near1:
                    s += 50.0
                elif not (inblast or dmo <= 3):
                    s -= 100.0
                elif margin < 2 or tier < 4 or l2:
                    s -= 100.0
                elif mine >= 2 and not n_kill:
                    s -= 100.0  # keep one bomb for a kill
                elif (dang >> me) & 1:  # inside a blast: this bomb goes off with that one (a chain)
                    if (BL[me] & ~dang).bit_count() < 2:
                        s -= 100.0
                    else:
                        s += 3.0
                else:
                    s += 2.5 + (1.0 if inblast else 0.0)
            if s > bs:
                bs, best = s, a
        return best
