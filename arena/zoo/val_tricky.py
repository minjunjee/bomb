"""val_tricky -- a trap-building, ambushing opponent (validation only, never a training opponent).

It does not play the usual "walk up to the opponent and bomb when it is in the blast" game:

* Every decision is checked with an exact, chain-aware timing search on bitboards (one int = one set of cells):
  which cells can a player still be alive on, step by step, with the bombs on the board (bombs block the way until
  they have exploded, a bomb in an earlier blast goes off with it, the sudden-death shrink removes bombs on the ring
  and teleports players).
* Kills: it drops a bomb only when EVERY reply of the opponent ends inside an explosion (dead ends closed by the bomb
  itself, a second bomb inside the blast of an older one that therefore goes off early, the landing cell of a
  teleport at the shrink), or it simply stands still in the only way out of somebody else's blast (body block).
* One step ahead it counts for how many opponent replies such a kill exists next turn (weighted by how the opponent
  has moved so far) and walks to the cell with the best chances: this is what makes it step back around a corner
  and wait there.
* Unusual bombs: at range to cut the cells the opponent would run to, "seed" bombs it keeps close to so that the
  next bomb chains, bombs that explode right after the shrink on the cell the opponent is teleported to.
* Three temperaments drawn per game (global `random`): lurker (waits at a post near the centre, sorties when nothing
  happens), cutter (hovers at distance 2 and bombs at range) and feinter (comes close, runs back to its post).
* Safety first: an action is used only if it survives the bombs on the board, also when the opponent drops a bomb
  right now, and (when the opponent is near) also when the opponent steps next to it and bombs or just stands in
  the way next turn.

Timing convention: "time k" = the end of the k-th step from now (k = 1 is the step being decided).
A bomb whose obs fuse is f explodes at time f, a bomb dropped now at time RULES.fuse.
"""

import random

from bombcompetition import RULES

_F = RULES.fuse
_R = RULES.radius
_MB = RULES.max_bombs
_SE = RULES.shrink_every or 0
_MS = RULES.max_steps
_N = 9
_DIRS = ((1, -_N), (2, _N), (3, -1), (4, 1))  # (action, change of the cell index)
_DEBUG = False


def _bits(x):
    out = []
    while x:
        b = x & -x
        out.append(b.bit_length() - 1)
        x ^= b
    return out


# ----------------------------------------------------------------------------------------------------------------
# static data of one wall layout (a game has at most three: the start map and the two shrunken ones)
# ----------------------------------------------------------------------------------------------------------------
class _World:
    def __init__(self, walls):
        self.walls = walls.copy()
        wl = walls.tolist()
        op = 0
        cells = []
        for r in range(_N):
            for c in range(_N):
                if not wl[r][c]:
                    op |= 1 << (r * _N + c)
                    cells.append(r * _N + c)
        self.open = op
        self.cells = cells
        blast = [0] * (_N * _N)
        nb = [()] * (_N * _N)
        nbm = [0] * (_N * _N)
        ring = [0] * (_N * _N)
        for i in cells:
            r, c = divmod(i, _N)
            ring[i] = min(r, c, _N - 1 - r, _N - 1 - c)
            m = 1 << i
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                for k in range(1, _R + 1):
                    rr, cc = r + dr * k, c + dc * k
                    if rr < 0 or rr >= _N or cc < 0 or cc >= _N or wl[rr][cc]:
                        break
                    m |= 1 << (rr * _N + cc)
            blast[i] = m
            ns = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < _N and 0 <= cc < _N and not wl[rr][cc]:
                    ns.append(rr * _N + cc)
            nb[i] = tuple(ns)
            x = 0
            for j in ns:
                x |= 1 << j
            nbm[i] = x
        self.blast, self.nb, self.nbm, self.ring = blast, nb, nbm, ring
        self._dist = {}
        self._shrink = {}
        self.kz = None
        self.vul = None
        self._posts = None

    def dist(self, s):
        """Walking distances from s (walls only), 99 = cannot get there."""
        d = self._dist.get(s)
        if d is None:
            d = [99] * (_N * _N)
            d[s] = 0
            q = [s]
            nb = self.nb
            for i in q:
                di = d[i] + 1
                for j in nb[i]:
                    if d[j] == 99:
                        d[j] = di
                        q.append(j)
            self._dist[s] = d
        return d

    def shrink(self, k):
        """Sudden death of ring k: (ring, open cells inside, the world after it, teleport targets, steps to get
        inside) or False when the map does not shrink."""
        s = self._shrink.get(k)
        if s is None:
            ring = inner = 0
            for i in range(_N * _N):
                r, c = divmod(i, _N)
                ro = min(r, c, _N - 1 - r, _N - 1 - c)
                if ro == k:
                    ring |= 1 << i
                elif ro > k and (self.open >> i) & 1:
                    inner |= 1 << i
            if not inner:
                s = False
            else:
                w2 = self.walls.copy()
                for i in _bits(ring):
                    w2[divmod(i, _N)] = True
                inner_cells = _bits(inner)
                tele = {}
                for i in _bits(ring & self.open):
                    r0, c0 = divmod(i, _N)
                    tele[i] = sorted(inner_cells, key=lambda x: (abs(x // _N - r0) + abs(x % _N - c0), x))[:4]
                din = [99] * (_N * _N)
                q = list(inner_cells)
                for i in q:
                    din[i] = 0
                for i in q:
                    for j in self.nb[i]:
                        if din[j] == 99:
                            din[j] = din[i] + 1
                            q.append(j)
                s = (ring, inner, _world(w2), tele, din)
            self._shrink[k] = s
        return s

    def killzones(self):
        """kz[x]: the cells on which a player is lost when a single bomb is dropped on x (it cannot get out of the
        blast in time, the bomb itself closing the way). vul[c]: the cells x with c in kz[x]."""
        if self.kz is None:
            kz = [0] * (_N * _N)
            vul = [0] * (_N * _N)
            nbm, op = self.nbm, self.open
            for x in self.cells:
                bl = self.blast[x]
                bx = 1 << x
                free = op & ~bx
                z = 0
                for c in _bits(bl & ~bx):
                    f = (1 << c) | (nbm[c] & ~bx)
                    for _ in range(_F - 2):
                        f = (f | (f << 1) | (f >> 1) | (f << _N) | (f >> _N)) & free
                    f = (f | (f << 1) | (f >> 1) | (f << _N) | (f >> _N)) & free & ~bl
                    if not f:
                        z |= 1 << c
                        vul[c] |= bx
                kz[x] = z
            self.kz, self.vul = kz, vul
        return self.kz

    def posts(self):
        """Good cells to wait on: near the centre, at least two ways out, not a cell a single bomb can close."""
        if self._posts is None:
            self.killzones()
            sc = []
            for c in self.cells:
                s = 2.0 * self.ring[c] + 0.8 * len(self.nb[c]) - (4.0 if self.vul[c] else 0.0)
                if len(self.nb[c]) < 2:
                    s -= 4.0
                sc.append((s, c))
            sc.sort(reverse=True)
            self._posts = [c for _, c in sc[:5]]
        return self._posts


_WORLDS = {}


def _world(walls):
    key = walls.tobytes()
    w = _WORLDS.get(key)
    if w is None:
        if len(_WORLDS) > 120:
            _WORLDS.clear()
        w = _WORLDS[key] = _World(walls)
    return w


# ----------------------------------------------------------------------------------------------------------------
# bombs -> when and where it explodes
# ----------------------------------------------------------------------------------------------------------------
def _chain(bombs, blast):
    """{cell: fuse} -> {cell: time it really explodes} (a bomb inside an earlier blast goes off with it)."""
    t = dict(bombs)
    again = True
    while again:
        again = False
        for i in t:
            ti = t[i]
            bl = blast[i]
            for j in t:
                if t[j] > ti and (bl >> j) & 1:
                    t[j] = ti
                    again = True
    return t


def _scen(W, bombs, ks, S):
    """bombs {cell: fuse} -> (T, hit, alive, ks, S, later, times):
    hit[k] = cells exploding at time k, alive[k] = bombs still lying there during the moves of step k.
    ks/S: the shrink happens at time ks (0: not before the last explosion); then bombs on the ring vanish and the
    bombs left explode with the new walls (`later` = their cells)."""
    if not bombs:
        return None
    blast = W.blast
    t = _chain(bombs, blast) if len(bombs) > 1 else bombs
    T = 0
    for v in t.values():
        if v > T:
            T = v
    if ks and ks < T:
        T = max(bombs.values())
        hit = [0] * (T + 2)
        alive = [0] * (T + 2)
        ring = S[0]
        later = None
        for i, ti in t.items():
            b = 1 << i
            if ti <= ks:
                hit[ti] |= blast[i]
                for k in range(1, ti + 1):
                    alive[k] |= b
            elif ring & b:
                for k in range(1, ks + 1):
                    alive[k] |= b
            else:
                if later is None:
                    later = {}
                later[i] = bombs[i]
        lm = 0
        if later:
            b2 = S[2].blast
            t2 = _chain(later, b2) if len(later) > 1 else later
            for i, ti in t2.items():
                b = 1 << i
                lm |= b
                hit[ti] |= b2[i]
                for k in range(1, ti + 1):
                    alive[k] |= b
        return (T, hit, alive, ks, S, lm, t)
    hit = [0] * (T + 2)
    alive = [0] * (T + 2)
    for i, ti in t.items():
        hit[ti] |= blast[i]
        b = 1 << i
        for k in range(1, ti + 1):
            alive[k] |= b
    return (T, hit, alive, 0, None, 0, t)


def _remap(f, S, avoid, wide):
    """The shrink: players on the ring are put on the nearest open cell inside (bombs are avoided).
    wide: also the second choice (the other player may stand on the first)."""
    g = f & S[1]
    r = f & S[0]
    tele = S[3]
    while r:
        b = r & -r
        r ^= b
        order = tele.get(b.bit_length() - 1)
        if not order:
            continue
        n = 0
        for x in order:
            if not (avoid >> x) & 1:
                g |= 1 << x
                n += 1
                if n == 2 or not wide:
                    break
        if not n:
            g |= 1 << order[0]
    return g


def _surv(sc, f, free, wide=False):
    """f = the cells a player may stand on after the first step. Returns the cells it can still be alive on after
    the last explosion (0 = it is lost). free = the cells it may enter (walls and blocked cells removed)."""
    T, hit, alive, ks, S, lm, _ = sc
    f &= ~hit[1]
    if ks:
        if ks == 1:
            f = _remap(f, S, lm, wide)
            free &= S[1]
        for k in range(2, T + 1):
            if not f:
                return 0
            al = alive[k]
            g = (f | (f << 1) | (f >> 1) | (f << _N) | (f >> _N)) & free & ~al
            f = (g | (f & al)) & ~hit[k]
            if k == ks:
                f = _remap(f, S, lm, wide)
                free &= S[1]
        return f
    for k in range(2, T + 1):
        if not f:
            return 0
        al = alive[k]
        g = (f | (f << 1) | (f >> 1) | (f << _N) | (f >> _N)) & free & ~al
        f = (g | (f & al)) & ~hit[k]
    return f


def _surv_par(sc, f, open_, reach):
    """Like _surv, but without ever entering a cell the opponent could stand on or step on at the same time
    (reach = the cells it can be on after the first step; it is stopped by bombs like everybody). Shrink ignored."""
    T, hit, alive = sc[0], sc[1], sc[2]
    f &= ~hit[1]
    for k in range(2, T + 1):
        if not f:
            return 0
        al = alive[k]
        reach |= ((reach << 1) | (reach >> 1) | (reach << _N) | (reach >> _N)) & open_ & ~al
        g = ((f << 1) | (f >> 1) | (f << _N) | (f >> _N)) & open_ & ~al & ~reach
        f = (g | f) & ~hit[k]
    return f


def _terr(a, b, free):
    """Cells a player starting on a reaches strictly before one starting on b (both sets of start cells)."""
    seen = a | b
    mine = a & ~b
    fa, fb = mine, b
    while fa:
        na = ((fa << 1) | (fa >> 1) | (fa << _N) | (fa >> _N)) & free & ~seen
        nb = ((fb << 1) | (fb >> 1) | (fb << _N) | (fb >> _N)) & free & ~seen
        na &= ~nb
        mine |= na
        seen |= na | nb
        fa, fb = na, nb
    return mine


def _life(sc, f, free):
    """How many steps a player starting on f stays alive (shrink ignored)."""
    T, hit, alive = sc[0], sc[1], sc[2]
    f &= ~hit[1]
    if not f:
        return 0
    for k in range(2, T + 1):
        al = alive[k]
        g = (f | (f << 1) | (f >> 1) | (f << _N) | (f >> _N)) & free & ~al
        f = (g | (f & al)) & ~hit[k]
        if not f:
            return k - 1
    return T


def _never(sc, b):
    for h in sc[1]:
        if h & b:
            return False
    return True


# ----------------------------------------------------------------------------------------------------------------
class Policy:
    def __init__(self):
        r = random.random()
        self.style = 0 if r < 0.34 else (1 if r < 0.67 else 2)  # 0 lurker, 1 cutter, 2 feinter
        self.direct_p = 0.55 + 0.4 * random.random()  # bomb when the opponent stands in the blast
        self.cut_p = 0.25 + 0.4 * random.random()  # bomb at range on the opponent's ways
        self.seed_p = 0.15 + 0.3 * random.random()  # a bomb to chain from later
        self.lead = random.randint(9, 15)  # go inside this many steps before a shrink
        self.patience = random.randint(18, 36)
        self.app = 0.6  # how often the opponent steps towards me
        self.prev = None
        self.post = None
        self.world = None
        self.quiet = 0
        self.phase = 0
        self.phase_t = 0
        self.sgoal = None

    # ------------------------------------------------------------------------------------------------------------
    def act(self, obs):
        try:
            a = int(self._act(obs))
            if 0 <= a <= 5:
                return a
        except Exception:  # noqa: BLE001
            if _DEBUG:
                raise
        return 0

    def _post(self, W, m):
        if self.post is None:
            dm = W.dist(m)
            ps = [c for c in W.posts() if dm[c] < 99]
            self.post = random.choice(ps[:3]) if ps else m
        return self.post

    # ------------------------------------------------------------------------------------------------------------
    def _act(self, obs):
        walls = obs["walls"]
        W = _world(walls)
        p = obs["player"]
        me, op = obs["me"], obs["opp"]
        m = me[0] * _N + me[1]
        o = op[0] * _N + op[1]
        bm, bo = 1 << m, 1 << o
        step = obs["step"]
        open_, nbm, blast = W.open, W.nbm, W.blast

        bombs, own = {}, {}
        mine = theirs = bmask = 0
        for b in obs["bombs"]:
            q = b["pos"]
            i = q[0] * _N + q[1]
            bombs[i] = b["fuse"]
            bmask |= 1 << i
            if b["owner"] == p:
                mine += 1
                own[i] = True
            else:
                theirs += 1
                own[i] = False

        # the next shrink: at time ks (0 = none)
        ks, S = 0, None
        if _SE:
            left = _SE - step % _SE
            if step + left < _MS:
                S = W.shrink(step // _SE + 1) or None
                if S:
                    ks = left

        dm = W.dist(m)
        do = W.dist(o)
        d_mo = dm[o]

        # how does the opponent move: towards me or away?
        if self.prev is not None:
            pm, po = self.prev
            if po != o and (open_ >> pm) & 1:
                dp = W.dist(pm)
                if dp[o] != dp[po]:
                    self.app += 0.15 * ((1.0 if dp[o] < dp[po] else 0.0) - self.app)
        self.prev = (m, o)

        can_bomb = mine < _MB and not (bmask & bm)
        opp_can_bomb = theirs < _MB and not (bmask & bo)
        mvm = nbm[m] & ~bmask & ~bo  # cells I can step on
        ovm = nbm[o] & ~bmask & ~bm  # cells the opponent can step on
        free_me = open_ & ~bo
        ostart = bo | ovm

        # ---------------------------------------------------------------- my actions
        acts = [0]
        cell = [m, -1, -1, -1, -1, m]
        for a, d in _DIRS:
            j = m + d
            if (mvm >> j) & 1:
                acts.append(a)
                cell[a] = j
        scA = _scen(W, bombs, ks, S)
        scB = None
        if can_bomb:
            acts.append(5)
            bombsB = dict(bombs)
            bombsB[m] = _F
            scB = _scen(W, bombsB, ks, S)

        alive = [0] * 6  # 2 = I get away, 1 = only if the opponent moves out of the way, 0 = lost
        for a in acts:
            sc = scB if a == 5 else scA
            if sc is None:
                alive[a] = 2
            else:
                f = 1 << cell[a]
                if _surv(sc, f, free_me):
                    alive[a] = 2
                elif _surv(sc, f, open_):
                    alive[a] = 1

        # ... also when the opponent drops a bomb right now
        rob = [True] * 6
        near = abs(me[0] - op[0]) + abs(me[1] - op[1]) <= 6
        if opp_can_bomb and near:
            bA = dict(bombs)
            bA[o] = _F
            scAo = _scen(W, bA, ks, S)
            for a in acts:
                if alive[a] == 2:
                    if a == 5:
                        bB = dict(bombsB)
                        bB[o] = _F
                        if not _surv(_scen(W, bB, ks, S), bm, free_me):
                            rob[5] = False
                    elif not _surv(scAo, 1 << cell[a], free_me):
                        rob[a] = False

        # ---------------------------------------------------------------- kills that are certain
        dead0 = scA is not None and not _surv(scA, ostart, open_, True)
        kill_b = scB is not None and alive[5] > 0 and not dead0 and not _surv(scB, ostart, open_, True)
        kill_s = (scA is not None and not dead0 and alive[0] == 2 and scA[3] == 0 and _never(scA, bm)
                  and not _surv(scA, ostart, open_ & ~bm))

        # ---------------------------------------------------------------- one step ahead
        trap0 = [0] * 6  # opponent replies after which I am lost if it keeps standing in my way
        trap = [0] * 6  # opponent replies after which it can kill me with a bomb next turn
        blk = [0] * 6  # ... those of them that kill the opponent too
        pros = [0.0] * 6  # share of the opponent's replies after which I have a certain kill next turn
        nsane = [-1] * 6  # number of replies that keep the opponent alive
        two_ply = ks != 1 and d_mo <= 6
        if two_ply:
            tA = scA[6] if scA is not None else {}
            hit1 = scA[1][1] if scA is not None else 0
            bombs1 = {}
            bmask1 = mine1 = theirs1 = 0
            for i, f in bombs.items():
                if tA[i] > 1:
                    bombs1[i] = f - 1
                    bmask1 |= 1 << i
                    if own[i]:
                        mine1 += 1
                    else:
                        theirs1 += 1
            ks1 = ks - 1 if ks else 0
            oreps = _bits(ostart & ~hit1)
            sc1 = _scen(W, bombs1, ks1, S)
            sc1b = None
            saneA = [n for n in oreps if scA is None or _surv(scA, 1 << n, open_, True)]
            saneB = saneA
            if can_bomb and alive[5] == 2:
                b1b = dict(bombs1)
                b1b[m] = _F - 1
                sc1b = _scen(W, b1b, ks1, S)
                saneB = [n for n in oreps if _surv(scB, 1 << n, open_, True)]

            # can the opponent step to n and then kill me: just by standing in / stepping into my way (trap0),
            # or with a bomb on n (trap)? My way out must not use cells it can contest.
            par1 = sc1 is not None and sc1[3] == 0
            for n in oreps:
                bn = 1 << n
                canb = theirs1 < _MB and not (bmask1 & bn)
                scn = scnb = None
                if canb:
                    b2 = dict(bombs1)
                    b2[n] = _F
                    scn = _scen(W, b2, ks1, S)
                frn = open_ & ~bn
                rn = bn | (nbm[n] & ~bmask1)  # where it can be after next turn's step
                for a in acts:
                    if alive[a] < 2:
                        continue
                    c = cell[a]
                    if c == n:
                        continue
                    if a == 5:
                        full = bm | (nbm[m] & ~bmask1 & ~bn)
                        s1x, snx, bc = sc1b, scnb, bm
                        if canb and snx is None:
                            b3 = dict(b2)
                            b3[m] = _F - 1
                            snx = scnb = _scen(W, b3, ks1, S)
                    else:
                        bc = 1 << c
                        full = bc | (nbm[c] & ~bmask1 & ~bn)
                        s1x, snx = sc1, scn
                    if s1x is not None:
                        if s1x[3]:
                            ok = _surv(s1x, full, frn)
                        else:
                            ok = _surv_par(s1x, full & ~(rn & ~bc), open_, rn & ~bc)
                        if not ok:
                            trap0[a] += 1
                            continue
                    if canb:
                        if snx[3]:
                            ok = _surv(snx, full, frn)
                        else:
                            ok = _surv_par(snx, full, open_, bn)
                        if not ok:
                            trap[a] += 1
                            if not _surv(snx, bn, open_ & ~bc):
                                blk[a] += 1

            # for how many replies of the opponent do I have a certain kill next turn?
            wc = 0.4 + 2.2 * self.app
            wf = 0.4 + 2.2 * (1.0 - self.app)
            for a in acts:
                if alive[a] < 2:
                    continue
                c = cell[a]
                bc = 1 << c
                sc2 = None
                if a == 5:
                    sane, s1, bmk = saneB, sc1b, bmask1 | bm
                else:
                    sane, s1, bmk = saneA, sc1, bmask1
                    if mine1 < _MB and not (bmask1 & bc):
                        b2 = dict(bombs1)
                        b2[c] = _F
                        sc2 = _scen(W, b2, ks1, S)
                hold = a != 5 and s1 is not None and s1[3] == 0 and _never(s1, bc)
                tot = val = 0.0
                cnt = 0
                for n in sane:
                    if n == c:
                        continue
                    cnt += 1
                    bn = 1 << n
                    dn = dm[n]
                    w = wc if dn < d_mo else (wf if dn > d_mo else 1.0)
                    tot += w
                    ost = bn | (nbm[n] & ~bmk & ~bc)
                    if sc2 is not None and not _surv(sc2, ost, open_, True) and _surv(sc2, bc, open_ & ~bn):
                        val += w
                    elif hold and not _surv(s1, ost, open_ & ~bc, True):
                        val += 0.6 * w
                nsane[a] = cnt
                if tot > 0:
                    pros[a] = val / tot

        # ---------------------------------------------------------------- how safe is each action
        # 0 lost, 1 only if the opponent moves out of my way, 2 my move may bounce off the opponent, 3 lost if the
        # opponent drops a bomb now, 4 lost if it steps into my way next turn, 5 lost if it steps somewhere and
        # bombs next turn, 6 in a pocket whose mouth the opponent reaches before I am out, 7 fine,
        # 8 fine and nothing on my way out can be contested by the opponent
        W.killzones()
        vul = W.vul
        level = [-1] * 6
        for a in acts:
            if alive[a] == 0:
                level[a] = 0
            elif alive[a] == 1:
                level[a] = 1
            elif not rob[a]:
                level[a] = 3
            elif trap0[a]:
                level[a] = 4
            elif trap[a]:
                level[a] = 5
            else:
                c = cell[a]
                v = vul[c]
                lv = 7
                if v and d_mo <= 8:
                    dcx = W.dist(c)
                    for x in _bits(v):
                        if do[x] <= dcx[x] and do[x] <= 4:
                            lv = 6
                            break
                if lv == 7:
                    sc = scB if a == 5 else scA
                    if sc is None or (sc[3] == 0 and _surv_par(sc, 1 << c, open_, ostart)):
                        lv = 8
                level[a] = lv
        l0 = level[0]
        for a in acts:  # a move into a cell the opponent may also want can leave me where I am
            if 1 <= a <= 4 and level[a] > l0 and (nbm[cell[a]] & bo):
                if l0 <= 1:
                    level[a] = min(level[a], 2)
                else:
                    level[a] = max(l0, level[a] - 1)
        moves = [a for a in acts if a != 5]
        best_mv = max(level[a] for a in moves)

        # ---------------------------------------------------------------- take a certain kill
        if dead0:  # the opponent is lost already: no new bomb, just stay safe
            pool = [a for a in moves if level[a] == best_mv]
            return 0 if 0 in pool else random.choice(pool)
        if kill_b and alive[5] == 2 and (rob[5] or not kill_s):
            return 5
        if kill_s and rob[0]:
            return 0
        if kill_b:
            return 5

        best = max(best_mv, level[5]) if can_bomb else best_mv
        if best == 0:  # lost: take the opponent along if there is a chance, else live as long as possible
            if can_bomb and (blast[m] & ostart):
                return 5
            if scA is None:
                return 0
            return max(moves, key=lambda a: _life(scA, 1 << cell[a], open_))

        # ---------------------------------------------------------------- style: where do I want to be
        if d_mo <= 3:
            self.quiet = 0
        else:
            self.quiet += 1
        if self.world is not W:
            self.world, self.post = W, None
        goal, want_d, hide = None, 2, False
        if self.style == 0:  # lurker
            if self.phase == 0:
                if self.quiet > self.patience:
                    self.phase, self.phase_t = 1, step + random.randint(15, 30)
            elif step >= self.phase_t or d_mo <= 2:
                self.phase, self.quiet = 0, 0
            if self.phase == 0:
                goal, hide, want_d = self._post(W, m), True, 3
        elif self.style == 2:  # feinter
            if self.phase == 0:
                if d_mo <= 3:
                    self.phase, self.phase_t = 1, step + random.randint(8, 16)
            elif step >= self.phase_t:
                self.phase = 0
            if self.phase == 1:
                goal, hide = self._post(W, m), True
            want_d = 3
        din = None
        if S is not None and ks <= self.lead + (8 if self.style == 0 else 0) + (8 if step >= _SE else 0):
            din = S[4]  # the ring closes soon: get inside, to a good cell of the map as it will be
            W2 = S[2]
            if self.sgoal is None or self.sgoal[0] is not W2:
                ps = [c for c in W2.posts() if dm[c] < 99][:3]
                self.sgoal = (W2, min(ps, key=lambda c: dm[c] - 0.1 * W2.ring[c]) if ps else None)
            goal = self.sgoal[1]

        # ---------------------------------------------------------------- bombs that are not (yet) a kill
        if can_bomb and level[5] == 8:
            inbl = (blast[m] >> o) & 1
            chain = scB[6][m] < _F  # it would go off early, with an older bomb
            nA, nB = nsane[0], nsane[5]
            want = 0.0
            if two_ply and nB >= 0 and nB <= 1 < nA:
                want = 1.0  # one way out left
            elif inbl:
                want = 0.95 if chain else self.direct_p
                if nB >= 0 and nB < nA:
                    want += 0.2
            elif d_mo <= 4:
                reach = ostart
                reach |= ((reach << 1) | (reach >> 1) | (reach << _N) | (reach >> _N)) & open_ & ~bmask
                cut = (blast[m] & reach).bit_count()
                if chain and cut >= 1:
                    want = 0.8
                elif cut >= 2:
                    want = self.cut_p
                elif mine == 0 and self.style != 1 and d_mo >= 3 and self.app > 0.5:
                    want = self.seed_p
            if S is not None and 2 <= ks <= 4 and (S[1] & bm) and (S[0] & bo) and S[4][o] >= ks:
                land = S[3].get(o)  # the opponent will be put on land[0] by the shrink
                if land and (S[2].blast[m] >> land[0]) & 1:
                    want = max(want, 0.9)
            if din is not None and din[m] > 0 and ks <= _F:
                want = 0.0  # it would vanish with the ring
            if want > 0.0 and random.random() < want:
                return 5

        # ---------------------------------------------------------------- walk
        if best_mv < level[5] and can_bomb and level[5] >= 5:
            return 5  # the bomb is the safest thing to do (it closes a way the opponent would use)
        pool = [a for a in moves if level[a] == best_mv]
        if len(pool) == 1:
            return pool[0]
        dg = W.dist(goal) if goal is not None else None
        lineo = blast[o]
        freeb = open_ & ~bmask
        k_opp = 0.15 if hide else 0.4
        vul2 = S[2].vul if din is not None and S[2].killzones() else None
        best_a, best_s = pool[0], -1e9
        for a in pool:
            c = cell[a]
            bc = 1 << c
            dc = do[c]
            s = 60.0 * pros[a] - 6.0 * (trap[a] + trap0[a]) + 3.0 * blk[a] + 0.35 * random.random()
            if d_mo <= 8:  # room: cells I reach first / cells the opponent reaches first
                s += 0.5 * min(_terr(bc, ostart, freeb).bit_count(), 12)
                s -= k_opp * min(_terr(bo, bc | (nbm[c] & freeb), freeb).bit_count(), 12)
            if din is not None:
                s -= 5.0 * din[c] + (25.0 if din[c] >= ks else 0.0)
                if dg is not None:
                    s -= 1.2 * dg[c]
                if vul2[c]:
                    s -= 6.0
            elif dg is not None and d_mo > 2:
                s -= 1.5 * dg[c]
            else:
                s -= 1.5 * abs(dc - want_d)
            v = vul[c]
            if v:  # a cell a single bomb can close: not when the opponent is close to the cell it needs
                kd = min(do[x] for x in _bits(v))
                if kd <= 4:
                    s -= 2.5 * (5 - kd)
            if dc <= 3:
                s += 0.5 * (nbm[c] & ~bmask).bit_count()
                if hide and (lineo >> c) & 1:
                    s -= 2.0
            if s > best_s:
                best_a, best_s = a, s
        return best_a
