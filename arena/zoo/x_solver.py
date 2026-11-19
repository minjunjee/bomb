"""zoo.x_solver: an EXACT ADVERSARIAL SOLVER used as a teacher (stateless: act(obs) depends on obs only).

Every decision solves the current position with the exact forward model in x_solver_core (numba):

  (b) forced win   an action after which the opponent alone is hit within D_KILL steps whatever it does
                   (exists my move, for all its moves, iterative deepening -> the fastest kill is played);
  (a) safety       an action is SAFE if after it, whatever the opponent does (bombs, body blocks, chains), I can
                   still guarantee not to be hit for H_SAFE steps; unsafe actions are discarded when a safe one exists;
  (c) pressure     among safe actions: the share of plausible opponent replies after which I have a forced kill
                   (1.0), after which the opponent can no longer guarantee its own survival (V_UNSAFE) or merely
                   stands in a blast (V_DANGER); then the shortest path to the opponent; then a fixed order;
  (d) guessing     if no action is safe, the action with the smallest weighted share of losing opponent replies.

Deterministic; all tie-breaks are turned with the board for player B, so the labels are consistent in the
network's (turned) frame.  action_values() returns the 6 numbers whose argmax act() plays.
"""

import numpy as np

try:
    from . import x_solver_core as C
except Exception:  # loaded by path
    import importlib.util
    import os
    import sys

    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "x_solver_core.py")
    if "x_solver_core" in sys.modules:
        C = sys.modules["x_solver_core"]
    else:
        _s = importlib.util.spec_from_file_location("x_solver_core", _p)
        C = importlib.util.module_from_spec(_s)
        sys.modules["x_solver_core"] = C
        _s.loader.exec_module(C)

# ---------------------------------------------------------------------------------------------------------------
# the search budget (easy to change)
# ---------------------------------------------------------------------------------------------------------------
H_SAFE = 8  # guaranteed-survival horizon in steps (covers every bomb on the board and the next two drops)
D_KILL = 9  # depth of the forced-win search at the root (5 = "bomb now and it cannot get out")
D_PRESS = 7  # depth of the forced-kill search after each opponent reply (pressure)
N_KILL = 30000  # node budgets of the three searches (a node = one expanded position)
N_SAFE = 30000
N_PRESS = 30000

V_UNSAFE = 0.35  # value of an opponent reply after which it cannot guarantee its survival
V_DANGER = 0.05  # value of an opponent reply after which it stands in a blast (it still has to act)
W_IMPLAUSIBLE = 0.15  # weight of opponent replies that walk into the bombs already on the board
SHRINK_AHEAD = 10  # steps before a shrink from which distances are taken on the shrunk map

FLIP = (0, 2, 1, 4, 3, 5)
PREF_TURNED = (2, 4, 1, 3, 0, 5)  # tie-break order in the turned frame: DOWN, RIGHT, UP, LEFT, STAY, BOMB

_CFG = np.array([H_SAFE, D_KILL, D_PRESS, N_KILL, N_SAFE, N_PRESS], dtype=np.int64)
_W = C.new_work()
_OUT = np.zeros((C.R_ROWS, 36), dtype=np.int64)
_DIST = np.zeros(49, dtype=np.int64)
_DIN = np.zeros(49, dtype=np.int64)
_TABLES = {}
_GEN = [0]


def _tables(walls, ph):
    key = (walls.tobytes(), ph)
    T = _TABLES.get(key)
    if T is None:
        if len(_TABLES) > 600:
            _TABLES.clear()
        T = C.build_tables(walls, ph)
        _TABLES[key] = T
    return T


def _solve(obs, cfg=None):
    """-> (values[6], info)"""
    if cfg is None:
        cfg = _CFG
    walls = np.asarray(obs["walls"], dtype=bool)
    t = int(obs["step"])
    me = int(obs["player"])
    if t >= 300:
        return [0.0, -1.0, -1.0, -1.0, -1.0, -1.0], None
    ph = t // 100
    T = _tables(walls, ph)
    mr, mc = obs["me"]
    orr, oc = obs["opp"]
    mcell = (int(mr) - 1) * 7 + int(mc) - 1
    ocell = (int(orr) - 1) * 7 + int(oc) - 1
    bl = []
    for b in obs["bombs"]:
        r, c = b["pos"]
        bl.append(((int(r) - 1) * 7 + int(c) - 1, min(7, max(1, int(b["fuse"]))), int(b["owner"]) & 1))
    bombs = C.pack_bombs(bl[:6])
    p0, p1 = (mcell, ocell) if me == 0 else (ocell, mcell)
    _GEN[0] = (_GEN[0] + 1) & 0x3FFFFFFF
    _W[C.W_GEN] = _GEN[0] + 1
    OUT = _OUT
    C.root_eval(p0, p1, bombs, t, me, cfg, T, _W, OUT)

    legal_a = OUT[C.R_ROOT, 6:12]
    legal_b = OUT[C.R_ROOT, 12:18]
    wind = OUT[C.R_ROOT, 0:6]
    B = [b for b in range(6) if legal_b[b]]
    wts = [1.0 if OUT[C.R_MISC, b] else W_IMPLAUSIBLE for b in B]
    wsum = sum(wts)

    # distances to the opponent (on the shrunk map shortly before a shrink)
    left = 100 - t % 100
    k = ph + 1
    use_next = k <= 2 and left <= SHRINK_AHEAD and T[C.O_SHR + k] != 0
    if use_next:
        ring = int(T[C.O_RING + k])
        src = ocell
        if (ring >> ocell) & 1:
            src = int(T[C.O_TEL + (k - 1) * 1225 + ocell * 25])
        C.bfs_dist(1 << src, k, T, _DIST)
        C.bfs_dist(int(T[C.O_OPEN + k]), ph, T, _DIN)
    else:
        ring = 0
        C.bfs_dist(1 << ocell, ph, T, _DIST)

    def dist_of(cell):
        if (ring >> cell) & 1:
            return 40.0 + float(_DIN[cell])
        d = int(_DIST[cell])
        return 25.0 if d < 0 else float(d)

    pref = PREF_TURNED if me == 0 else tuple(FLIP[a] for a in PREF_TURNED)
    eps = {a: 0.01 * (5 - i) for i, a in enumerate(pref)}
    vals = [None] * 6
    info = {"safe": [], "win": [], "press": {}, "danger": {}}
    for a in range(6):
        if not legal_a[a]:
            continue
        cell = int(T[C.O_NBR + ph * 245 + mcell * 5 + a]) if 1 <= a <= 4 else mcell
        base = a * 6
        if wind[a] < 99:
            vals[a] = 1e6 - 1e4 * float(wind[a]) + eps[a]
            info["win"].append(a)
            continue
        danger = 0.0
        unsafe = 0
        for b, w in zip(B, wts):
            j = base + b
            if OUT[C.R_HITM, j]:
                danger += w * (0.9 if OUT[C.R_HITO, j] else 1.0)
                unsafe += 1
            elif not OUT[C.R_SVM, j]:
                danger += w * 0.6
                unsafe += 1
        if unsafe:
            danger /= wsum
            info["danger"][a] = danger
            vals[a] = -1e5 * (0.05 + danger) - 10.0 * dist_of(cell) + eps[a]
            continue
        press = 0.0
        for b, w in zip(B, wts):
            j = base + b
            if OUT[C.R_KILL, j] < 99:
                press += w
            elif OUT[C.R_SVO, j] == 0:
                press += w * V_UNSAFE
            elif OUT[C.R_DNG, j]:
                press += w * V_DANGER
        press /= wsum
        info["safe"].append(a)
        info["press"][a] = press
        vals[a] = 1e5 + 1000.0 * press - 10.0 * dist_of(cell) + eps[a]
    stay = vals[0] if vals[0] is not None else 0.0
    for a in range(6):
        if vals[a] is None:
            vals[a] = stay - 3.0 - 0.01 * a
    info["stat"] = [int(x) for x in OUT[C.R_STAT, 0:6]]
    return vals, info


def _best(vals):
    best = 0
    for a in range(1, 6):
        if vals[a] > vals[best]:
            best = a
    return best


class Policy:
    cfg = None  # None = the module constants

    def __init__(self):
        pass

    def action_values(self, obs):
        try:
            return [float(v) for v in _solve(obs, self.cfg)[0]]
        except Exception:  # noqa: BLE001
            return [0.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    def act(self, obs):
        try:
            return int(_best(_solve(obs, self.cfg)[0]))
        except Exception:  # noqa: BLE001
            return 0


def make_policy(arg):
    """zoo.x_solver:H-DK-DP[-NK-NS-NP]  (experiments with other search budgets)"""
    v = [int(x) for x in str(arg).split("-")]
    full = [H_SAFE, D_KILL, D_PRESS, N_KILL, N_SAFE, N_PRESS]
    full[:len(v)] = v

    class P(Policy):
        cfg = np.array(full, dtype=np.int64)

    return P


def explain(obs):
    """Debug helper: (values, info, a copy of the root matrices)."""
    vals, info = _solve(obs)
    return vals, info, _OUT.copy()
