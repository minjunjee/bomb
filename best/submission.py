# Everything grading needs is in this file. The server has: numpy, torch (CPU), bombcompetition.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bombcompetition.neural import CHANNELS, IN_CHANNELS  # the fixed input: (IN_CHANNELS, 9, 9)

CFG = {'channels': 96, 'blocks': 8, 'norm': 'gn', 'head': 'gather', 'head_hidden': 256, 'sym': 1}
RETURN_VALUE = False  # training: (scores, value); submission: the 6 action scores only

_ACTION_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))  # STAY UP DOWN LEFT RIGHT BOMB
# The 8 symmetries of the board as 2x2 matrices on (row, col) offsets from the centre; 0 = identity, 1 = transpose.
_SYMMETRIES = (((1, 0), (0, 1)), ((0, 1), (1, 0)), ((-1, 0), (0, -1)), ((0, -1), (-1, 0)),
               ((-1, 0), (0, 1)), ((1, 0), (0, -1)), ((0, 1), (-1, 0)), ((0, -1), (1, 0)))


def _symmetry_tables(n=9):
    """cell[k][i] = the source cell of cell i in the board seen through symmetry k;
    act[k][a] = the action that action a becomes in that view."""
    c = (n - 1) // 2
    cells, acts = [], []
    for m in _SYMMETRIES:
        src = [0] * (n * n)
        for r in range(n):
            for q in range(n):
                r2 = m[0][0] * (r - c) + m[0][1] * (q - c) + c
                q2 = m[1][0] * (r - c) + m[1][1] * (q - c) + c
                src[r2 * n + q2] = r * n + q
        cells.append(src)
        act = []
        for a, (dr, dc) in enumerate(_ACTION_DELTAS):
            d2 = (m[0][0] * dr + m[0][1] * dc, m[1][0] * dr + m[1][1] * dc)
            act.append(a if a in (0, 5) else _ACTION_DELTAS.index(d2, 1, 5))
        acts.append(act)
    return cells, acts


def _norm(kind, c):
    if kind == "gn":
        return nn.GroupNorm(8, c)
    if kind == "ln":
        return nn.GroupNorm(1, c)
    return nn.Identity()


class Block(nn.Module):
    """Pre-activation residual block; the last conv starts at zero, so every block starts as the identity."""

    def __init__(self, c, kind):
        super().__init__()
        self.n1, self.c1 = _norm(kind, c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = _norm(kind, c), nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        y = self.c1(F.relu(self.n1(x)))
        return x + self.c2(F.relu(self.n2(y)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c, kind, hid = CFG["channels"], CFG["norm"], CFG["head_hidden"]
        self.sym = int(CFG["sym"])
        self.head = CFG["head"]
        self.stem = nn.Conv2d(IN_CHANNELS, c, 3, padding=1)
        self.blocks = nn.ModuleList([Block(c, kind) for _ in range(CFG["blocks"])])
        self.nf = _norm(kind, c)
        if self.head == "gather":
            feat = 4 * c
        else:
            self.squeeze = nn.Conv2d(c, 16, 1)
            feat = 16 * 81
        self.pi = nn.Sequential(nn.Linear(feat, hid), nn.ReLU(), nn.Linear(hid, 6))
        self.v = nn.Sequential(nn.Linear(feat, hid), nn.ReLU(), nn.Linear(hid, 1))
        cells, acts = _symmetry_tables()
        self.register_buffer("sym_cell", torch.tensor(cells, dtype=torch.long), persistent=False)
        self.register_buffer("sym_act", torch.tensor(acts, dtype=torch.long), persistent=False)

    def features(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = F.relu(self.nf(h))
        if self.head == "gather":
            me, opp = x[:, 1:2], x[:, 2:3]  # the input planes that mark my cell and the opponent's cell
            return torch.cat([(h * me).sum((2, 3)), (h * opp).sum((2, 3)), h.mean((2, 3)), h.amax((2, 3))], 1)
        return F.relu(self.squeeze(h)).flatten(1)

    def scores_value(self, x):
        z = self.features(x)
        return self.pi(z), self.v(z).squeeze(1)

    def forward(self, x):
        if self.sym <= 1:
            scores, value = self.scores_value(x)
        else:  # average over board symmetries: view k of the board, scores mapped back to the real actions
            b, k = x.shape[0], self.sym
            flat = x.flatten(2)
            views = torch.cat([flat[:, :, self.sym_cell[i]] for i in range(k)], 0).view(k * b, *x.shape[1:])
            s, v = self.scores_value(views)
            s = s.view(k, b, 6)
            scores = sum(s[i][:, self.sym_act[i]] for i in range(k)) / k
            value = v.view(k, b).mean(0)
        return (scores, value) if RETURN_VALUE else scores
