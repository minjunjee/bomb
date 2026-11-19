"""Network definitions shared by every training pipeline, and the export to the submission format.

The grading server gets only `submission.py` (class `Net`, built as `Net()` with no arguments) and `submission.pth`
(its state_dict) and plays `NeuralPolicy(net)`: turn(obs) -> the fixed preprocess -> Net -> argmax of 6 scores.
So the architecture lives in ONE source template (`NET_SOURCE`) that is used both for training (`build_net(cfg)`)
and for the exported file (`export_submission`): the two can never drift apart.

Rules the exported Net obeys
- learned layers only (no game logic), `Net()` without arguments, output = the 6 action scores (a plain tensor);
- identical behaviour in train() and eval() mode (no BatchNorm / Dropout), so it does not matter which one the server uses;
- deterministic, CPU friendly (a few ms per step on one core).

Symmetry helpers (`SYM_CELL`, `SYM_ACT`, `augment`) implement the 8 board symmetries; index 0 is the identity and
index 1 the transpose (rows <-> columns, which also keeps the start corners in place).

    from arena.nets import build_net, export_submission, check_submission, augment, DEFAULT_CFG
    net = build_net({"channels": 96, "blocks": 8})         # forward(x) -> (logits (B, 6), value (B,))
    export_submission(net, "runs/my_run/best")             # writes submission.py + submission.pth
    check_submission("runs/my_run/best")                   # the notebook's server check + timing
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CFG = {
    "channels": 96,     # width of the residual tower (a multiple of 8 when norm == "gn")
    "blocks": 8,        # residual blocks (2 convs each)
    "norm": "gn",       # "gn" (GroupNorm, 8 groups) | "ln" (GroupNorm, 1 group) | "none"
    "head": "gather",   # "gather": read the tower at my / the opponent's cell + global pooling | "flat": 1x1 conv + flatten
    "head_hidden": 256,
    "sym": 1,           # 1 | 2 | 8: average the scores over this many board symmetries inside forward (k x compute)
}

# ----------------------------------------------------------------------------------------------------------------
# The single source of truth for the architecture. `__CFG__` and `__RETURN_VALUE__` are filled in by render_source().
# ----------------------------------------------------------------------------------------------------------------
NET_SOURCE = '''# Everything grading needs is in this file. The server has: numpy, torch (CPU), bombcompetition.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bombcompetition.neural import CHANNELS, IN_CHANNELS  # the fixed input: (IN_CHANNELS, 9, 9)

CFG = __CFG__
RETURN_VALUE = __RETURN_VALUE__  # training: (scores, value); submission: the 6 action scores only

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
'''


def render_source(cfg: dict | None = None, return_value: bool = False) -> str:
    full = {**DEFAULT_CFG, **(cfg or {})}
    unknown = set(full) - set(DEFAULT_CFG)
    if unknown:
        raise ValueError(f"unknown cfg keys: {unknown}")
    return NET_SOURCE.replace("__CFG__", repr(full)).replace("__RETURN_VALUE__", repr(bool(return_value)))


def build_net(cfg: dict | None = None, return_value: bool = True) -> torch.nn.Module:
    """The training-time network: forward(x (B, 13, 9, 9)) -> (scores (B, 6), value (B,)). `net.cfg` keeps the config."""
    ns: dict = {"__name__": "arena_net_generated"}
    exec(compile(render_source(cfg, return_value), "<arena.nets.NET_SOURCE>", "exec"), ns)  # noqa: S102
    net = ns["Net"]()
    net.cfg = dict(ns["CFG"])
    return net


def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------------------------------------------------------
# symmetries for data augmentation (numpy / torch, any device)
# ----------------------------------------------------------------------------------------------------------------
_ns: dict = {"__name__": "arena_net_tables"}
exec(compile(render_source(), "<arena.nets.NET_SOURCE>", "exec"), _ns)  # noqa: S102
_cells, _acts = _ns["_symmetry_tables"]()
SYM_CELL = np.array(_cells, dtype=np.int64)  # (8, 81): view k of a board = board.reshape(.., 81)[..., SYM_CELL[k]]
SYM_ACT = np.array(_acts, dtype=np.int64)    # (8, 6):  action a is action SYM_ACT[k][a] in view k
SYM_ACT_INV = np.argsort(SYM_ACT, axis=1)    # (8, 6):  per-action vectors q (.., 6) become q[..., SYM_ACT_INV[k]] in view k
del _ns, _cells, _acts


def augment(x: torch.Tensor, y: torch.Tensor | None = None, k: torch.Tensor | int | None = None,
            q: torch.Tensor | None = None, n_sym: int = 8):
    """Apply board symmetries to a batch. x: (B, C, 9, 9); y: (B,) action labels; q: (B, 6) per-action targets.
    k: (B,) symmetry index per sample (default: random in [0, n_sym)). Returns (x, y, q) transformed (None stays None)."""
    b = x.shape[0]
    dev = x.device
    if k is None:
        k = torch.randint(0, n_sym, (b,), device=dev)
    elif isinstance(k, int):
        k = torch.full((b,), k, device=dev, dtype=torch.long)
    cell = torch.as_tensor(SYM_CELL, device=dev)[k]  # (B, 81)
    xs = torch.gather(x.flatten(2), 2, cell[:, None, :].expand(-1, x.shape[1], -1)).view_as(x)
    ys = None if y is None else torch.as_tensor(SYM_ACT, device=dev)[k, y.long()]
    qs = None if q is None else torch.gather(q, 1, torch.as_tensor(SYM_ACT_INV, device=dev)[k])
    return xs, ys, qs


# ----------------------------------------------------------------------------------------------------------------
# export / load / check
# ----------------------------------------------------------------------------------------------------------------
def export_submission(net, folder: str | Path, cfg: dict | None = None, extra: dict | None = None) -> Path:
    """Write <folder>/submission.py (self-contained, returns the 6 scores) and <folder>/submission.pth (CPU state_dict).
    `net`: a module from build_net (its .cfg is used) or a state_dict (then pass cfg). Atomic enough for readers:
    files are written to a temp name and renamed."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(net, dict):
        state, cfg = net, {**DEFAULT_CFG, **(cfg or {})}
    else:
        module = getattr(net, "_orig_mod", net)  # torch.compile wrapper
        module = getattr(module, "module", module)  # DataParallel
        state, cfg = module.state_dict(), {**getattr(module, "cfg", DEFAULT_CFG), **(cfg or {})}
    state = {k: v.detach().to("cpu", copy=True).contiguous() for k, v in state.items()}
    tmp_py, tmp_pth = folder / ".submission.py.tmp", folder / ".submission.pth.tmp"
    tmp_py.write_text(render_source(cfg, return_value=False))
    torch.save(state, tmp_pth)
    os.replace(tmp_pth, folder / "submission.pth")
    os.replace(tmp_py, folder / "submission.py")
    (folder / "meta.json").write_text(json.dumps({"cfg": cfg, "params": int(sum(v.numel() for v in state.values())),
                                                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}, indent=1))
    return folder


def load_submission(folder: str | Path, return_value: bool = True, device: str = "cpu") -> torch.nn.Module:
    """Rebuild a training-time network (scores, value) from an exported folder."""
    folder = Path(folder)
    cfg = json.loads((folder / "meta.json").read_text())["cfg"]
    net = build_net(cfg, return_value)
    net.load_state_dict(torch.load(folder / "submission.pth", map_location="cpu"))
    return net.to(device)


_CHECK = r"""
import importlib.util, json, sys, time
import torch
torch.set_num_threads(1)
from bombcompetition import RULES, Game, SafeBot, random_rules
from bombcompetition.neural import NeuralPolicy
t0 = time.perf_counter()
spec = importlib.util.spec_from_file_location("submission_check", "submission.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def server_policy():  # what the server does
    net = module.Net()
    net.load_state_dict(torch.load("submission.pth", map_location="cpu"))
    return NeuralPolicy(net.eval())  # the fixed preprocess

bot = SafeBot(server_policy)
load_s = time.perf_counter() - t0
assert bot.policy is not None, "Net() or loading submission.pth failed: " + "; ".join(bot.errors)
# the scores must not depend on train()/eval() mode
net = bot.policy.net
x = torch.rand(4, 13, 9, 9)
with torch.no_grad():
    a = net.eval()(x); b = net.train()(x); net.eval()
assert isinstance(a, torch.Tensor) and a.shape == (4, 6), f"bad output {type(a)}"
assert torch.allclose(a, b, atol=1e-5), "the output differs between train() and eval() mode"
steps, spent = 0, 0.0
for seed in range(3):
    game = Game(random_rules(seed))
    while not game.done:
        t = time.perf_counter(); act = bot.act(game.obs(seed % 2)); spent += time.perf_counter() - t; steps += 1
        game.step(*((act, 0) if seed % 2 == 0 else (0, act)))
assert bot.faults == 0, "Net failed: " + "; ".join(bot.errors)
print(json.dumps({"ok": True, "load_s": round(load_s, 3), "ms_per_act": round(1000 * spent / steps, 3),
                  "slowest_ms": round(bot.slowest * 1000, 2), "steps": steps}))
"""


def check_submission(folder: str | Path, verbose: bool = True) -> dict:
    """The notebook's last-cell check: a fresh process in a folder that holds only the two files, no GPU, 1 thread.
    Returns {"ok", "load_s", "ms_per_act", "slowest_ms", "py_bytes", "pth_bytes"} or {"ok": False, "error"}."""
    folder = Path(folder)
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("submission.py", "submission.pth"):
            shutil.copy(folder / f, tmp)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        r = subprocess.run([sys.executable, "-c", _CHECK], cwd=tmp, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        out = {"ok": False, "error": (r.stdout + r.stderr)[-3000:]}
    else:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    out["py_bytes"] = (folder / "submission.py").stat().st_size
    out["pth_bytes"] = (folder / "submission.pth").stat().st_size
    if verbose:
        print("check_submission:", out)
    return out


def make_zip(folder: str | Path, out: str | Path | None = None) -> Path:
    """submission.zip with the two files at the top level (what the leaderboard page takes)."""
    import zipfile

    folder = Path(folder)
    out = Path(out) if out else folder / "submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ("submission.py", "submission.pth"):
            zf.write(folder / f, f)
    return out


# ----------------------------------------------------------------------------------------------------------------
# ensembles: one submitted Net that averages the action probabilities of several trained networks
# ----------------------------------------------------------------------------------------------------------------
def render_ensemble_source(cfgs: list[dict]) -> str:
    src = NET_SOURCE.replace("__CFG__", "None").replace("__RETURN_VALUE__", "False")
    old = "class Net(nn.Module):\n    def __init__(self):"
    assert old in src
    src = src.replace(old, "class Member(nn.Module):\n    def __init__(self, CFG):")
    full = [{**DEFAULT_CFG, **c} for c in cfgs]
    return src + f"""

CFGS = {full!r}


class Net(nn.Module):
    \"\"\"The mean of the action probabilities of several trained networks (an ensemble).\"\"\"

    def __init__(self):
        super().__init__()
        self.members = nn.ModuleList([Member(c) for c in CFGS])

    def forward(self, x):
        p = torch.softmax(self.members[0](x), 1)
        for m in self.members[1:]:
            p = p + torch.softmax(m(x), 1)
        return p / len(self.members)
"""


def export_ensemble(folders: list, out: str | Path, extra: dict | None = None) -> Path:
    """Write <out>/submission.{py,pth}: a Net that averages softmax(scores) of the networks exported in `folders`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfgs, state = [], {}
    for i, f in enumerate(folders):
        f = Path(f)
        cfgs.append(json.loads((f / "meta.json").read_text())["cfg"])
        for k, v in torch.load(f / "submission.pth", map_location="cpu").items():
            state[f"members.{i}.{k}"] = v
    (out / "submission.py").write_text(render_ensemble_source(cfgs))
    torch.save(state, out / "submission.pth")
    (out / "meta.json").write_text(json.dumps({"ensemble": [str(f) for f in folders], "cfgs": cfgs,
                                               "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}, indent=1))
    return out


if __name__ == "__main__":  # python -m arena.nets check <folder>
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        res = check_submission(sys.argv[2])
        sys.exit(0 if res.get("ok") else 1)
    print(__doc__)
