"""The fixed way a submitted network plays (the same code in the notebook and on the grading server).

Every step:  obs -> turn(obs) -> preprocess -> Net -> the action with the highest score -> turned back.

preprocess is fixed too (below): every network gets the same IN_CHANNELS x 9 x 9 input, made only of facts about the
board (no move is chosen for the network). Students write Net and train it; they do not write preprocess or act().

turn(obs) shows B the board turned 180°, so a network always sees itself starting top-left, like A
(maps are symmetric, so both sides are the same game). The chosen action is turned back for B
(UP<->DOWN, LEFT<->RIGHT).
"""

from collections import deque

import numpy as np
import torch

from .game import RULES, blast_cells, shrink_ring

FLIP = [0, 2, 1, 4, 3, 5]  # turning the board 180° swaps UP<->DOWN and LEFT<->RIGHT; STAY and BOMB stay


def turn(obs: dict) -> dict:
    """The observation as the player sees it: for B (player 1) the board turned 180°."""
    if obs["player"] == 0:
        return obs
    n = len(obs["walls"])
    rot = lambda p: (n - 1 - p[0], n - 1 - p[1])
    return {**obs,
            "walls": np.asarray(obs["walls"])[::-1, ::-1].copy(),
            "me": rot(obs["me"]), "opp": rot(obs["opp"]),
            "bombs": tuple({**b, "pos": rot(b["pos"])} for b in obs["bombs"])}


def turn_action(action: int, obs: dict) -> int:
    """An action chosen on the turned board, as a real action."""
    return FLIP[int(action)] if obs["player"] == 1 else int(action)


CHANNELS = (
    "walls",          # 1 = wall
    "me",             # 1 = my cell
    "opp",            # 1 = the opponent's cell
    "my_bombs",       # 1 = a bomb of mine
    "opp_bombs",      # 1 = a bomb of the opponent
    "danger",         # cells a bomb will hit: 1 = at the end of this step, 1/2 = of the next step, 1/3 ... (chains included)
    "my_reach",       # cells my bomb would hit if I dropped it now
    "opp_reach",      # cells the opponent's bomb would hit if it dropped it now
    "my_bombs_left",  # the whole plane: bombs I can still drop / max_bombs
    "free",           # 1 = a cell you can walk on now (no wall, no bomb)
    "my_dist",        # 1 / (1 + steps from me), walking around walls, bombs and the opponent; 0 = cannot get there
    "opp_dist",       # the same from the opponent
    "closing",        # open cells that turn into walls at the next sudden-death shrink: 1 / (steps until then)
)
IN_CHANNELS = len(CHANNELS)  # the network's input: (IN_CHANNELS, 9, 9)


def _steps(walls, start, blocked):
    """BFS: {cell: steps from start}, not entering walls or blocked cells (start itself is allowed)."""
    seen, queue = {start: 0}, deque([start])
    while queue:
        r, c = queue.popleft()
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not walls[n] and n not in blocked and n not in seen:
                seen[n] = seen[(r, c)] + 1
                queue.append(n)
    return seen


def preprocess(obs: dict, rules=RULES) -> torch.Tensor:
    """The fixed input of every network: a turned observation (see turn) -> (IN_CHANNELS, 9, 9) float tensor.
    Channel k is CHANNELS[k]. Bomb timing uses obs["fuse"]: exact with fixed fuses (the league rules)."""
    me, walls = obs["player"], np.asarray(obs["walls"])
    n = len(walls)
    x = {name: np.zeros((n, n), dtype=np.float32) for name in CHANNELS}
    x["walls"][:] = walls
    x["me"][obs["me"]] = 1
    x["opp"][obs["opp"]] = 1
    t = {b["pos"]: b["fuse"] for b in obs["bombs"]}  # when each bomb explodes, chain reactions included
    for _ in range(len(t)):
        for c in t:
            for d in blast_cells(walls, c, rules.radius):
                if d in t:
                    t[d] = min(t[d], t[c])
    for c, tc in t.items():
        for cell in blast_cells(walls, c, rules.radius):
            x["danger"][cell] = max(x["danger"][cell], 1 / tc)
    mine = 0
    for b in obs["bombs"]:
        x["my_bombs" if b["owner"] == me else "opp_bombs"][b["pos"]] = 1
        mine += b["owner"] == me
    for cell in blast_cells(walls, obs["me"], rules.radius):
        x["my_reach"][cell] = 1
    for cell in blast_cells(walls, obs["opp"], rules.radius):
        x["opp_reach"][cell] = 1
    x["my_bombs_left"][:] = (rules.max_bombs - mine) / rules.max_bombs
    bombs = set(t)
    x["free"][:] = walls == 0
    for c in bombs:
        x["free"][c] = 0
    for name, who, other in (("my_dist", obs["me"], obs["opp"]), ("opp_dist", obs["opp"], obs["me"])):
        for cell, d in _steps(walls, who, bombs | {other}).items():
            x[name][cell] = 1 / (1 + d)
    every = rules.shrink_every
    if every:
        k, left = obs["step"] // every + 1, every - obs["step"] % every
        ring, inner = shrink_ring(walls, k)
        if inner and obs["step"] + left < rules.max_steps:
            for cell in ring:
                if not walls[cell]:
                    x["closing"][cell] = 1 / left
    return torch.from_numpy(np.stack([x[name] for name in CHANNELS]))


def scores(net, preprocess, obs) -> torch.Tensor:
    """The network's 6 action scores for obs (on the turned board). A Net may return a tuple; the first part counts."""
    device = next(net.parameters()).device
    out = net(preprocess(turn(obs))[None].to(device))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.reshape(-1, 6)[0]


class NeuralPolicy:
    """act(obs): the action with the highest score, turned back. This is the server's act(), with the fixed
    preprocess above (the second argument exists only for TA experiments)."""

    def __init__(self, net, preprocess=preprocess):
        self.net, self.preprocess = net, preprocess

    def act(self, obs) -> int:
        with torch.no_grad():
            return turn_action(int(scores(self.net, self.preprocess, obs).argmax()), obs)
