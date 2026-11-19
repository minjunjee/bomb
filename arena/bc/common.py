"""arena.bc.common - helpers shared by gen.py, train.py, dagger.py and noise.py (no CLI of its own).

1. The exact uint8 codec for the 13 input planes
   Every value `preprocess` can produce is 0, float32(1/n) (n = 1..254: 0/1 planes, danger 1/t, distances 1/(1+d),
   closing 1/left) or float32(2/3) (my_bombs_left). A plane value is stored as one byte: code 0 -> 0.0,
   code n -> float32(1/n), code 255 -> float32(2/3). `LUT[codes]` (numpy) or `lut[codes.long()]` (torch, any device)
   gives back bit-identical float32; `encode_planes` checks `np.array_equal(LUT[codes], x)` and raises otherwise.
   1053 bytes per sample instead of 4212.

2. The pair mini-language (`--pairs` of gen.py / dagger.py)
       pairs := pair ("," pair)*
       pair  := actor "|" actor ["*" weight]            weight: relative share of the games (default 1)
       actor := ["+"] spec                              "+" = label this actor's states with the teacher
       spec  := any arena.core bot spec; a ":"-separated field that is exactly  T  is replaced by the teacher spec,
                S  by net:<student folder> and  S@sample  by net:<student folder>@sample
   If no actor of a pair carries "+", BOTH actors' states are labelled (the default).
       "T|T, T|noisy:0.2:T*2, T|simple, +S|T, +S@sample|zoo.x, +noisy:0.1:S|+S"

3. The dataset layout (written by gen.py, read by train.py)
       <dir>/manifest.json            format, arguments, counts, the list of shards (exists only when the run finished)
       <dir>/w<worker>_p<part>.npz    one shard: per-sample arrays
           x (n,13,9,9) uint8 codes | y (n,) teacher action, turned frame | q (n,6) float32 teacher action values,
           turned frame (only if the teacher has action_values) | game (n,) game index (map seed = seed0 + game) |
           step | player | act (the action actually played, turned frame) | outcome (+1 win, 0 draw, -1 loss for
           that player) | togo (steps until the game ended, >= 1)
         and per-game arrays g_index, g_seed, g_pair, g_swap, g_winner (-1 = draw), g_steps, g_samples.
"""

from __future__ import annotations

import json
import os
import random
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_SEED0 = 1_000_000  # evaluation maps start here: training data must only use map seeds below it
FORMAT = "arena.bc/1"
FLIP = (0, 2, 1, 4, 3, 5)  # the turned frame of player 1: UP<->DOWN, LEFT<->RIGHT (its own inverse)
FLIP_IDX = np.array(FLIP, dtype=np.int64)
ACTION_NAMES = ("STAY", "UP", "DOWN", "LEFT", "RIGHT", "BOMB")

# ----------------------------------------------------------------------------------------------------------------
# 1. exact uint8 codec
# ----------------------------------------------------------------------------------------------------------------
LUT = np.zeros(256, dtype=np.float32)
# preprocess computes a python float (double) 1/n and stores it in a float32 array: the same rounding here
LUT[1:255] = np.array([1 / n for n in range(1, 255)], dtype=np.float64).astype(np.float32)
LUT[255] = np.float32(2 / 3)
_ORDER = np.argsort(LUT, kind="stable")
_SORTED = LUT[_ORDER]
assert len(np.unique(LUT)) == 256


def encode_planes(x: np.ndarray) -> np.ndarray:
    """float32 planes (any shape) -> uint8 codes of the same shape. Exact or it raises:
    np.array_equal(decode_planes(encode_planes(x)), x) is checked for every call."""
    x = np.asarray(x)
    if x.dtype != np.float32:
        raise TypeError(f"planes must be float32, got {x.dtype}")
    idx = np.searchsorted(_SORTED, x.ravel())
    np.minimum(idx, 255, out=idx)
    codes = _ORDER[idx].astype(np.uint8).reshape(x.shape)
    if not np.array_equal(LUT[codes], x):
        bad = np.unique(x[LUT[codes] != x])[:10]
        raise ValueError(f"plane values outside the code table (cannot be stored exactly): {bad.tolist()}")
    return codes


def decode_planes(codes: np.ndarray) -> np.ndarray:
    return LUT[codes]


def torch_lut(device):
    """The lookup table as a torch tensor: x = lut[codes.long()] is bit-identical to the float32 planes."""
    import torch

    return torch.from_numpy(LUT.copy()).to(device)


# ----------------------------------------------------------------------------------------------------------------
# 2. pairs
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Pair:
    a: str          # resolved harness spec of the first actor
    b: str
    label_a: bool   # label the first actor's states with the teacher
    label_b: bool
    weight: float
    text: str       # as written

    @property
    def key(self) -> str:
        return f"{'+' if self.label_a else ''}{self.a}|{'+' if self.label_b else ''}{self.b}"


def resolve_spec(spec: str, teacher: str | None = None, student: str | None = None) -> str:
    """Replace the tokens T / S / S@sample (whole ':'-separated fields only)."""
    out = []
    for field in spec.strip().split(":"):
        if field == "T":
            if not teacher:
                raise ValueError(f"{spec!r} uses T but no teacher was given")
            out.append(teacher)
        elif field in ("S", "S@sample"):
            if not student:
                raise ValueError(f"{spec!r} uses S but no student folder was given (--student)")
            out.append(f"net:{student}{field[1:]}")
        else:
            out.append(field)
    return ":".join(out)


def parse_pairs(text: str, teacher: str | None = None, student: str | None = None) -> list[Pair]:
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        body, star, w = item.rpartition("*")
        if star:
            weight = float(w)
        else:
            body, weight = item, 1.0
        sides = [s.strip() for s in body.split("|")]
        if len(sides) != 2 or not all(sides):
            raise ValueError(f"bad pair {item!r}: expected  actor|actor[*weight]")
        if not weight > 0:
            raise ValueError(f"bad weight in {item!r}")
        marks = [s.startswith("+") for s in sides]
        specs = [resolve_spec(s.lstrip("+"), teacher, student) for s in sides]
        if not any(marks):
            marks = [True, True]
        pairs.append(Pair(specs[0], specs[1], marks[0], marks[1], weight, item))
    if not pairs:
        raise ValueError("no pairs given")
    return pairs


def plan_games(pairs: list[Pair], games: int, seed0: int):
    """Which pair plays game g and whether its actors swap sides: (pair index (games,), swap (games,)).
    Exact shares (largest remainder), shuffled; deterministic in (pairs, games, seed0)."""
    w = np.array([p.weight for p in pairs], dtype=np.float64)
    quota = w / w.sum() * games
    counts = np.floor(quota).astype(np.int64)
    order = np.argsort(-(quota - counts), kind="stable")
    counts[order[: games - int(counts.sum())]] += 1
    idx = [i for i, c in enumerate(counts) for _ in range(int(c))]
    rng = random.Random(zlib.crc32(f"plan|{seed0}|{games}|{[p.key for p in pairs]}|{w.tolist()}".encode()))
    rng.shuffle(idx)
    swap = [rng.random() < 0.5 for _ in range(games)]
    return np.array(idx, dtype=np.int16), np.array(swap, dtype=bool)


def pairs_as_json(pairs: list[Pair]) -> list[dict]:
    return [asdict(p) for p in pairs]


# ----------------------------------------------------------------------------------------------------------------
# 3. datasets
# ----------------------------------------------------------------------------------------------------------------
def save_npz_atomic(path: Path, **arrays):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def read_manifest(folder) -> dict:
    """The manifest of a finished dataset (raises if the folder holds no finished dataset)."""
    path = Path(folder) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{folder}: no manifest.json (the generation did not finish)")
    m = json.loads(path.read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"{folder}: unknown dataset format {m.get('format')!r}")
    return m


def iter_shards(folder):
    """Yield every shard of a dataset as a dict of numpy arrays."""
    m = read_manifest(folder)
    for s in m["shards"]:
        with np.load(Path(folder) / s["file"]) as z:
            yield {k: z[k] for k in z.files}
