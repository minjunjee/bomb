"""arena.cengine: the game engine and the fixed network preprocess in C (bomb.c), exactly equal to the reference.

    from arena.cengine import CGame, CBatch, features, layouts_to_arrays

Equivalence with bombcompetition/game.py + neural.py is checked step by step in `python -m arena.cengine.test_equiv`
(positions, bombs in obs order, walls, lives, kills, t, done, deaths with killers, both players' input tensors).
The engine supports FIXED fuses only (the league rules; Rules.fuse_max > fuse raises NotImplementedError), 9x9 maps
only, and Rules.max_bombs <= 8.  Cells in Python are (row, col) tuples as in the reference; inside the C state a
cell is the byte row * 9 + col.

One game: `CGame(rules=RULES, seed=None)` mirrors `bombcompetition.Game` (seed is accepted and ignored: no random fuses)
    step(a0, a1) -> list[Death]       RuntimeError when the game is over, ValueError for an action outside 0..5
    obs(p) -> dict                    the same keys / order / types as Game.obs: walls is a bool (9, 9) ndarray copy,
                                      cells are tuples of Python ints, bombs a tuple of {"pos","owner","age","fuse"}
                                      dicts in the reference order
    done, t, lives, kills, pos, walls, bombs ({cell: Bomb}), starts, rules, size, last_blast, last_pos, winner()
                                      read-only views of the state (lists / arrays are fresh copies: writing into
                                      them does not change the game; `walls` is a read-only array)
    clone() -> CGame                  an independent copy (one memcpy)
    CGame.from_game(game) -> CGame    copy a reference Game (or a CGame) in the middle of a game
    to_game() -> Game                 the other way round
    features(p, rules=None)           = features(game, p)
    legal_actions(p) -> list[Action]  = bombcompetition.legal_actions(obs(p)) (STAY, BOMB, then the open moves)
    legal_mask(p) / effective_mask(p) -> int bit masks (bit a = action a); `effective` has BOMB only when a bomb
                                      would really be dropped, so its actions all have different effects
    chain_fuses() -> list[int]        per bomb in obs order: the step at whose end it explodes, chains included
    danger_times() -> uint8 (9, 9)    per cell: the earliest chain-adjusted fuse of a bomb covering it, 0 = none
                                      (the "danger" input plane is 1 / this; the shrink is ignored like there)
    blast_times() -> uint8 (9, 9)     EXACT per cell: k if the cell is inside an explosion at the end of the k-th step
                                      from now (1 = the step being decided) and not before, 0 = never; plays the
                                      bombs forward with chains and the sudden-death shrink, assuming nobody drops
                                      another bomb. blast_masks() -> uint64 (9, 9): bit k-1 = explodes at step k.
    state                             the ctypes BombState (shares memory with the game)

Network input
    features(game_or_state, player, rules=None, out=None) -> float32 (13, 9, 9)
        np.array_equal (and bit-equal) to neural.preprocess(neural.turn(game.obs(player))).numpy(). Like that
        reference call it uses the LEAGUE constants (radius 2, max_bombs 3, shrink_every 100, max_steps 300) whatever
        the game's own rules are; pass rules=some_rules to get preprocess(turn(obs), some_rules).
        game_or_state: a CGame, a reference Game (converted on the fly: slow), a BombState, or one element of a
        CBatch.states array (states[i:i+1]).

Many games at once: `CBatch(n, rules=RULES, threads=1)` holds n states in ONE contiguous numpy buffer
    states                            structured array (n,) of STATE_DTYPE = the C struct BombState; fields grid
                                      (bit 0 wall, bit 1 bomb), n_bombs, done, pos, starts, bombs (cell, owner, fuse,
                                      age), t, lives, kills, rules. Reading fields is free; write only if you know
                                      the invariants (bomb.h).
    reset(layouts, idx=None)          layouts: a sequence of layouts (tuples of 9 strings, e.g. random_layout(seed))
                                      or Rules objects (their layout / starts), one per state in idx (default: all);
                                      a single layout / Rules is used for all of them. Maps come from the Python map
                                      generator; the other rules are the batch's rules.
    reset_arrays(walls, starts=None, idx=None)   the same from layouts_to_arrays(layouts) output (no parsing)
    step(actions, threads=None, active=None, return_flags=False) -> hit (n, 2) bool, done (n,) bool, winner (n,) int8
                                      actions: (n, 2) ints in 0..5. winner = 0 / 1 / -1 (-1: equal lives, a draw once
                                      done). A state that is already done (or active[i] false) is NOT stepped and
                                      reports no hit: auto-reset is the caller's job. flags: the C BOMB_FLAG_* bits.
    features(players, out=None, threads=None, rules=None) -> float32 (n, 13, 9, 9)   players: (n,) of 0/1, or 0 / 1
    legal_masks(effective=False, threads=None) -> bool (n, 2, 6)
    t, done, lives, kills, pos, walls, winner   arrays over the batch;  game(i) -> CGame copy;  set_game(i, game)
    `threads` = OpenMP threads for that call (default: the constructor's). threads=1 never enters OpenMP, which is
    what a forked worker process should use (libgomp thread pools do not survive fork()).

Helpers: layouts_to_arrays(layouts) -> (walls uint8 (k, 81), starts uint8 (k, 2)); CELLS[cell] -> (row, col);
cell_of((row, col)) -> int; FLAG_* (bits returned by the C step); inv32(k) (the float32 the preprocess stores for 1/k).
C users: see bomb.h (bomb_init / bomb_step / memcpy / bomb_effective_mask / bomb_blast_times / bomb_features).
"""

from __future__ import annotations

import ctypes
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bombcompetition.game import RULES, Action, Bomb, Death, Game, Rules  # noqa: E402

from . import build as _build  # noqa: E402

N = 9
N_CELLS = 81
BOMB_CAP = 16
N_CHANNELS = 13
CELLS = tuple((r, c) for r in range(N) for c in range(N))  # cell index -> (row, col)

FLAG_HIT0, FLAG_HIT1, FLAG_KILLED0, FLAG_KILLED1 = 0x1, 0x2, 0x4, 0x8
FLAG_DONE, FLAG_SHRINK, FLAG_TELEPORT0, FLAG_TELEPORT1 = 0x10, 0x20, 0x40, 0x80
FLAG_CHAIN, FLAG_COLLISION, FLAG_NEW_ROUND, FLAG_DROP0, FLAG_DROP1 = 0x100, 0x200, 0x400, 0x800, 0x1000
FLAG_EXPLOSION, FLAG_SHRINK_BOMB = 0x2000, 0x4000


def cell_of(rc) -> int:
    return int(rc[0]) * N + int(rc[1])


# ----------------------------------------------------------------------------------------------------------------
# the C structs (bomb.h)
# ----------------------------------------------------------------------------------------------------------------
class BombItem(ctypes.Structure):
    _fields_ = [("cell", ctypes.c_uint8), ("owner", ctypes.c_uint8), ("fuse", ctypes.c_int16), ("age", ctypes.c_int16)]


class BombRules(ctypes.Structure):
    _fields_ = [("fuse", ctypes.c_int32), ("radius", ctypes.c_int32), ("max_bombs", ctypes.c_int32),
                ("lives", ctypes.c_int32), ("shrink_every", ctypes.c_int32), ("max_steps", ctypes.c_int32)]


class BombState(ctypes.Structure):
    _fields_ = [("grid", ctypes.c_uint8 * N_CELLS), ("n_bombs", ctypes.c_uint8), ("done", ctypes.c_uint8),
                ("pos", ctypes.c_uint8 * 2), ("starts", ctypes.c_uint8 * 2), ("pad_", ctypes.c_uint8),
                ("bombs", BombItem * BOMB_CAP), ("t", ctypes.c_int32), ("lives", ctypes.c_int32 * 2),
                ("kills", ctypes.c_int32 * 2), ("rules", BombRules)]


class BombStepInfo(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_int32), ("hit_pos", ctypes.c_uint8 * 2), ("n_exploded", ctypes.c_uint8),
                ("n_blast", ctypes.c_uint8), ("blast", ctypes.c_uint8 * N_CELLS)]


_lib = ctypes.CDLL(str(_build.ensure_built()))
_vp, _ci = ctypes.c_void_p, ctypes.c_int


def _fn(name, restype, *argtypes):
    f = getattr(_lib, name)
    f.restype, f.argtypes = restype, list(argtypes)
    return f


_init = _fn("bomb_init", _ci, _vp, _vp, _ci, _ci, _vp)
_load = _fn("bomb_load", _ci, _vp, _vp, _vp, _vp, _ci, _vp)
_step_info = _fn("bomb_step_info", _ci, _vp, _ci, _ci, _vp)
_legal = _fn("bomb_legal_mask", _ci, _vp, _ci)
_effective = _fn("bomb_effective_mask", _ci, _vp, _ci)
_chain_fuses = _fn("bomb_chain_fuses", _ci, _vp, _vp)
_danger_times = _fn("bomb_danger_times", None, _vp, _vp)
_blast_times = _fn("bomb_blast_times", _ci, _vp, _vp, _vp)
_features = _fn("bomb_features", None, _vp, _ci, _vp)
_features_rules = _fn("bomb_features_rules", None, _vp, _ci, _vp, _vp)
_reset_batch = _fn("bomb_reset_batch", _ci, _vp, _vp, _ci, _vp, _vp, _vp)
_step_batch = _fn("bomb_step_batch", _ci, _vp, _ci, _vp, _vp, _vp, _vp, _vp, _vp, _ci)
_features_batch = _fn("bomb_features_batch", None, _vp, _ci, _vp, _vp, _ci)
_features_rules_batch = _fn("bomb_features_rules_batch", None, _vp, _ci, _vp, _vp, _vp, _ci)
_legal_batch = _fn("bomb_legal_batch", None, _vp, _ci, _vp, _ci, _ci)
_inv = _fn("bomb_inv", ctypes.c_float, _ci)
_layout = _fn("bomb_layout", _ci, _vp, _ci)
_bench = _fn("bomb_bench", ctypes.c_uint64, _vp, ctypes.c_int64, ctypes.c_uint64, _ci, _ci, _vp)
_build_id = _fn("bomb_build_id", ctypes.c_char_p)


def _check_layout():
    """The ctypes mirrors above must have exactly the layout the compiler gave the C structs."""
    got = (ctypes.c_int32 * 32)()
    n = _layout(got, 32)
    S, I = BombState, BombStepInfo
    want = [ctypes.sizeof(S), ctypes.sizeof(BombItem), ctypes.sizeof(BombRules), ctypes.sizeof(I),
            S.grid.offset, S.n_bombs.offset, S.done.offset, S.pos.offset, S.starts.offset, S.bombs.offset,
            S.t.offset, S.lives.offset, S.kills.offset, S.rules.offset,
            I.flags.offset, I.hit_pos.offset, I.n_exploded.offset, I.n_blast.offset, I.blast.offset]
    if n != len(want) or list(got[:n]) != want:
        raise ImportError(f"arena.cengine: struct layout mismatch between api.py and libbomb.so: {list(got[:n])} != {want}")


_check_layout()

STATE_SIZE = ctypes.sizeof(BombState)
STATE_DTYPE = np.dtype(BombState)  # the numpy view of one C state (CBatch.states)
assert STATE_DTYPE.itemsize == STATE_SIZE
_StateBytes = ctypes.c_ubyte * STATE_SIZE
_O_NB, _O_DONE, _O_POS, _O_BOMBS = BombState.n_bombs.offset, BombState.done.offset, BombState.pos.offset, BombState.bombs.offset
_O_T, _O_LIVES, _O_KILLS = BombState.t.offset, BombState.lives.offset, BombState.kills.offset
assert _O_LIVES == _O_T + 4 and _O_KILLS == _O_T + 12
_BOMB = struct.Struct("<BBhh")  # cell, owner, fuse, age
_BOMB_SIZE = _BOMB.size
assert _BOMB_SIZE == ctypes.sizeof(BombItem)
_T_LIVES = struct.Struct("<3i")  # t, lives0, lives1
_I2 = struct.Struct("<2i")
_I1 = struct.Struct("<i")

_DEATHS = []  # indexed by flags & 15
for _f in range(16):
    _d = []
    if _f & FLAG_HIT0:
        _d.append(Death(0, 1 if _f & FLAG_KILLED0 else None))
    if _f & FLAG_HIT1:
        _d.append(Death(1, 0 if _f & FLAG_KILLED1 else None))
    _DEATHS.append(tuple(_d))
_LEGAL = [[Action.STAY, Action.BOMB] + [Action(a) for a in (1, 2, 3, 4) if m >> a & 1] for m in range(64)]


def inv32(k: int) -> np.float32:
    """The float32 the reference preprocess stores for the Python float 1 / k."""
    return np.float32(_inv(int(k)))


def build_id() -> str:
    return _build_id().decode()


# ----------------------------------------------------------------------------------------------------------------
# rules and layouts
# ----------------------------------------------------------------------------------------------------------------
_RULES_CACHE: dict = {}
_LAYOUT_CACHE: dict = {}


def _c_rules(rules: Rules) -> BombRules:
    """The engine parameters of `rules` as a C struct."""
    key = (rules.fuse, rules.fuse_max, rules.radius, rules.max_bombs, rules.lives, rules.shrink_every, rules.max_steps)
    r = _RULES_CACHE.get(key)
    if r is None:
        if rules.fuse_max is not None and rules.fuse_max > rules.fuse:
            raise NotImplementedError("arena.cengine has fixed fuses only (Rules.fuse_max > Rules.fuse is not supported)")
        if rules.max_bombs > BOMB_CAP // 2:
            raise ValueError(f"arena.cengine supports max_bombs <= {BOMB_CAP // 2}")
        if not -32000 <= rules.fuse <= 32000:
            raise ValueError("arena.cengine: fuse out of range")
        every = rules.shrink_every or 0
        r = BombRules(int(rules.fuse), int(rules.radius), int(rules.max_bombs), int(rules.lives),
                      max(0, int(every)), int(rules.max_steps))  # a negative shrink_every never shrinks in the reference
        _RULES_CACHE[key] = r
    return r


def _c_feature_rules(rules: Rules) -> BombRules:
    """The parameters preprocess(obs, rules) reads: radius, max_bombs, shrink_every, max_steps."""
    every = rules.shrink_every or 0
    if every < 0:
        raise ValueError("features: a negative shrink_every is not supported")
    if rules.max_bombs == 0:
        raise ZeroDivisionError("features: rules.max_bombs is 0 (the reference preprocess divides by it)")
    return BombRules(int(rules.fuse), int(rules.radius), int(rules.max_bombs), int(rules.lives), int(every),
                     int(rules.max_steps))


def _parse_layout(layout, starts=None):
    """(walls as 81 bytes, (start0, start1) cells, walls as a read-only bool (9, 9) array); the reference checks."""
    try:
        key = (tuple(layout), None if starts is None else tuple(map(tuple, starts)))
        hit = _LAYOUT_CACHE.get(key)
    except TypeError:
        key, hit = None, None
    if hit is not None:
        return hit
    size = len(layout)
    walls = np.array([[ch == "#" for ch in row] for row in layout])
    find = {ch: (r, row.index(ch)) for r, row in enumerate(layout) for ch in "AB" if ch in row}
    st = list(starts or (find.get("A"), find.get("B")))
    if walls.shape != (size, size) or not (walls[[0, -1]].all() and walls[:, [0, -1]].all()):
        raise ValueError("the map must be square with walls all around")
    if None in st or st[0] == st[1] or any(walls[tuple(s)] for s in st):
        raise ValueError(f"bad start cells: {st}")
    if size != N:
        raise ValueError("arena.cengine supports 9x9 maps only")
    walls.flags.writeable = False
    out = (walls.astype(np.uint8).tobytes(), (cell_of(st[0]), cell_of(st[1])), walls)
    if key is not None:
        if len(_LAYOUT_CACHE) > 200_000:
            _LAYOUT_CACHE.clear()
        _LAYOUT_CACHE[key] = out
    return out


def layouts_to_arrays(layouts) -> tuple[np.ndarray, np.ndarray]:
    """layouts: a sequence of layouts (9 strings each) or Rules -> (walls uint8 (k, 81), starts uint8 (k, 2))."""
    walls = np.empty((len(layouts), N_CELLS), np.uint8)
    starts = np.empty((len(layouts), 2), np.uint8)
    for i, lay in enumerate(layouts):
        w, s, _ = _parse_layout(lay.layout, lay.starts) if isinstance(lay, Rules) else _parse_layout(lay)
        walls[i] = np.frombuffer(w, np.uint8)
        starts[i] = s
    return walls, starts


# ----------------------------------------------------------------------------------------------------------------
# one game
# ----------------------------------------------------------------------------------------------------------------
class CGame:
    """bombcompetition.Game on the C engine (see the module docstring). Players are 0 (A) and 1 (B)."""

    __slots__ = ("rules", "size", "starts", "_buf", "_c", "_p", "_info", "_ip", "_walls")

    def __init__(self, rules: Rules = RULES, seed=None):
        walls, starts, wall_array = _parse_layout(rules.layout, rules.starts)
        crules = _c_rules(rules)
        self._alloc()
        if _init(self._p, walls, starts[0], starts[1], ctypes.addressof(crules)) != 0:
            raise ValueError("bomb_init rejected the map / rules")
        self.rules, self.size = rules, N
        self.starts = [CELLS[starts[0]], CELLS[starts[1]]]
        self._info.hit_pos[0], self._info.hit_pos[1] = starts
        self._walls = wall_array

    def _alloc(self):
        self._buf = bytearray(STATE_SIZE)
        self._c = _StateBytes.from_buffer(self._buf)
        self._p = ctypes.addressof(self._c)
        self._info = BombStepInfo()
        self._ip = ctypes.addressof(self._info)

    # -- construction from / conversion to other representations ---------------------------------------------------
    @classmethod
    def from_game(cls, game) -> "CGame":
        """A copy of a reference `Game` (or of a CGame) as it is now, in the middle of a game."""
        if isinstance(game, CGame):
            return game.clone()
        rules = game.rules
        crules = _c_rules(rules)
        walls = np.asarray(game.walls)
        if walls.shape != (N, N):
            raise ValueError("arena.cengine supports 9x9 maps only")
        wall_array = walls.astype(bool)
        v = struct.pack("<10i", cell_of(game.pos[0]), cell_of(game.pos[1]), cell_of(game.starts[0]),
                        cell_of(game.starts[1]), int(game.t), int(game.lives[0]), int(game.lives[1]),
                        int(game.kills[0]), int(game.kills[1]), int(bool(game.done)))
        flat = []
        for (r, c), b in game.bombs.items():
            flat += (int(r) * N + int(c), int(b.owner), int(b.fuse), int(b.age))
        bombs = struct.pack(f"<{len(flat)}i", *flat)
        self = cls.__new__(cls)
        self._alloc()
        if _load(self._p, wall_array.astype(np.uint8).tobytes(), v, bombs, len(flat) // 4, ctypes.addressof(crules)) != 0:
            raise ValueError("arena.cengine cannot represent this game state (border not wall, a player or bomb on "
                             "the border, more than 8 bombs of one player, ...)")
        self.rules, self.size = rules, N
        self.starts = [tuple(map(int, s)) for s in game.starts]
        for r, c in game.last_blast:
            self._info.blast[r * N + c] = 1
        self._info.n_blast = len(game.last_blast)
        self._info.hit_pos[0], self._info.hit_pos[1] = cell_of(game.last_pos[0]), cell_of(game.last_pos[1])
        wall_array.flags.writeable = False
        self._walls = wall_array
        return self

    @classmethod
    def from_state(cls, state, rules: Rules = RULES) -> "CGame":
        """A copy of a raw C state: a BombState, or one element of CBatch.states (states[i:i+1]).
        `rules` is only kept as the .rules attribute (the engine parameters live inside the state)."""
        ptr, keep = _state_ptr(state)
        self = cls.__new__(cls)
        self._alloc()
        ctypes.memmove(self._p, ptr, STATE_SIZE)
        del keep
        self.rules, self.size = rules, N
        s = self._buf[BombState.starts.offset:BombState.starts.offset + 2]
        self.starts = [CELLS[s[0]], CELLS[s[1]]]
        self._info.hit_pos[0], self._info.hit_pos[1] = self._buf[_O_POS], self._buf[_O_POS + 1]
        self._walls = None
        return self

    def clone(self) -> "CGame":
        """An independent copy: stepping one does not touch the other."""
        g = CGame.__new__(CGame)
        g.rules, g.size, g.starts = self.rules, self.size, list(self.starts)
        g._buf = bytearray(self._buf)
        g._c = _StateBytes.from_buffer(g._buf)
        g._p = ctypes.addressof(g._c)
        g._info = BombStepInfo.from_buffer_copy(self._info)
        g._ip = ctypes.addressof(g._info)
        g._walls = self._walls  # read-only and replaced (never modified) at a shrink, so it can be shared
        return g

    __copy__ = clone

    def __deepcopy__(self, memo):
        return self.clone()

    def to_game(self) -> Game:
        """A reference Game in the same state."""
        g = Game(self.rules)
        g.walls = self.walls.copy()
        g.starts = list(self.starts)
        g.pos, g.bombs, g.lives, g.kills, g.t, g.done = self.pos, self.bombs, self.lives, self.kills, self.t, self.done
        g.last_blast, g.last_pos = self.last_blast, self.last_pos
        return g

    # -- the Game interface ----------------------------------------------------------------------------------------
    def step(self, a0: int, a1: int) -> list:
        """Play one step and return who died in it (a list of bombcompetition.Death)."""
        if type(a0) is not int or type(a1) is not int or not (0 <= a0 <= 5 and 0 <= a1 <= 5):
            if self._buf[_O_DONE]:
                raise RuntimeError("the game is over")
            a0, a1 = int(Action(a0)), int(Action(a1))  # ValueError for an invalid action, like the reference
        f = _step_info(self._p, a0, a1, self._ip)
        if f & 0x2F:  # a hit, a shrink, or an error (negative)
            if f < 0:
                if f == -1:
                    raise RuntimeError("the game is over")
                raise RuntimeError(f"bomb_step failed with code {f}")
            if f & FLAG_SHRINK:
                self._walls = None
            return list(_DEATHS[f & 15])
        return []

    def obs(self, p: int) -> dict:
        """What player p sees: the same dict as Game.obs(p)."""
        if p != 0 and p != 1:
            raise IndexError("player must be 0 or 1")
        buf = self._buf
        n = buf[_O_NB]
        bombs = tuple([{"pos": CELLS[c], "owner": o, "age": a, "fuse": f} for c, o, f, a in
                       _BOMB.iter_unpack(buf[_O_BOMBS:_O_BOMBS + _BOMB_SIZE * n])]) if n else ()
        t, l0, l1 = _T_LIVES.unpack_from(buf, _O_T)
        return {"player": p, "walls": self.walls.copy(), "me": CELLS[buf[_O_POS + p]], "opp": CELLS[buf[_O_POS + 1 - p]],
                "bombs": bombs, "lives": (l1, l0) if p else (l0, l1), "step": t}

    def winner(self):
        """The player with more lives left, None for a draw."""
        l0, l1 = _I2.unpack_from(self._buf, _O_LIVES)
        return None if l0 == l1 else int(l1 > l0)

    @property
    def done(self) -> bool:
        return self._buf[_O_DONE] != 0

    @property
    def t(self) -> int:
        return _I1.unpack_from(self._buf, _O_T)[0]

    @property
    def lives(self) -> list:
        return list(_I2.unpack_from(self._buf, _O_LIVES))

    @property
    def kills(self) -> list:
        return list(_I2.unpack_from(self._buf, _O_KILLS))

    @property
    def pos(self) -> list:
        return [CELLS[self._buf[_O_POS]], CELLS[self._buf[_O_POS + 1]]]

    @property
    def walls(self) -> np.ndarray:
        """bool (9, 9), read-only (obs()["walls"] is a writable copy)."""
        w = self._walls
        if w is None:
            w = (np.frombuffer(self._buf, np.uint8, N_CELLS) & 1).astype(bool).reshape(N, N)
            w.flags.writeable = False
            self._walls = w
        return w

    @property
    def bombs(self) -> dict:
        """{cell: Bomb(owner, fuse, age)} in the reference (insertion) order."""
        buf = self._buf
        return {CELLS[c]: Bomb(o, f, a) for c, o, f, a in
                _BOMB.iter_unpack(buf[_O_BOMBS:_O_BOMBS + _BOMB_SIZE * buf[_O_NB]])}

    @property
    def bomb_list(self) -> list:
        """[(cell, owner, fuse, age)] in obs order."""
        buf = self._buf
        return [(CELLS[c], o, f, a) for c, o, f, a in _BOMB.iter_unpack(buf[_O_BOMBS:_O_BOMBS + _BOMB_SIZE * buf[_O_NB]])]

    @property
    def last_blast(self) -> set:
        """Cells that exploded in the last step."""
        return {CELLS[i] for i, v in enumerate(bytes(self._info.blast)) if v}

    @property
    def last_pos(self) -> list:
        """Positions at the end of the last step, before a new round."""
        return [CELLS[self._info.hit_pos[0]], CELLS[self._info.hit_pos[1]]]

    @property
    def last_flags(self) -> int:
        """The FLAG_* bits of the last step."""
        return self._info.flags

    @property
    def random_fuse(self) -> bool:
        return False

    @property
    def state(self) -> BombState:
        """The C state as a ctypes struct sharing memory with this game."""
        return BombState.from_buffer(self._buf)

    # -- extras ------------------------------------------------------------------------------------------------------
    def features(self, p: int, rules: Rules | None = None, out: np.ndarray | None = None) -> np.ndarray:
        return features(self, p, rules, out)

    def legal_mask(self, p: int) -> int:
        return _legal(self._p, p)

    def effective_mask(self, p: int) -> int:
        return _effective(self._p, p)

    def legal_actions(self, p: int) -> list:
        return list(_LEGAL[_legal(self._p, p)])

    def chain_fuses(self) -> list:
        out = (ctypes.c_int16 * BOMB_CAP)()
        return list(out[:_chain_fuses(self._p, out)])

    def danger_times(self) -> np.ndarray:
        out = np.empty((N, N), np.uint8)
        _danger_times(self._p, out.ctypes.data)
        return out

    def blast_times(self) -> np.ndarray:
        out = np.empty((N, N), np.uint8)
        _blast_times(self._p, out.ctypes.data, None)
        return out

    def blast_masks(self) -> np.ndarray:
        out = np.empty((N, N), np.uint64)
        _blast_times(self._p, None, out.ctypes.data)
        return out

    def __repr__(self):
        return f"CGame(t={self.t}, pos={self.pos}, bombs={self.bomb_list}, lives={self.lives}, done={self.done})"


def _state_ptr(x):
    """(address of the C state, an object to keep alive while it is used)."""
    if isinstance(x, CGame):
        return x._p, x
    if isinstance(x, BombState):
        return ctypes.addressof(x), x
    if isinstance(x, np.ndarray) and x.dtype == STATE_DTYPE and x.size == 1:
        x = np.ascontiguousarray(x)
        return x.ctypes.data, x
    if isinstance(x, np.void) and x.dtype == STATE_DTYPE:
        a = np.array(x)
        return a.ctypes.data, a
    if isinstance(x, Game) or hasattr(x, "bombs") and hasattr(x, "walls"):
        g = CGame.from_game(x)
        return g._p, g
    raise TypeError(f"not a game or a state: {type(x)!r}")


def features(game, player: int, rules: Rules | None = None, out: np.ndarray | None = None) -> np.ndarray:
    """float32 (13, 9, 9) == bombcompetition.neural.preprocess(turn(game.obs(player))) (bit for bit).
    rules=None: the league constants, like the reference call above; rules=r: preprocess(turn(obs), r)."""
    if player != 0 and player != 1:
        raise IndexError("player must be 0 or 1")
    ptr, keep = _state_ptr(game)
    if out is None:
        out = np.empty((N_CHANNELS, N, N), np.float32)
    elif out.dtype != np.float32 or out.shape != (N_CHANNELS, N, N) or not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous float32 (13, 9, 9) array")
    if rules is None:
        _features(ptr, int(player), out.ctypes.data)
    else:
        fr = _c_feature_rules(rules)
        _features_rules(ptr, int(player), ctypes.addressof(fr), out.ctypes.data)
    del keep
    return out


# ----------------------------------------------------------------------------------------------------------------
# many games in one buffer
# ----------------------------------------------------------------------------------------------------------------
class CBatch:
    """n games in one contiguous buffer, stepped / featurised by C loops (OpenMP with threads > 1).
    See the module docstring. A new batch holds n league-rule games on the default map."""

    def __init__(self, n: int, rules: Rules = RULES, threads: int = 1):
        self.n = int(n)
        self.rules = rules
        self.threads = int(threads)
        self._crules = _c_rules(rules)
        self.states = np.zeros(self.n, STATE_DTYPE)
        self._p = self.states.ctypes.data
        self.reset(rules)

    def _threads(self, threads):
        return self.threads if threads is None else int(threads)

    # -- reset -----------------------------------------------------------------------------------------------------
    def reset(self, layouts, idx=None):
        """Start new games. layouts: one layout / Rules per state in idx (default all states), or a single one."""
        single = isinstance(layouts, Rules) or (len(layouts) > 0 and isinstance(layouts[0], str))
        k = self.n if idx is None else len(idx)
        if single:
            w, s, _ = _parse_layout(layouts.layout, layouts.starts) if isinstance(layouts, Rules) else _parse_layout(layouts)
            walls = np.broadcast_to(np.frombuffer(w, np.uint8), (k, N_CELLS))
            starts = np.broadcast_to(np.array(s, np.uint8), (k, 2))
        else:
            walls, starts = layouts_to_arrays(layouts)
        self.reset_arrays(walls, starts, idx)

    def reset_arrays(self, walls, starts=None, idx=None):
        """walls: (k, 81) or (k, 9, 9), non-zero = wall; starts: (k, 2) cells (default (1, 1) and (7, 7));
        idx: the k states to reset (default: all n)."""
        walls = np.ascontiguousarray(walls, dtype=np.uint8).reshape(-1, N_CELLS)
        k = len(walls)
        if idx is None:
            if k != self.n:
                raise ValueError(f"{k} maps for {self.n} states")
            ip = None
        else:
            idx = np.ascontiguousarray(idx, dtype=np.int32)
            if idx.shape != (k,) or (k and (idx.min() < 0 or idx.max() >= self.n)):
                raise ValueError("idx does not match the maps / is out of range")
            ip = idx.ctypes.data
        if starts is not None:
            starts = np.ascontiguousarray(starts, dtype=np.uint8)
            if starts.shape != (k, 2):
                raise ValueError("starts must be (k, 2) cell indices")
        bad = _reset_batch(self._p, ip, k, walls.ctypes.data, None if starts is None else starts.ctypes.data,
                           ctypes.addressof(self._crules))
        if bad:
            raise ValueError(f"map {bad - 1} is not valid (walls all around, two different open start cells)")

    # -- step / features / masks -----------------------------------------------------------------------------------
    def step(self, actions, threads=None, active=None, return_flags=False):
        """actions (n, 2) in 0..5 -> (hit (n, 2) bool, done (n,) bool, winner (n,) int8[, flags (n,) int32])."""
        a = np.asarray(actions)
        if a.shape != (self.n, 2):
            raise ValueError(f"actions must have shape ({self.n}, 2)")
        if a.dtype != np.uint8:
            if a.dtype.kind not in "iub" or (a.size and (a.min() < 0 or a.max() > 5)):
                raise ValueError("actions must be integers in 0..5")
            a = a.astype(np.uint8)
        elif a.size and a.max() > 5:
            raise ValueError("actions must be integers in 0..5")
        a = np.ascontiguousarray(a)
        hit = np.empty((self.n, 2), np.uint8)
        done = np.empty(self.n, np.uint8)
        winner = np.empty(self.n, np.int8)
        flags = np.empty(self.n, np.int32) if return_flags else None
        if active is not None:
            active = np.ascontiguousarray(active, dtype=np.uint8)
            if active.shape != (self.n,):
                raise ValueError("active must have shape (n,)")
        _step_batch(self._p, self.n, a.ctypes.data, hit.ctypes.data, done.ctypes.data, winner.ctypes.data,
                    None if flags is None else flags.ctypes.data, None if active is None else active.ctypes.data,
                    self._threads(threads))
        out = (hit.view(np.bool_), done.view(np.bool_), winner)
        return (*out, flags) if return_flags else out

    def features(self, players, out=None, threads=None, rules: Rules | None = None) -> np.ndarray:
        """(n, 13, 9, 9) float32: row i = features(state i, players[i]). players: (n,) of 0 / 1, or one int."""
        if np.isscalar(players):
            players = np.full(self.n, int(players), np.uint8)
        else:
            players = np.ascontiguousarray(players, dtype=np.uint8)
        if players.shape != (self.n,) or (self.n and players.max() > 1):
            raise ValueError("players must be (n,) with values 0 / 1")
        if out is None:
            out = np.empty((self.n, N_CHANNELS, N, N), np.float32)
        elif out.dtype != np.float32 or out.shape != (self.n, N_CHANNELS, N, N) or not out.flags.c_contiguous:
            raise ValueError("out must be a C-contiguous float32 (n, 13, 9, 9) array")
        if rules is None:
            _features_batch(self._p, self.n, players.ctypes.data, out.ctypes.data, self._threads(threads))
        else:
            fr = _c_feature_rules(rules)
            _features_rules_batch(self._p, self.n, players.ctypes.data, ctypes.addressof(fr), out.ctypes.data,
                                  self._threads(threads))
        return out

    def legal_masks(self, effective: bool = False, threads=None) -> np.ndarray:
        """(n, 2, 6) bool: [i, p, a] = action a is legal for player p in state i (see CGame.legal_mask)."""
        out = np.empty((self.n, 2, 6), np.uint8)
        _legal_batch(self._p, self.n, out.ctypes.data, int(bool(effective)), self._threads(threads))
        return out.view(np.bool_)

    # -- views -----------------------------------------------------------------------------------------------------
    @property
    def t(self) -> np.ndarray:
        return self.states["t"]

    @property
    def done(self) -> np.ndarray:
        return self.states["done"].view(np.bool_)

    @property
    def lives(self) -> np.ndarray:
        return self.states["lives"]

    @property
    def kills(self) -> np.ndarray:
        return self.states["kills"]

    @property
    def pos(self) -> np.ndarray:
        """(n, 2) cell indices (row * 9 + col)."""
        return self.states["pos"]

    @property
    def walls(self) -> np.ndarray:
        """(n, 9, 9) bool (a copy)."""
        return (self.states["grid"] & 1).astype(bool).reshape(self.n, N, N)

    @property
    def winner(self) -> np.ndarray:
        """(n,) int8: 0 / 1 = the player with more lives, -1 = equal."""
        lives = self.states["lives"]
        return np.where(lives[:, 0] == lives[:, 1], -1, lives[:, 1] > lives[:, 0]).astype(np.int8)

    def game(self, i: int, rules: Rules | None = None) -> CGame:
        """A CGame copy of state i (its .rules attribute: `rules`, default the batch's rules with their own layout)."""
        return CGame.from_state(self.states[i:i + 1], self.rules if rules is None else rules)

    def set_game(self, i: int, game: CGame):
        """Overwrite state i with a copy of `game` (a CGame or a reference Game)."""
        g = game if isinstance(game, CGame) else CGame.from_game(game)
        self.states[i:i + 1] = np.frombuffer(g._buf, STATE_DTYPE)

    def __len__(self):
        return self.n
