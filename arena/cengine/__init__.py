"""The BOMB game engine and the fixed network preprocess in C, exactly equal to bombcompetition (see api.py).

    from arena.cengine import CGame, CBatch, features

Importing this package builds libbomb.so when it is missing or out of date (build.py; safe with many processes).
    python -m arena.cengine.build        build by hand
    python -m arena.cengine.test_equiv   differential tests against the reference engine
    python -m arena.cengine.bench        speed numbers
"""

import warnings as _warnings

# `python -m arena.cengine.build` imports this package first, which already imported .build: harmless, keep it quiet.
_warnings.filterwarnings("ignore", message=r"'arena\.cengine\.build' found in sys\.modules", category=RuntimeWarning)

from .api import (  # noqa: E402,F401
    BOMB_CAP, CELLS, FLAG_CHAIN, FLAG_COLLISION, FLAG_DONE, FLAG_DROP0, FLAG_DROP1, FLAG_EXPLOSION, FLAG_HIT0,
    FLAG_HIT1, FLAG_KILLED0, FLAG_KILLED1, FLAG_NEW_ROUND, FLAG_SHRINK, FLAG_SHRINK_BOMB, FLAG_TELEPORT0,
    FLAG_TELEPORT1, N_CHANNELS, STATE_DTYPE, STATE_SIZE, BombItem, BombRules, BombState, BombStepInfo, CBatch, CGame,
    build_id, cell_of, features, inv32, layouts_to_arrays,
)

__all__ = ["CGame", "CBatch", "features", "layouts_to_arrays", "STATE_DTYPE", "STATE_SIZE", "BombState", "BombItem",
           "BombRules", "BombStepInfo", "CELLS", "cell_of", "inv32", "build_id", "BOMB_CAP", "N_CHANNELS",
           "FLAG_HIT0", "FLAG_HIT1", "FLAG_KILLED0", "FLAG_KILLED1", "FLAG_DONE", "FLAG_SHRINK", "FLAG_TELEPORT0",
           "FLAG_TELEPORT1", "FLAG_CHAIN", "FLAG_COLLISION", "FLAG_NEW_ROUND", "FLAG_DROP0", "FLAG_DROP1",
           "FLAG_EXPLOSION", "FLAG_SHRINK_BOMB"]
