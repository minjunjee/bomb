"""Random maps. Every map is 9x9 with walls all around and the same after a 180° turn, so neither side has an
advantage; A starts at (1, 1) and B at (7, 7) as on the default map.

    from bombcompetition import random_rules
    game = Game(random_rules(seed=3))     # the same seed gives the same map
    env = BomberEnv(opponent, rules=random_rules())   # a new random map
"""

import random
from collections import deque
from dataclasses import replace

from .game import RULES, Rules, blast_cells

SIZE = 9
A, B = (1, 1), (SIZE - 2, SIZE - 2)


def _mirror(cell):
    return SIZE - 1 - cell[0], SIZE - 1 - cell[1]


def _floor_cells(walls):
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if not walls[r][c]]


def _bfs(walls, start, blocked=frozenset()):
    seen, queue = {start: 0}, deque([start])
    while queue:
        r, c = queue.popleft()
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not walls[n[0]][n[1]] and n not in seen and n not in blocked:
                seen[n] = seen[(r, c)] + 1
                queue.append(n)
    return seen


def _can_escape_own_bomb(walls, cell, rules):
    """From `cell`, after dropping a bomb there, is a cell outside its blast reachable in time?"""
    import numpy as np
    hit = blast_cells(np.array(walls), cell, rules.radius)
    dist = _bfs(walls, cell)
    return any(c not in hit and d <= rules.fuse - 1 for c, d in dist.items())


def _ok(walls, rules):
    floor = _floor_cells(walls)
    if not 28 <= len(floor) <= 44:
        return False
    reach = _bfs(walls, A)
    if len(reach) != len(floor) or reach.get(B, 0) < 8:  # one connected area, players not too close
        return False
    # Each start has room to dodge its own first bomb, so nobody is stuck in the opening.
    return all(_can_escape_own_bomb(walls, s, rules) for s in (A, B))


def random_layout(seed=None, rules: Rules = RULES) -> tuple[str, ...]:
    """A random symmetric 9x9 layout ('#' wall, '.' floor, 'A'/'B' starts)."""
    rng = random.Random(seed)
    inner = [(r, c) for r in range(1, SIZE - 1) for c in range(1, SIZE - 1)]
    pairs = [cell for cell in inner if cell <= _mirror(cell)]  # one cell of each mirrored pair
    keep_open = {A, B, (1, 2), (2, 1), _mirror((1, 2)), _mirror((2, 1))}
    for _ in range(100_000):
        density = rng.uniform(0.2, 0.36)
        walls = [[True] * SIZE] + [[True] + [False] * (SIZE - 2) + [True] for _ in range(SIZE - 2)] + [[True] * SIZE]
        for cell in pairs:
            if cell not in keep_open and _mirror(cell) not in keep_open and rng.random() < density:
                for r, c in (cell, _mirror(cell)):
                    walls[r][c] = True
        if _ok(walls, rules):
            rows = [["#" if w else "." for w in row] for row in walls]
            rows[A[0]][A[1]], rows[B[0]][B[1]] = "A", "B"
            return tuple("".join(row) for row in rows)
    raise RuntimeError("no valid map found")


def random_rules(seed=None, rules: Rules = RULES) -> Rules:
    """`rules` with a random symmetric map."""
    return replace(rules, layout=random_layout(seed, rules), starts=None)
