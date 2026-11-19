"""Baseline yardstick bot (rule-based, stateless).

1. Never take an action after which it cannot outrun the bombs on the board (a time-aware search, chains included).
2. Drop a bomb when the opponent is inside the blast it would make and it can still get away from it.
3. Otherwise walk toward the opponent.

It ignores bombs the opponent may drop later and that the opponent can block its way. Timing convention used here:
"time k" = the end of the k-th step from now (the step being decided is k = 1); a bomb whose obs fuse is f explodes
at time f, a bomb dropped now explodes at time RULES.fuse.
"""

import random
from collections import deque

from bombcompetition import DELTAS, MOVES, RULES, Action, blast_cells


def explosion_times(walls, bombs: dict) -> dict:
    """{bomb cell: time it explodes}, a bomb inside an earlier blast goes off with it (chain reaction)."""
    t = dict(bombs)
    for _ in range(len(t)):
        for c in t:
            for d in blast_cells(walls, c, RULES.radius):
                if d in t and t[d] > t[c]:
                    t[d] = t[c]
    return t


def survives(walls, start, bombs: dict, blocked=frozenset()) -> bool:
    """Standing on `start` after the first step (time 1) with these bombs (cell -> time): is there a way to never
    be inside an explosion? Bombs block the way until they have exploded. `blocked`: other cells that cannot be entered."""
    if not bombs:
        return True
    times = explosion_times(walls, bombs)
    hit = {}  # time -> cells that explode then
    for c, tc in times.items():
        hit.setdefault(tc, set()).update(blast_cells(walls, c, RULES.radius))
    last = max(times.values())
    if start in hit.get(1, ()):
        return False
    frontier = {start}
    for k in range(2, last + 1):  # the move of step k, then the explosions of time k
        nxt = set()
        for (r, c) in frontier:
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                n = (r + dr, c + dc)
                if n in nxt or walls[n] or n in blocked:
                    continue
                if (dr or dc) and times.get(n, 0) >= k:  # a bomb that has not exploded yet is in the way
                    continue
                if n in hit.get(k, ()):
                    continue
                nxt.add(n)
        if not nxt:
            return False
        frontier = nxt
    return True


def distances(walls, start, blocked) -> dict:
    seen, queue = {start: 0}, deque([start])
    while queue:
        r, c = queue.popleft()
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not walls[n] and n not in blocked and n not in seen:
                seen[n] = seen[(r, c)] + 1
                queue.append(n)
    return seen


class Policy:
    def act(self, obs):
        walls, me, opp = obs["walls"], obs["me"], obs["opp"]
        bombs = {b["pos"]: b["fuse"] for b in obs["bombs"]}
        mine = sum(b["owner"] == obs["player"] for b in obs["bombs"])

        # what each action leads to: (my cell after the step, the bombs then)
        options = {Action.STAY: (me, bombs)}
        for a in MOVES:
            n = (me[0] + DELTAS[a][0], me[1] + DELTAS[a][1])
            if not walls[n] and n not in bombs and n != opp:
                options[a] = (n, bombs)
        if mine < RULES.max_bombs and me not in bombs:
            options[Action.BOMB] = (me, {**bombs, me: RULES.fuse})
        safe = [a for a, (cell, bs) in options.items() if survives(walls, cell, bs, {opp})]
        if not safe:  # try again as if the opponent were not in the way, then give up
            safe = [a for a, (cell, bs) in options.items() if survives(walls, cell, bs)] or [Action.STAY]

        if Action.BOMB in safe and opp in blast_cells(walls, me, RULES.radius):
            return Action.BOMB
        moves = [a for a in safe if a != Action.BOMB]
        if not moves:
            return safe[0]
        if random.random() < 0.1:
            return random.choice(moves)
        dist = distances(walls, opp, set(bombs))  # steps to the opponent from every cell
        best = min(dist.get(options[a][0], 99) for a in moves)
        return random.choice([a for a in moves if dist.get(options[a][0], 99) == best])
