"""Heuristic baseline bot. Read it top to bottom, like bots/simple.py.

What it does, in order:
1. Never takes an action after which it cannot outrun the bombs on the board.
2. Drops a bomb when the opponent is inside its blast and it can still get away itself.
3. Otherwise walks toward the opponent, and steps off the outer ring shortly before the map shrinks.

Use it as a training opponent or in the Local evaluation cell:

    OPPONENT = Policy                                   # the plain bot
    OPPONENT = lambda: Policy(mistake=0.25)             # an easier one: 25% of its moves are random (but never deadly)
    OPPONENT = lambda: Policy(bomb_only_if_trapped=True)  # a harder one: it only bombs when you cannot escape
    env = BomberEnv(opponent=Policy(mistake=0.25).act, player=0, rules=random_rules())
"""

import random

from bombcompetition import DELTAS, MOVES, RULES, Action, blast_cells, legal_actions, ring_of


def explosion_times(walls, fuses):
    """fuses: {bomb cell: steps until it explodes}. A bomb inside another bomb's blast goes off with it
    (chain reaction), so it takes the earlier time. Returns the corrected {bomb cell: steps}."""
    times = dict(fuses)
    changed = True
    while changed:
        changed = False
        for cell, t in times.items():
            for hit in blast_cells(walls, cell, RULES.radius):
                if hit in times and times[hit] > t:
                    times[hit] = t
                    changed = True
    return times


def can_escape(walls, start, times):
    """Standing on `start`, is there a way to walk so that no bomb in `times` ever hits me?
    times: {bomb cell: n}, the bomb explodes at the end of the n-th step from now (1 = the coming step).
    It tries every walk at once: `cells` holds all the cells I could be standing on after each step."""
    if not times:
        return True
    cells = {start}
    for step in range(1, max(times.values()) + 1):
        reached = set()
        for row, col in cells:
            reached.add((row, col))  # staying is always possible
            for action in MOVES:
                target = (row + DELTAS[action][0], col + DELTAS[action][1])
                if walls[target]:
                    continue
                if times.get(target, 0) >= step:  # a bomb is still lying there, and bombs block movement
                    continue
                reached.add(target)
        for bomb_cell, t in times.items():  # players move first, then the bombs of this step explode
            if t == step:
                reached -= blast_cells(walls, bomb_cell, RULES.radius)
        if not reached:
            return False
        cells = reached
    return True


def walking_distance(walls, start, blocked):
    """Breadth-first search: {cell: steps from start}, not entering walls or `blocked` cells."""
    distance, queue = {start: 0}, [start]
    for row, col in queue:
        for action in MOVES:
            target = (row + DELTAS[action][0], col + DELTAS[action][1])
            if not walls[target] and target not in blocked and target not in distance:
                distance[target] = distance[(row, col)] + 1
                queue.append(target)
    return distance


class Policy:
    def __init__(self, mistake=0.0, bomb_only_if_trapped=False):
        # mistake: how often it plays a random action instead of its best one. The random action is still
        # one it survives, so the bot never blows itself up; it just wanders into bad spots.
        # bomb_only_if_trapped: wait with the bomb until the opponent has no way out of it.
        self.mistake = mistake
        self.bomb_only_if_trapped = bomb_only_if_trapped

    def act(self, obs):
        walls = obs["walls"]
        me, opp = obs["me"], obs["opp"]

        # The bombs on the board: where they are and when they explode.
        fuses = {bomb["pos"]: bomb["fuse"] for bomb in obs["bombs"]}
        my_bombs = sum(bomb["owner"] == obs["player"] for bomb in obs["bombs"])

        # Step 1: which actions can I survive? For each action, look at the board right after this step
        # and check that I am not hit now and can still outrun every bomb afterwards.
        safe = {}  # action -> (my cell after the step, the bombs left after the step)
        for action in legal_actions(obs):
            cell, after = me, dict(fuses)
            if action == Action.BOMB:
                if my_bombs >= RULES.max_bombs or me in fuses:
                    continue  # the engine would refuse this bomb
                after[me] = RULES.fuse  # a new bomb starts at 5 and ticks once in this same step
            elif action != Action.STAY:
                cell = (me[0] + DELTAS[action][0], me[1] + DELTAS[action][1])
            # One step passes: every fuse goes down by 1, and bombs at 0 explode.
            times = {bomb_cell: t - 1 for bomb_cell, t in explosion_times(walls, after).items()}
            exploding = set()
            for bomb_cell, t in times.items():
                if t <= 0:
                    exploding |= blast_cells(walls, bomb_cell, RULES.radius)
            if cell in exploding:
                continue
            left = {bomb_cell: t for bomb_cell, t in times.items() if t > 0}
            if can_escape(walls, cell, left):
                safe[action] = (cell, left)

        if not safe:  # nothing saves me any more
            return Action.STAY

        # Sometimes play a random action on purpose (see `mistake` above).
        if random.random() < self.mistake:
            return random.choice(list(safe))

        # Step 2: bomb when the opponent stands inside my blast and I can still get away.
        if Action.BOMB in safe and opp in blast_cells(walls, me, RULES.radius):
            if not self.bomb_only_if_trapped:
                return Action.BOMB
            _, left = safe[Action.BOMB]
            if not can_escape(walls, opp, left):  # with my new bomb on the board, the opponent has no way out
                return Action.BOMB

        # Step 3: walk. Go toward the opponent (fewest steps, walking around walls and bombs).
        to_opp = walking_distance(walls, opp, set(fuses))

        # Sudden death: the ring that turns into walls at the next shrink. A player standing on it is pushed
        # to the nearest open cell inside, which may be a dead end, so leave the ring a little early.
        closing_ring = None
        if RULES.shrink_every:
            steps_left = RULES.shrink_every - obs["step"] % RULES.shrink_every
            if steps_left <= 10 and obs["step"] + steps_left < RULES.max_steps:
                closing_ring = obs["step"] // RULES.shrink_every + 1

        best_actions, best_cost = [], None
        for action, (cell, _) in safe.items():
            if action == Action.BOMB:
                continue
            cost = to_opp.get(cell, 99)
            if closing_ring is not None and ring_of(cell, len(walls)) == closing_ring:
                cost += 100
            if best_cost is None or cost < best_cost:
                best_actions, best_cost = [action], cost
            elif cost == best_cost:
                best_actions.append(action)
        if best_actions:
            return random.choice(best_actions)
        return Action.STAY
