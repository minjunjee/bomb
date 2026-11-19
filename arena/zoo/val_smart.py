"""SmartBot, as we guess the course TA wrote it. Validation only: never train against this file.

A rule-based bot in the spirit of bots/simple.py, but one that knows exactly when every bomb goes off.
Every step it walks down this list and takes the first rule that applies:

  1. DODGE   Never do anything after which it could not outrun the bombs on the board
             (exact timing, chain reactions and the sudden-death shrink included).
  2. KILL    Drop a bomb when the opponent cannot escape it, and SmartBot itself can.
  3. PRESS   Drop a bomb when the opponent stands inside its blast and SmartBot has a way out that the opponent
             cannot spoil, even if the opponent drops a bomb in the same step.
  4. SHRINK  Shortly before a sudden-death shrink, walk inside the ring that is about to close.
  5. CHASE   Otherwise walk toward the opponent, never stepping into a blast it does not have to.

Timing. "Time k" is the end of the k-th step from now (the step being decided now is k = 1). A bomb whose fuse in
obs is f explodes at time f; a bomb dropped now explodes at time RULES.fuse (5), so after a drop there are 4 moves
left. A bomb lies in the way until it has exploded. When the ring closes, the explosions of that step come first.

What it does NOT do (this is how it can be beaten): it only looks at the bombs that are on the board now. It never
asks where the opponent could put the next one, so it can be cornered with a second bomb or caught in a dead end.
"""

import os
import random

import numpy as np

from bombcompetition import DELTAS, MOVES, RULES, Action, blast_cells, ring_of

STAY, BOMB = int(Action.STAY), int(Action.BOMB)
FUSE = RULES.fuse  # a bomb dropped now explodes at time FUSE
EVERY = RULES.shrink_every or 0  # sudden death: a ring closes every EVERY steps
DEAD = 100  # risk values from DEAD up mean "no way out"
NOBODY = {}  # for the searches below: no other player is in the way
DEBUG = bool(os.environ.get("VAL_SMART_DEBUG"))  # testing only: let errors through instead of standing still


# ----------------------------------------------------------------------------------------------------------------
# The board: facts that depend only on the walls. Worked out once per map (and once more after every shrink).
# ----------------------------------------------------------------------------------------------------------------
class Board:
    def __init__(self, walls):
        size = len(walls)
        self.walls = walls
        self.cells = [(r, c) for r in range(size) for c in range(size) if not walls[r, c]]  # the open cells
        self.moves = {}  # cell -> ((action, neighbour cell), ...) for the open neighbours
        self.around = {}  # cell -> (neighbour cell, ...)
        self.blast = {}  # cell -> the cells a bomb on it hits
        for cell in self.cells:
            steps = []
            for action in MOVES:
                target = (cell[0] + DELTAS[action][0], cell[1] + DELTAS[action][1])
                if not walls[target]:
                    steps.append((int(action), target))
            self.moves[cell] = tuple(steps)
            self.around[cell] = tuple(target for _, target in steps)
            self.blast[cell] = frozenset(blast_cells(walls, cell, RULES.radius))
        self._steps_to = {}  # filled when asked: target cell -> {cell: walking steps to it}
        self._closing = {}  # filled when asked: ring number -> what its closing does

    def walk(self, starts):
        """{cell: walking steps to the nearest of `starts`}. Bombs and players are not in the way here."""
        steps = {cell: 0 for cell in starts}
        queue = list(starts)
        for cell in queue:  # the list grows while we walk along it: a simple queue
            for target in self.around[cell]:
                if target not in steps:
                    steps[target] = steps[cell] + 1
                    queue.append(target)
        return steps

    def steps_to(self, target):
        if target not in self._steps_to:
            self._steps_to[target] = self.walk([target])
        return self._steps_to[target]

    def closing(self, k):
        """Sudden death, ring k: (its open cells, the Board after it closed, {cell: walking steps to get inside}).
        None if nothing inside is open: then the engine does not shrink."""
        if k not in self._closing:
            size = len(self.walls)
            inside = [cell for cell in self.cells if ring_of(cell, size) > k]
            info = None
            if inside:
                walls = self.walls.copy()
                for r in range(size):
                    for c in range(size):
                        if ring_of((r, c), size) == k:
                            walls[r, c] = True
                ring = frozenset(cell for cell in self.cells if ring_of(cell, size) == k)
                info = (ring, board_of(walls), self.walk(inside))
            self._closing[k] = info
        return self._closing[k]


_BOARDS = {}


def board_of(walls):
    key = walls.tobytes()
    board = _BOARDS.get(key)
    if board is None:
        if len(_BOARDS) >= 128:
            _BOARDS.clear()
        board = _BOARDS[key] = Board(walls.copy())
    return board


# ----------------------------------------------------------------------------------------------------------------
# The bombs: when does what explode?
# ----------------------------------------------------------------------------------------------------------------
def chain(board, fuses):
    """{bomb cell: fuse} -> {bomb cell: the time it really explodes}. A bomb inside an earlier blast goes off with it."""
    when = dict(fuses)
    if len(when) > 1:
        again = True
        while again:
            again = False
            for first, t in list(when.items()):
                for other in when:
                    if when[other] > t and other in board.blast[first]:
                        when[other] = t
                        again = True
    return when


def explosions(board, fuses, shrink=None):
    """The plan of what will explode. Returns (hit, last, block, end):
        hit    {time: the cells that explode then}
        last   {cell: the last time it explodes}; a cell that is not in it never explodes
        block  {bomb cell: the time until which this bomb lies in the way}
        end    the time of the last explosion (0 without bombs)
    shrink = (time the ring closes, its cells, the board after), given only when that is before a new bomb's fuse ends:
    a bomb on the ring that has not exploded by then is taken off the board and never explodes."""
    when = block = chain(board, fuses)
    if shrink is not None:
        closes, ring, _ = shrink
        gone = [cell for cell, t in when.items() if t > closes and cell in ring]
        if gone:
            when = chain(board, {cell: f for cell, f in fuses.items() if cell not in gone})
            block = dict(when)
            for cell in gone:
                block[cell] = closes
    hit, last = {}, {}
    for cell, t in when.items():
        zone = board.blast[cell]
        hit[t] = hit[t] | zone if t in hit else zone
        for c in zone:
            if last.get(c, 0) < t:
                last[c] = t
    return hit, last, block, max(when.values(), default=0)


# ----------------------------------------------------------------------------------------------------------------
# Running away: searches through time.
# ----------------------------------------------------------------------------------------------------------------
def arrival(board, block, start, limit):
    """{cell: the earliest time a player now on `start` can stand on it} for all cells it can reach by time `limit`,
    walking around the bombs (a bomb lies in the way until it has exploded)."""
    when = {start: 0}
    edge = [start]  # cells that may still have a neighbour to walk to
    for k in range(1, limit + 1):
        reached, waiting = [], []
        for cell in edge:
            for target in board.around[cell]:
                if target not in when:
                    if block.get(target, 0) < k:
                        when[target] = k
                        reached.append(target)
                    elif cell not in waiting:
                        waiting.append(cell)  # a bomb is in the way: try again in the next step
        edge = reached + waiting
        if not edge:
            break
    return when


def escape(board, plan, start, rival=NOBODY, shrink=None):
    """A player stands on `start` at the end of this step (time 1). How bad is that?
        1 .. FUSE   fine: the earliest time it can stand on a cell where nothing explodes any more (1 = it is there)
        >= DEAD     no way out; the later it gets hit, the smaller the number
    rival: {cell: time}: from that time on the other player may be on the cell or step onto it, so the step of
    time k may only go to a cell whose time is larger than k. {its cell: 0} = "it stays where it is"."""
    hit, last, block, end = plan
    if shrink is not None and shrink[0] < end:
        return escape_while_closing(board, plan, start, rival, shrink)
    if start not in last:
        return 1
    if start in hit.get(1, ()):
        return DEAD + 9
    around = board.around
    front = (start,)  # every cell the player can be on at time k - 1
    for k in range(2, end + 1):  # the move of step k, then the explosions of time k
        boom = hit.get(k, ())
        reached = set()
        for cell in front:
            if cell not in boom:
                reached.add(cell)  # wait here
            for target in around[cell]:
                if target in boom or target in reached or block.get(target, 0) >= k or rival.get(target, 99) <= k:
                    continue  # it explodes now / we know it already / a bomb or the other player is in the way
                if last.get(target, 0) < k:
                    return k  # nothing explodes there any more
                reached.add(target)
        if not reached:
            return DEAD + 10 - k
        front = reached
    return 1


def escape_while_closing(board, plan, start, rival, shrink):
    """The same search for the few steps in which the ring closes before the last bomb has exploded. The engine's
    order at that time: explosions, then the ring becomes wall and a player on it is put on the nearest free cell
    inside (Manhattan distance, ties -> smallest (row, col))."""
    hit, last, block, end = plan
    closes, ring, after = shrink
    k, front = 1, {start}
    while True:
        boom = hit.get(k, ())
        front = {cell for cell in front if cell not in boom}
        if not front:
            return DEAD + 10 - k
        if k == closes:
            free = [cell for cell in after.cells if block.get(cell, 0) <= k and rival.get(cell, 99) > 0] or after.cells
            front = {cell if cell not in ring else
                     min(free, key=lambda x: (abs(x[0] - cell[0]) + abs(x[1] - cell[1]), x)) for cell in front}
        if k >= 12 or any(last.get(cell, 0) <= k and (k >= closes or cell not in ring) for cell in front):
            return k
        k += 1
        around = after.around if k > closes else board.around
        reached = set(front)
        for cell in front:
            for target in around.get(cell, ()):
                if block.get(target, 0) < k and rival.get(target, 99) > k:
                    reached.add(target)
        front = reached


def judge(board, plan, cell, reach, opp, shrink):
    """How safe is it to stand on `cell` at the end of this step? Returns (trust, risk), smaller is better:
        trust 0   there is a way out that the opponent cannot spoil: it cannot be first on any cell of it
              1   there is a way out if the opponent stays where it is
              2   there is a way out if the opponent gets out of the way
              3   there is none
        risk      see escape()"""
    risk = escape(board, plan, cell, reach, shrink)
    if risk < DEAD:
        return 0, risk
    risk = escape(board, plan, cell, {opp: 0}, shrink)
    if risk < DEAD:
        return 1, risk
    risk = escape(board, plan, cell, NOBODY, shrink)
    return (2, risk) if risk < DEAD else (3, risk)


def cannot_escape(board, plan, who, other, shrink=None):
    """True if the player on `who` gets hit whatever it does. `other` is where the other player stands now: nobody can
    step there in this step (after that we do not count on it being in the way)."""
    hit, last, block, end = plan
    if who not in last:
        return False  # nothing will explode where it stands
    if escape(board, plan, who, NOBODY, shrink) < DEAD:
        return False  # it can wait first
    for target in board.around[who]:
        if target not in block and target != other and escape(board, plan, target, NOBODY, shrink) < DEAD:
            return False
    return True


# ----------------------------------------------------------------------------------------------------------------
# The bot
# ----------------------------------------------------------------------------------------------------------------
class Policy:
    press = 1.0  # rule 3: how often it takes the chance (1.0 = always)
    wary = 1  # 0: looks only at the bombs on the board. 1: also keeps off cells where the opponent could get it with
    #           a chain reaction (a bomb dropped right now, next to a ticking one). 2 ("hard"): with any bomb dropped now.

    def __init__(self):
        # Called once per game. SmartBot keeps nothing between steps: everything is read from obs.
        pass

    def act(self, obs):
        try:
            return int(self.think(obs))
        except Exception:
            if DEBUG:
                raise
            return STAY  # never crash a game

    def think(self, obs):
        # Where is everything? Cells are (row, col); bombs become {cell: steps until the fuse runs out}.
        board = board_of(np.asarray(obs["walls"], dtype=bool))
        me, opp = tuple(obs["me"]), tuple(obs["opp"])
        fuses = {tuple(bomb["pos"]): int(bomb["fuse"]) for bomb in obs["bombs"]}
        mine = sum(1 for bomb in obs["bombs"] if bomb["owner"] == obs["player"])
        can_bomb = mine < RULES.max_bombs and me not in fuses
        opp_can_bomb = len(fuses) - mine < RULES.max_bombs and opp not in fuses

        # Sudden death: does a ring close soon? It closes at the end of the `left`-th step from now.
        closing, left = None, 0
        if EVERY:
            left = EVERY - obs["step"] % EVERY
            if obs["step"] + left < RULES.max_steps:
                closing = board.closing(obs["step"] // EVERY + 1)
        # It only changes the bomb timing if it comes before the fuse of a bomb dropped now has run out.
        shrink = (left, closing[0], closing[1]) if closing and left < FUSE else None

        # 1. DODGE. What can I do, where does it leave me at the end of this step, and how safe is it to be there?
        options = [(STAY, me)]
        for action, cell in board.moves[me]:
            if cell not in fuses and cell != opp:
                options.append((action, cell))
        now = explosions(board, fuses, shrink)
        if not fuses:
            safety = {action: (0, 1) for action, _ in options}  # no bombs, no danger
        else:
            reach = arrival(board, now[2], opp, FUSE)  # where the opponent can be, and when
            safety = {STAY: judge(board, now, me, reach, opp, shrink)}
            for action, cell in options[1:]:
                trust, risk = judge(board, now, cell, reach, opp, shrink)
                if reach.get(cell, 99) <= 1 and safety[STAY][0] > 0:
                    # The opponent can step there too, and then neither of us moves. If I cannot afford to lose that
                    # step, this move is no better than "works if the opponent stays where it is".
                    trust = max(trust, 1)
                safety[action] = (trust, risk)

        # Would a bomb of mine be worth it? Only look when it can matter: the opponent is in its blast, or there are
        # other bombs it could work together with.
        if can_bomb and (fuses or opp in board.blast[me]):
            mined = explosions(board, {**fuses, me: FUSE}, shrink)
            trust, _ = judge(board, mined, me, arrival(board, mined[2], opp, FUSE), opp, shrink)
            if trust <= 1 and opp_can_bomb:
                # The opponent may drop a bomb too in this very step (a lost opponent likes to): can I still get away?
                both = explosions(board, {**fuses, me: FUSE, opp: FUSE}, shrink)
                trust = max(trust, judge(board, both, me, arrival(board, both[2], opp, FUSE), opp, shrink)[0])
            if trust <= 1 and cannot_escape(board, mined, opp, me, shrink):
                # 2. KILL (if the opponent is lost even without my bomb, keep the bomb and just stay safe)
                if not cannot_escape(board, now, opp, me, shrink):
                    return BOMB
            elif (trust == 0  # 3. PRESS: I have a way out that the opponent cannot spoil, ...
                  and opp in board.blast[me]  # ... the opponent is in the blast, ...
                  and me not in now[1]  # ... I am not running from a bomb myself, ...
                  and mine < RULES.max_bombs - 1  # ... one bomb stays in reserve for rule 2, ...
                  and not (shrink and me in shrink[1])  # ... the bomb would not vanish with the closing ring
                  and random.random() < self.press):
                return BOMB

        # Chain reactions: a bomb that the opponent drops next to a ticking bomb goes off with it, long before its own
        # fuse ends. Which of my options could the opponent punish like that, with a bomb dropped right now?
        # (At the "hard" level it asks this for any bomb the opponent could drop now, not only for chains.)
        trap = {}
        if opp_can_bomb and (fuses if self.wary == 1 else self.wary):
            bombed = explosions(board, {**fuses, opp: FUSE}, shrink)
            if self.wary > 1 or bombed[2].get(opp, FUSE) < FUSE:
                trap = {action: escape(board, bombed, cell, {opp: 0}, shrink) >= DEAD for action, cell in options}

        # 4. SHRINK and 5. CHASE: pick the best move. What counts, in this order:
        #    a way out that needs as little luck as possible > (hard level: keep off trap cells)
        #    > be out of the blasts as early as possible > get inside a closing ring > get close to the opponent.
        steps_to_opp = board.steps_to(opp)
        steps_in = None
        if closing and left <= closing[2].get(me, 0) + 4:
            steps_in = closing[2]  # time to go inside (or to stay inside)

        def badness(option):
            action, cell = option
            trust, risk = safety[action]
            return (trust,
                    trap.get(action, False),
                    risk,
                    steps_in.get(cell, 99) if steps_in else 0,
                    steps_to_opp.get(cell, 99),
                    random.random())  # ties: any of them

        return min(options, key=badness)[0]


LEVELS = {
    "calm": dict(press=0.0),  # bombs only when the opponent cannot escape (the notebook's sentence, word for word)
    "hard": dict(wary=2),  # also keeps off cells where one fresh bomb of the opponent would trap it
    "naive": dict(wary=0),  # does not even think of chain reactions the opponent could start
}


def make_policy(level):
    """zoo.val_smart:<level> -> the same bot at another level (see LEVELS). The plain zoo.val_smart is the main one."""
    return type(f"Policy_{level}", (Policy,), dict(LEVELS[level]))
