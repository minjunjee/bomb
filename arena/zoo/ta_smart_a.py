"""SmartBot, variant A: the kind of rule-based bot a TA writes in an afternoon (clear rules, no deep search).

Rules, in this order:
1. Dodge.  Never take an action after which it cannot outrun the bombs on the board (exact fuse timing, chain
   reactions, bombs block the way, the sudden-death ring closes). If it can, it also keeps a way out for the case
   that the opponent drops a bomb right now.
2. Trap.   Drop a bomb when that takes the opponent's ways out away (none or one left) and it can still get away
   itself; now and then also when the opponent is simply inside the blast (pressure).
3. Shrink. Walk off the outer ring before it turns into walls, and do not step onto it at the last moment.
4. Chase.  Otherwise walk toward the opponent (BFS), random among equally good moves.

Timing: "time k" = the end of the k-th step from now (the step being decided is k = 1). A bomb whose obs fuse is f
explodes at time f, a bomb dropped now at time RULES.fuse. The ring closes at time `close`, after the explosions.
"""

import random
from collections import deque

import numpy as np

from bombcompetition import DELTAS, MOVES, RULES, Action, blast_cells
from bombcompetition.game import shrink_ring

SAFE = 99        # Policy.life(): "there is always a cell to stand on"
PRESSURE = 0.15  # chance per step of a bomb when the opponent is in the blast but could get away
EARLY = 10       # steps before a shrink from which the closing ring is avoided
DEBUG = True     # True: let exceptions through (for testing)


class Board:
    """One wall layout as plain python sets and dicts (numpy indexing is slow)."""

    def __init__(self, walls):
        n = len(walls)
        self.walls = walls
        self.open = {(r, c) for r in range(n) for c in range(n) if not walls[r, c]}
        self.nbrs = {(r, c): [x for x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if x in self.open]
                     for (r, c) in self.open}
        self.blast = {x: frozenset(blast_cells(walls, x, RULES.radius)) for x in self.open}
        self.shrunk = {}  # ring number -> (open cells of that ring, the walls after it has closed) or None


def chain(blast, fuses):
    """{bomb cell: time it explodes}: a bomb inside an earlier blast goes off with it (chain reaction)."""
    t, changed = dict(fuses), len(fuses) > 1
    while changed:
        changed = False
        for c in t:
            for d in t:
                if t[d] > t[c] and d in blast[c]:
                    t[d], changed = t[c], True
    return t


class Policy:
    def __init__(self, wary=1, pressure=PRESSURE):
        self.boards = {}  # wall layout -> Board (a game has at most three layouts)
        self.wary, self.pressure = wary, pressure

    def board(self, walls):
        key = walls.tobytes()
        if key not in self.boards:
            self.boards[key] = Board(walls)
        return self.boards[key]

    def look(self, obs):
        """Set the board, and what the next sudden-death shrink does: `left` steps until then (0 = never),
        the `ring` cells that close, the board `after`, and `close` = `left` if bombs on the board can still matter."""
        self.now = now = self.board(np.asarray(obs["walls"]))
        every, step = RULES.shrink_every, obs["step"]
        self.left, self.ring, self.after = 0, frozenset(), now
        if every and step + every - step % every < RULES.max_steps:
            k = step // every + 1
            if k not in now.shrunk:
                ring, inner = shrink_ring(now.walls, k)
                walls = now.walls.copy()
                for cell in ring:
                    walls[cell] = True
                now.shrunk[k] = (frozenset(ring & now.open), walls) if inner else None
            if now.shrunk[k]:
                self.left, self.ring, self.after = every - step % every, now.shrunk[k][0], self.board(now.shrunk[k][1])
        self.close = self.left if self.left <= RULES.fuse else 0

    def timeline(self, fuses):
        """fuses: {bomb cell: time it explodes on its own} -> (gone, hit, last, danger): the bomb on `cell` blocks it
        up to step gone[cell], hit[k] = the cells that explode at time k, last = the last event, danger = all hit cells.
        A bomb on the closing ring vanishes at `close`; the bombs left then explode inside the new walls."""
        gone, hit, late = chain(self.now.blast, fuses), {}, {}
        for c, t in list(gone.items()):
            if self.close and t > self.close:
                if c in self.ring:
                    gone[c] = self.close
                else:
                    late[c] = fuses[c]
            else:
                hit[t] = hit.get(t, frozenset()) | self.now.blast[c]
        for c, t in chain(self.after.blast, late).items():
            gone[c] = t
            hit[t] = hit.get(t, frozenset()) | self.after.blast[c]
        return gone, hit, max(gone.values(), default=0), frozenset().union(*hit.values())

    def life(self, start, plan, blocked=frozenset(), race=None, teleport=False):
        """Standing on `start` after the step being decided (time 1): the first time at which no cell outside the
        explosions is left to stand on, SAFE if there always is one. Bombs block the way until they are gone,
        `blocked` cells cannot be entered, and with `race` ({cell: the step at which the other player can be there})
        only cells that I reach first can be entered. The closing ring counts as deadly (we do not want to be
        teleported); with teleport=True being moved inward counts as getting away (used for the opponent)."""
        gone, hit, last, danger = plan
        race = race or {}
        if start not in danger and not (self.close and start in self.ring):
            return SAFE
        frontier, k = {start}, 1
        while True:
            if k in hit:
                frontier -= hit[k]
            if k == self.close:
                if teleport and not frontier.isdisjoint(self.ring):
                    return SAFE
                frontier -= self.ring
            if not frontier:
                return k
            if k >= last:
                return SAFE
            k += 1  # the move of step k
            nbrs = self.after.nbrs if self.close and k > self.close else self.now.nbrs
            nxt = set(frontier)
            for x in frontier:
                for y in nbrs[x]:
                    if gone.get(y, 0) < k and y not in blocked and race.get(y, SAFE) > k:
                        nxt.add(y)
            frontier = nxt

    def ways_out(self, who, other, fuses, plan):
        """The number of first moves (staying included) after which player `who` can still get away."""
        cells = [who] + [x for x in self.now.nbrs[who] if x not in fuses and x != other]
        return sum(self.life(x, plan, teleport=True) == SAFE for x in cells)

    def steps(self, sources, blocked):
        """BFS: {cell: steps to the nearest source}, not entering blocked cells."""
        seen, queue, nbrs = dict.fromkeys(sources, 0), deque(sources), self.now.nbrs
        while queue:
            x = queue.popleft()
            for y in nbrs[x]:
                if y not in seen and y not in blocked:
                    seen[y] = seen[x] + 1
                    queue.append(y)
        return seen

    def act(self, obs):
        try:
            return int(self.decide(obs))
        except Exception:  # a bot must never crash a game
            if DEBUG:
                raise
            return int(Action.STAY)

    def decide(self, obs):
        self.look(obs)
        now, me, opp = self.now, tuple(obs["me"]), tuple(obs["opp"])
        fuses = {tuple(b["pos"]): b["fuse"] for b in obs["bombs"]}
        mine = sum(b["owner"] == obs["player"] for b in obs["bombs"])
        can_bomb = mine < RULES.max_bombs and me not in fuses
        opp_can_bomb = len(fuses) - mine < RULES.max_bombs and opp not in fuses
        plan = self.timeline(fuses)
        danger = plan[3]  # the cells inside a future blast
        cell = {Action.STAY: me}  # action -> my cell after the step
        for a in MOVES:
            x = (me[0] + DELTAS[a][0], me[1] + DELTAS[a][1])
            if x in now.open and x not in fuses and x != opp:
                cell[a] = x

        # 1. dodge: keep the actions after which I can outrun every bomb. Three guesses about the opponent:
        #    it gets in my way wherever it can / it stays where it is / it steps aside (and a teleport is fine).
        wall = frozenset([opp])
        race = self.steps([opp], set(fuses) | {me})  # the step at which the opponent can be on a cell
        for blocked, ahead in ((frozenset(), race), (wall, None), (frozenset(), None)):
            life = {a: self.life(x, plan, blocked, ahead, teleport=not (blocked or ahead)) for a, x in cell.items()}
            if ahead:  # if the opponent steps onto the same cell neither of us moves: then staying must be fine too
                life = {a: min(t, life[Action.STAY]) if race.get(cell[a]) == 1 else t for a, t in life.items()}
            if max(life.values()) == SAFE:
                break
        best = max(life.values())
        safe = [a for a in cell if life[a] == best]
        if best == SAFE and opp_can_bomb and self.wary == 1:  # ... and after which I could still get away if the opponent bombed now
            threat = self.timeline({**fuses, opp: RULES.fuse})
            safe = [a for a in safe if self.life(cell[a], threat, wall) == SAFE] or safe
        if best == SAFE and opp_can_bomb and self.wary == 2:
            threat = self.timeline({**fuses, opp: RULES.fuse})
            safe = [a for a in safe if self.life(cell[a], threat, race=race) == SAFE] or safe
        if best == SAFE and opp_can_bomb and self.wary >= 3:
            # ... and after which the opponent could not trap me with a bomb dropped now, or after one step
            for n in [opp] + now.nbrs[opp]:
                if n not in fuses and n != me:
                    threat = self.timeline({**fuses, n: RULES.fuse + (n != opp)})
                    if self.wary == 3:
                        safe = [a for a in safe if cell[a] == n or self.life(cell[a], threat, race=race) == SAFE] or safe
                    else:
                        safe = [a for a in safe if cell[a] == n or self.life(cell[a], threat, wall | {n}) == SAFE] or safe

        # 2. bomb
        if can_bomb:
            reach, with_mine = opp in now.blast[me], {**fuses, me: RULES.fuse}
            bomb = self.timeline(with_mine)
            if best < SAFE:  # I am lost anyway: take the opponent along
                if reach and life[Action.STAY] == best:
                    return Action.BOMB
            elif self.life(me, bomb, wall) == SAFE and not (opp_can_bomb and self.life(
                    me, self.timeline({**with_mine, opp: RULES.fuse}), wall) < SAFE):  # I can get away from it
                if reach or opp in danger:  # trap: the bomb takes the opponent's ways out away
                    if self.ways_out(opp, me, with_mine, bomb) <= min(1, self.ways_out(opp, me, fuses, plan) - 1):
                        return Action.BOMB
                if (reach and not self.close and random.random() < self.pressure
                        and self.life(me, bomb, race=race) == SAFE):  # pressure, only without any risk of a block
                    return Action.BOMB

        # 3. run: inside a future blast take the shortest way out, and do not walk into one
        moves = safe
        if not ahead:  # no sure way: move anyway, if possible not onto a cell the opponent can step onto as well
            go = [a for a in moves if a != Action.STAY]
            moves = [a for a in go if race.get(cell[a]) != 1] or go or moves
        if me in danger:
            calm = self.steps(now.open - danger, set(fuses) | wall)
            return self.pick(moves, lambda a: calm.get(cell[a], 99))
        moves = [a for a in moves if cell[a] not in danger] or moves

        # 4. shrink: some steps before the ring closes, walk off it and do not step onto it any more
        if 0 < self.left <= EARLY:
            if me in self.ring:
                inward = self.steps(self.after.open - set(fuses), fuses)
                return self.pick(moves, lambda a: inward.get(cell[a], 99))
            moves = [a for a in moves if cell[a] not in self.ring] or moves

        # 5. chase: walk toward the opponent
        dist = self.steps([opp], fuses)
        if me not in dist:  # bombs in the way: head there anyway
            dist = self.steps([opp], ())
        return self.pick(moves, lambda a: dist.get(cell[a], 99))

    @staticmethod
    def pick(actions, cost):
        """A random one of the actions with the lowest cost."""
        costs = [cost(a) for a in actions]
        return random.choice([a for a, c in zip(actions, costs) if c == min(costs)])


def make_policy(arg):
    """zoo.ta_smart_a:<wary>,<pressure>  (testing)"""
    w, _, pr = arg.partition(",")
    return lambda: Policy(int(w), float(pr) if pr else PRESSURE)
