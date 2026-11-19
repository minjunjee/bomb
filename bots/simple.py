"""Example bot. Read it top to bottom: it shows how to read the observation and return an action.
It is deliberately simple and plays badly."""

import random

from bombcompetition import RULES, Action


class Policy:
    def __init__(self):
        # Called once at the start of every game. Load anything you need here.
        pass

    def act(self, obs):
        # Called every step. Look at obs, then return one action (0-5).

        # Where am I, and where is the opponent? Positions are (row, col); row 0 is the top.
        my_row, my_col = obs["me"]
        opp_row, opp_col = obs["opp"]

        # Which bombs on the board are mine? bomb["owner"] is the player who dropped it (0 = A, 1 = B).
        my_bombs = [bomb for bomb in obs["bombs"] if bomb["owner"] == obs["player"]]

        # If the opponent is within blast range and I may drop another bomb, drop one.
        # RULES holds the game settings: fuse, radius, max_bombs, lives, max_steps.
        distance = abs(my_row - opp_row) + abs(my_col - opp_col)
        if distance <= RULES.radius and len(my_bombs) < RULES.max_bombs:
            return Action.BOMB

        # Otherwise check the four neighbouring cells and keep the ones I can walk into.
        bomb_cells = [bomb["pos"] for bomb in obs["bombs"]]
        free_moves = []
        for action, d_row, d_col in [
            (Action.UP, -1, 0),
            (Action.DOWN, 1, 0),
            (Action.LEFT, 0, -1),
            (Action.RIGHT, 0, 1),
        ]:
            row, col = my_row + d_row, my_col + d_col
            if obs["walls"][row, col]:  # a wall
                continue
            if (row, col) in bomb_cells:  # a bomb
                continue
            if (row, col) == (opp_row, opp_col):  # the opponent
                continue
            free_moves.append(action)

        # Move in a random free direction, or stay if there is none.
        if free_moves:
            return random.choice(free_moves)
        return Action.STAY
