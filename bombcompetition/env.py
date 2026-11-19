"""Gymnasium env: you play one side, `opponent` (obs -> action) plays the other."""

import gymnasium as gym

from .game import RULES, Game, Rules
from .render import to_rgb, to_text


def kill_reward(prev_obs, action, obs, deaths) -> float:
    """The default reward: +1 when your bomb hits the opponent, -1 when you lose a life."""
    me = obs["player"]
    return float(sum((d.killer == me) - (d.player == me) for d in deaths))


class BomberEnv(gym.Env):
    """
    opponent:  obs -> action for the other player
    player:    the side you play, 0 = A or 1 = B
    rules:     a Rules (default: the official RULES)
    reward_fn: (prev_obs, action, obs, deaths) -> float, called after every step;
               deaths lists who was hit in that step and by whom (Death(player, killer))
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, opponent, player: int = 0, rules: Rules = RULES, reward_fn=kill_reward,
                 render_mode: str | None = None):
        self.opponent, self.player, self.rules, self.render_mode = opponent, player, rules, render_mode
        self.reward_fn = reward_fn
        n, sp = Game(rules).size, gym.spaces
        cell = sp.Tuple((sp.Discrete(n), sp.Discrete(n)))
        longest = max(rules.fuse, rules.fuse_max or 0)
        bomb = sp.Dict({"pos": cell, "owner": sp.Discrete(2), "fuse": sp.Discrete(rules.fuse + 1),
                        "age": sp.Discrete(longest + 1)})
        self.observation_space = sp.Dict({
            "player": sp.Discrete(2),
            "walls": sp.MultiBinary((n, n)),
            "me": cell,
            "opp": cell,
            "bombs": sp.Sequence(bomb),
            "lives": sp.Tuple((sp.Discrete(rules.lives + 1),) * 2),
            "step": sp.Discrete(rules.max_steps + 1),
        })
        self.action_space = sp.Discrete(6)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game(self.rules, seed=seed)
        return self.game.obs(self.player), {}

    def step(self, action):
        game, me = self.game, self.player
        prev_obs = game.obs(me)
        actions = [0, 0]
        actions[me], actions[1 - me] = int(action), int(self.opponent(game.obs(1 - me)))
        deaths = game.step(*actions)
        obs = game.obs(me)
        reward = float(self.reward_fn(prev_obs, int(action), obs, deaths))
        terminated = min(game.lives) <= 0
        info = {"deaths": deaths, "lives": tuple(game.lives), "kills": tuple(game.kills)}
        return obs, reward, terminated, game.done and not terminated, info

    def render(self):
        return to_rgb(self.game) if self.render_mode == "rgb_array" else to_text(self.game)
