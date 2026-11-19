"""Bots used only by arena/rl/test_vecenv.py (harness spec: arena/rl/_testbot.py -> class Policy).

Policy: a deterministic function of the observation (no random module), so that many interleaved environments in
one worker are reproducible. It checks that it gets the REAL (un-turned) observation: player A starts at (1, 1) and
player B at (7, 7); if the first observation of a game is turned it raises (and so would STAY for ever).
Every 13th call it raises and every 17th it returns an invalid action: the environment must play STAY then.
"""

import zlib


class Policy:
    def __init__(self):
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        if obs["step"] == 0:
            want = (1, 1) if obs["player"] == 0 else (7, 7)
            if tuple(obs["me"]) != want:
                raise RuntimeError("the opponent got a turned observation")
            self.ok = True
        if not getattr(self, "ok", False):
            raise RuntimeError("bad first observation")
        if self.calls % 13 == 0:
            raise RuntimeError("scheduled fault")
        if self.calls % 17 == 0:
            return 7  # invalid
        key = f"{obs['player']}|{obs['me']}|{obs['opp']}|{obs['step']}|{sorted((b['pos'], b['fuse']) for b in obs['bombs'])}"
        h = zlib.crc32(key.encode())
        return h % 6 if h % 5 else 0  # a bomb now and then, mostly moving
