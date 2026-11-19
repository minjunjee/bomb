# BOMB competition: project brief (read this first)

Goal: the strongest possible **neural-network** agent for an in-class 1-vs-1 bomb game. Only a network is submitted;
everything else here (bots, search, harness) exists to train and to grade it locally.
**Never submit anything to the competition website and never use the network. Local grading only.**

## Environment and conventions

- Project root: `/home/minjun_dev/bomb`. Run everything from the root with the project python:
  `cd /home/minjun_dev/bomb && nice -n 10 .venv/bin/python -m arena.core ...` (py3.11, torch 2.11+cu128, numpy 2.4,
  numba 0.68, gymnasium 1.4; gcc 13 is installed). Do not install packages.
- Shared machine (256 cores, other users): prefix heavy commands with `nice -n 10` and stay inside the CPU budget
  of your task. Long jobs: run them in the background and poll, a foreground command is killed after 10 minutes.
- Disk: the home disk is almost full. Large files (datasets, checkpoints, logs) go ONLY under `runs/`
  (a symlink to `/data/minjun_dev/bomb_runs`, 500 GB free). Do not put big files in `/tmp` or elsewhere under `/home`.
- GPUs: ONLY torch devices `cuda:0` .. `cuda:3` (physical GPUs 4-7) may be used. Pass the device string
  (do not set `CUDA_VISIBLE_DEVICES`); never touch `cuda:4`-`cuda:7` (other users).
- Read-only: `bombcompetition/` (the official engine), `bots/simple.py`, `arena/core.py`, `arena/nets.py`,
  `arena/zoo/base_cautious.py`, this file, and every file your task did not ask you to create. Other agents work in
  the same tree in parallel: create and modify ONLY the files your task names.
- Evaluation maps are `random_rules(seed)` with `seed >= 1_000_000`. Training code must only use map seeds below
  1_000_000 (or unseeded `random_rules()`).

## The game (ground truth: `bombcompetition/game.py`, 250 lines: read it)

- 9x9 board, cells are `(row, col)`, row 0 = top. The border is wall; inside, random walls, the same after a 180°
  turn; 28-44 open cells; one connected area. Player A (index 0) starts at `(1, 1)`, B (index 1) at `(7, 7)`.
  `random_rules(seed)` gives the rules with a random map (same seed = same map).
- Actions: `0 STAY`, `1 UP` (row-1), `2 DOWN` (row+1), `3 LEFT` (col-1), `4 RIGHT` (col+1), `5 BOMB`.
- One step, both players act at the same time, resolved in this order:
  1. **Drop.** BOMB puts a bomb (fuse 5) on the player's own cell, unless that cell already holds a bomb or the
     player already owns 3 bombs on the board. A BOMB action never moves the player.
  2. **Move.** A target cell is blocked if it is a wall, holds a bomb (also one dropped in this very step), or is the
     cell the other player stands on at the START of the step (even if that player walks away). If both players
     target the same cell, neither moves. Players never swap and never share a cell. A player standing on a bomb can
     walk off it; nobody can walk onto a bomb.
  3. **Tick and explode.** Every fuse goes down by 1; a bomb at 0 explodes. An explosion covers the bomb's cell and
     up to 2 cells in each of the 4 directions; only walls stop it (it passes through bombs and players). A bomb
     inside an explosion explodes in the same step (chain reaction).
  4. **Hits.** A player standing (after moving) on an exploded cell is hit and the game ends at once (1 life).
     Only one player hit: the other one WINS, whoever owned the bombs (being hit by your own bomb, a "self-bomb",
     is simply a loss). Both hit in the same step: DRAW. Nobody hit after 300 steps: DRAW.
  5. **Sudden death.** When the step counter reaches 100, ring 1 of the board (rows/cols 1 and 7) turns into walls;
     at 200, ring 2 (rows/cols 2 and 6): the open area goes 7x7 -> 5x5 -> 3x3. Bombs on the ring vanish; a player on
     the ring is teleported to the nearest open cell inside (Manhattan distance, ties -> smallest `(row, col)`,
     cells with a bomb or the other player are avoided if possible). If nothing inside is open, no shrink happens.
- `obs` for player p (a dict, the full state): `player` (p), `walls` (9x9 bool array), `me`, `opp` (cells),
  `bombs` (tuple of `{"pos", "owner", "fuse", "age"}`), `lives`, `step`.
- **Timing.** The `fuse` in `obs` = the number of steps until the bomb explodes: `fuse == 1` explodes at the end of
  the step being decided now. A bomb dropped now shows `fuse == 4` in the next obs and explodes at the end of the
  5th step counting the drop step, so after dropping you get exactly 4 more moves. A chain can set a bomb off
  earlier than its own fuse.
- Helpers in `bombcompetition`: `blast_cells(walls, cell, RULES.radius)`, `legal_actions(obs)`, `DELTAS`, `MOVES`,
  `Action`, `RULES` (fuse 5, radius 2, max_bombs 3, lives 1, max_steps 300, shrink_every 100), `Game(rules, seed)`
  with `.step(a0, a1)`, `.obs(p)`, `.done`, `.winner()`, `.t`, `.lives`, `.kills`.

## What is graded

- The team submits ONLY `submission.py` (class `Net`) + `submission.pth` (weights). The server plays it with the
  fixed `NeuralPolicy` (`bombcompetition/neural.py`): `obs -> turn(obs)` (player B sees the board turned 180°, so
  the network always sees itself starting top-left) `-> preprocess` (a fixed `(13, 9, 9)` tensor) `-> Net -> argmax`
  of 6 scores (the action is turned back for B). So the final agent is a **deterministic, memoryless function of
  the current board**. `Net.forward` must be a learned network: no hand-written game logic, search or rules inside.
  Rule-based bots and search are allowed only for training (as opponents and as teachers).
- Leaderboard: 100 games against a hidden TA bot ("SmartBot", probably rule-based: dodges blasts, chases, bombs
  when the opponent cannot escape) on unseen random maps, both sides. **Score = number of wins.** A draw is worth
  nothing; fewer losses only breaks ties. Later: a hidden "final TA bot" and a round robin between all teams'
  networks (other teams: mostly DQN agents trained for an hour or two).
- Server limits: 1 CPU, 2 GB, 2 s per step (we want < 20 ms).

## The harness: `arena/core.py`

Bot specs (strings): `random`, `simple`, `zoo.<name>` (= `arena/zoo/<name>.py`, class `Policy`),
`zoo.<name>:<arg>` (= that module's `make_policy(arg)`), `noisy:<eps>:<spec>` (random legal action with probability
eps), `net:<folder>` (a folder with `submission.py` + `submission.pth`, played exactly like the server),
`net:<folder>@sample`, or a path to any `.py` file with a `Policy` class.

A bot = a class `Policy` with `__init__(self)` (a new instance for every game) and `act(self, obs) -> int`.
Bots get the real (un-turned) `obs`; `obs["player"]` tells which side they are. An exception or an invalid action
counts as a fault and the bot stays put.

```
# "me" against opponents the way the server grades (each map from both sides), W/D/L, self-bombs, ms per act
nice -n 10 .venv/bin/python -m arena.core leaderboard --me zoo.base_cautious --opp random,simple --games 200 --procs 16
# round robin
nice -n 10 .venv/bin/python -m arena.core tournament --bots zoo.base_cautious,simple,random --games 100 --procs 16
# watch one game (GIF) on evaluation map 1000003
.venv/bin/python -m arena.core gif --a zoo.base_cautious --b simple --map 1000003 --out runs/tmp/x.gif
```

Python API: `from arena.core import play_game, run_games, leaderboard, tournament, bot_factory`.
`--json out.json` saves the full result. Yardsticks: `random` (suicidal), `simple` (freezes and blows itself up when
you come within 2 cells), `zoo.base_cautious` (dodges with exact timing, bombs when the opponent is in its blast,
chases; it ignores future opponent bombs and blocking).

## Networks: `arena/nets.py`

`build_net(cfg)` (training: `forward(x) -> (scores (B, 6), value (B,))`), `export_submission(net, folder)`,
`check_submission(folder)` (the notebook's server check in a clean process + timing), `load_submission(folder)`,
`augment(x, y, k, q)` (the 8 board symmetries for data augmentation), `make_zip(folder)`. The exported network
behaves the same in train() and eval() mode. Training targets are always in the TURNED frame:
`x = preprocess(turn(obs))`, `y = turn_action(real_action, obs)` (`turn_action` is its own inverse).
