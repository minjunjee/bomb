"""Build the reproduction notebook of the current best network.

    .venv/bin/python -m arena.mini.build_notebook

Writes (in the project root)
    bomb_best_repro.ipynb      the notebook: course Install cell, our code (embedded), the submission cell, the league,
                               the two PPO stages, local grading, the course's "make the submission" cell
    bomb_best_assets.tar.gz    binaries the notebook reads: the trained weights (assets/best) and the 12 student
                               networks used as opponents (runs/students/*)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent.parent
COURSE_NB = ROOT / "AIC20000_bomb_competition_2026.ipynb"
OUT_NB = ROOT / "bomb_best_repro.ipynb"
OUT_ASSETS = ROOT / "bomb_best_assets.tar.gz"

CODE_FILES = [
    "arena/__init__.py", "arena/core.py", "arena/nets.py", "arena/panel.py", "arena/grade.py",
    "arena/rl/__init__.py", "arena/rl/vecenv.py", "arena/rl/ppo.py",
    "arena/zoo/__init__.py", "arena/zoo/base_cautious.py",
    "arena/zoo/ta_smart_a.py", "arena/zoo/ta_smart_b.py", "arena/zoo/evader.py", "arena/zoo/ta_final_a.py",
    "arena/zoo/ta_final_b.py", "arena/zoo/param.py", "arena/zoo/x_solver.py", "arena/zoo/x_solver_core.py",
    "arena/zoo/val_smart.py", "arena/zoo/val_hunter.py", "arena/zoo/val_tricky.py",
    "arena/zoo/test_smart.py", "arena/zoo/test_boss.py", "arena/zoo/test_camper.py",
]


def _add(tar: tarfile.TarFile, path: Path, arcname: str):
    info = tar.gettarinfo(str(path), arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    with open(path, "rb") as f:
        tar.addfile(info, f)


def code_bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for rel in CODE_FILES:
            _add(tar, ROOT / rel, rel)
    return "\n".join(textwrap.wrap(base64.b64encode(buf.getvalue()).decode(), 120))


def build_assets() -> tuple[str, list[str], list[str]]:
    train = sorted(p.name for p in (ROOT / "runs/students").glob("train_*") if (p / "submission.pth").exists())
    test = sorted(p.name for p in (ROOT / "runs/students").glob("test_*") if (p / "submission.pth").exists())
    with tarfile.open(OUT_ASSETS, "w:gz", compresslevel=6) as tar:
        for f in ("submission.py", "submission.pth"):
            _add(tar, ROOT / "best" / f, f"assets/best/{f}")
        for name in train + test:
            for f in ("submission.py", "submission.pth"):
                _add(tar, ROOT / "runs/students" / name / f, f"runs/students/{name}/{f}")
    return hashlib.md5((ROOT / "best/submission.pth").read_bytes()).hexdigest(), train, test


def md(text: str):
    return new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text: str):
    return new_code_cell(textwrap.dedent(text).strip("\n"))


def main():
    course = json.loads(COURSE_NB.read_text(encoding="utf-8"))
    install_src = "".join(course["cells"][2]["source"])
    last_src = "".join(course["cells"][24]["source"])
    assert install_src.startswith("#@title Install the game engine") and "submission.zip" in last_src
    md5, train_students, test_students = build_assets()
    grade = json.loads((ROOT / "best/grade.json").read_text())
    per = grade["val"]["per_opp"]
    test = json.loads((ROOT / "best/test.json").read_text())["per_opp"]
    rule = [test[o]["win"] for o in ("zoo.test_smart", "zoo.test_boss", "zoo.test_camper")]
    stud = [t["win"] for o, t in test.items() if o.startswith("net:")]
    n_test = sum(t["games"] for t in test.values())
    test_text = (f"On a second set of held-out test bots that was not used for any decision (100 games each): {rule[0]}, {rule[1]} and "
                 f"{rule[2]} wins against three rule-based bots, {min(stud)} to {max(stud)} wins against six baseline DQN networks, "
                 f"{100 * sum(t['loss'] for t in test.values()) / n_test:.1f} losses per 100 games overall.")
    val_rows = "\n".join(f"    | `{o}` | {w} | {d} | {l} |" for o, (w, d, l) in per.items())
    submission_src = (ROOT / "best/submission.py").read_text()
    league_v1 = (ROOT / "runs/rl/league_v1.txt").read_text().strip()
    league_v2 = (ROOT / "runs/rl/league_v2.txt").read_text().strip()

    cells = []
    cells.append(md(f"""
    # BOMBcompeTItion: reproducing our submitted network

    This notebook rebuilds the network we submit (`submission.py` + `submission.pth`), from the game engine up.

    **What the network is.** A 64-channel, 6-block residual CNN (about 0.59 M parameters, 1.4 ms per step on one CPU
    core) that maps the fixed `(13, 9, 9)` input to 6 action scores. It was trained **only by reinforcement learning**
    (PPO) against a league of opponents: no imitation data, no search at play time, no game logic inside `Net`.

    **How it was trained** (sections 4 and 5):

    1. *Stage 1*, 200 updates = 26 M environment steps from random weights, against league v1
       (hand-written rule bots, 40 bots of a parametrised family, noisy variants, self-play).
       Reward: win +1, draw -0.1, loss -0.4.
    2. *Stage 2*, 628 more updates = 82 M steps starting from the stage-1 network, against the wider league v2
       (200 parametrised bots, a search bot, six "student" DQN networks, more noisy variants, more self-play).
       Reward: win +1, draw -0.3, loss -0.5. A draw scores nothing on the leaderboard, and the larger draw penalty is
       what makes the agent hunt instead of just surviving.

    Which run to submit was decided on **held-out validation bots** that the network never trained against.

    **Two ways to run this notebook** (`MODE` in section 5):

    - `MODE = "restore"` (default): takes the trained weights from `bomb_best_assets.tar.gz` and rebuilds exactly the
      submitted `submission.zip` (md5 of the weights: `{md5}`). Runs in a few minutes anywhere, also on Colab.
    - `MODE = "train"`: runs both PPO stages from scratch with the original settings. About one hour on a 64-core
      machine with one GPU; on a 2-core Colab runtime the game simulation is the bottleneck and it takes many hours.
      Reinforcement learning with many worker processes is not bit-reproducible, so a retrained network plays
      similarly but is not identical to the submitted one.

    **Files.** Put `bomb_best_assets.tar.gz` next to this notebook (on Colab: upload it when the setup cell asks).
    It holds the trained weights and the student networks used as opponents. Without it, `restore` is not possible and
    `train` runs without the student opponents (5 % of league v2).

    **Local results of the submitted network** (argmax policy through `NeuralPolicy`, exactly as the server plays it;
    every map from both sides; maps never used in training):

    | Validation opponent (never trained against), 200 games each | W | D | L |
    |---|---|---|---|
{val_rows}

    Mean {grade['val']['score']:.1f} wins per 100 games, {grade['val']['loss']:.1f} losses per 100.
    {test_text}
    Against opponents that dodge well most of the remaining games are 300-step draws.
    """))

    cells.append(md("""
    ## 1. Install

    The course's Install cell, unchanged: it unpacks the game engine (`bombcompetition`) and the example bot.
    """))
    cells.append(new_code_cell(install_src))

    cells.append(md("""
    ## 2. Our code

    The next cell unpacks our training code into the folder `bomb_repro/` (it is embedded in the cell, like the engine
    above) and copies the engine there, so that folder is self-contained:

    | Path in `bomb_repro/` | What it is |
    |---|---|
    | `arena/core.py`, `arena/grade.py`, `arena/panel.py` | local grading harness: plays bots and networks with the reference engine, the way the server does |
    | `arena/nets.py` | the network definition (one source for training and for `submission.py`) and the export |
    | `arena/rl/vecenv.py`, `arena/rl/ppo.py` | multi-process environments on the reference engine and the PPO league trainer |
    | `arena/zoo/*.py` | opponent bots (section 4) |
    | `runs/students/*` | student DQN networks (from the assets file) |
    | `assets/best/` | the trained weights (from the assets file) |

    Our code is only ever run in subprocesses inside `bomb_repro/` (`arena(...)` below); nothing of it is imported
    into this notebook's kernel.
    """))
    setup_src = textwrap.dedent('''
    #@title Unpack our code and the assets { display-mode: "form" }
    import base64, hashlib, io, json, os, shutil, signal, subprocess, sys, tarfile, time
    from pathlib import Path

    CODE = """
    __CODE__
    """
    WORK = Path("bomb_repro").resolve()   # our code, the opponents and the training runs live here
    ASSETS = "bomb_best_assets.tar.gz"    # the trained weights and the student networks

    WORK.mkdir(exist_ok=True)
    shutil.rmtree(WORK / "arena", ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))) as tar:
        tar.extractall(WORK, filter="data")
    for d in ("bombcompetition", "bots"):  # the engine from the Install cell
        shutil.rmtree(WORK / d, ignore_errors=True)
        shutil.copytree(d, WORK / d, ignore=shutil.ignore_patterns("__pycache__"))

    assets = next((p for p in (Path(ASSETS), WORK / ASSETS, Path("/content") / ASSETS) if p.exists()), None)
    if assets is None:
        try:
            from google.colab import files
            print(f"Upload {ASSETS} now (cancel to go on without it).")
            uploaded = files.upload()
            assets = Path(next(iter(uploaded))) if uploaded else None
        except ImportError:
            pass
    if assets is not None:
        with tarfile.open(assets) as tar:
            tar.extractall(WORK, filter="data")

    HAVE_BEST = (WORK / "assets/best/submission.pth").exists()
    STUDENTS = sorted(p.name for p in (WORK / "runs/students").glob("train_*") if (p / "submission.pth").exists())
    CPUS = os.cpu_count() or 2


    def arena(*args):
        """Run `python -m <module> ...` inside bomb_repro/ and show its output, e.g. arena("arena.grade", "net:...", "--panel", "val")."""
        r = subprocess.run([sys.executable, "-m", *map(str, args)], cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(r.stdout)
        if r.returncode:
            raise RuntimeError(f"{args[0]} failed with exit code {r.returncode}")


    print(f"code: {len(list((WORK / 'arena').rglob('*.py')))} files in {WORK / 'arena'}")
    print(f"assets: {'found ' + str(assets) if assets else 'NOT FOUND'} | trained weights: {HAVE_BEST} | student networks for training: {len(STUDENTS)}")
    ''').strip("\n").replace("__CODE__", code_bundle())
    cells.append(new_code_cell(setup_src))

    cells.append(md("""
    ## 3. Submission cell

    **This cell becomes `submission.py`.** It is the file `arena/nets.py` generates for a 64-channel, 6-block network
    (`arena.nets.render_source`), so the class below is exactly the one the trainer optimises.

    - Learned layers only: a 3x3 stem, 6 pre-activation residual blocks with GroupNorm, and a small head that reads the
      tower at my cell and at the opponent's cell (the two input planes that mark them) plus global mean and max
      pooling. No rules, no search.
    - No BatchNorm or Dropout, so the scores are the same in `train()` and `eval()` mode.
    - `forward` returns the 6 action scores (the policy logits). The value head `v` is only used during training.
    """))
    cells.append(new_code_cell("%%writefile submission.py\n" + submission_src.rstrip("\n")))

    cells.append(md(f"""
    ## 4. Opponents: the league

    Every training game draws its opponent from a weighted list; opponents that the agent does not beat yet are drawn
    more often (prioritised sampling: weight x (0.1 + 0.9 x (1 - win rate))). All bots are in `bomb_repro/arena/zoo/`.

    | Opponent | How it plays | v1 weight | v2 weight |
    |---|---|---|---|
    | `random` | a random legal action | 0.5 | 0.3 |
    | `simple` | the course's SimpleBot | 0.5 | 0.3 |
    | `zoo.base_cautious` | dodges with exact fuse timing, bombs when the opponent is in its blast, chases | 1 | 1 |
    | `zoo.ta_smart_a` | our guess at "SmartBot": dodge, bomb when the opponent has at most one way out, leave the closing ring, BFS chase | 3 | 3 |
    | `zoo.ta_smart_b` | aggressive: bombs whenever the opponent is in range, up to 3 bombs to cut escapes; plans its own escape as if the opponent stood still | 1.5 | 1.5 |
    | `zoo.evader` | survival first: only actions that survive every opponent reply, avoids dead ends, bombs only when fully safe | 2 | 2 |
    | `zoo.ta_final_a` | strong: resolves every (my action, opponent action) pair exactly, three levels of safety checks, forced-kill detection | 3 | 3 |
    | `zoo.ta_final_b` | positional: bombs as temporary walls, squeezes the opponent's region, takes the centre before a shrink | 3 | 3 |
    | `zoo.param` | the strongest preset of the parametrised family | 1 | 1 |
    | `zoo.param:<seed>` | parametrised family: about 20 behaviour switches (what it knows about explosions, when it bombs, how it chases, how random it is, SimpleBot's freeze bug, ...) sampled from the seed | seeds 0-39, 0.2 each | seeds 0-199, 0.1 each |
    | `zoo.x_solver` | search bot: forced-win search, keeps only actions that are safe against every reply, then maximum pressure | - | 2 |
    | `noisy:<eps>:<bot>` | a strong bot that plays a random legal action with probability eps (teaches punishing mistakes) | 3 bots at 0.1, evader at 0.2 | 4 bots at 0.1, evader at 0.2, ta_smart_a at 0.25 |
    | `net:runs/students/train_*` | {len(train_students)} baseline DQN networks trained with the course notebook and typical student changes | - | 0.4 each |
    | `self` | the current network on both sides (both sides are training data) | 3 | 3 |
    | `past` | a frozen copy of the network from earlier in training (pool of 4) | 2 | 3 |

    **Held out.** `zoo.val_smart`, `zoo.val_hunter`, `zoo.val_tricky` and `zoo.param:9000`-`9005` are only used to compare
    runs and checkpoints; `zoo.test_smart`, `zoo.test_boss`, `zoo.test_camper` and the `test_*` student networks are only
    used for the final report. None of them is ever a training opponent.
    """))
    cells.append(code('''
    # The two leagues as "spec:weight" lists (the trainer's --league format).
    BOTS = ["zoo.base_cautious:1", "zoo.ta_smart_a:3", "zoo.ta_smart_b:1.5", "zoo.evader:2", "zoo.ta_final_a:3",
            "zoo.ta_final_b:3", "zoo.param:1"]

    LEAGUE_V1 = ",".join(
        ["simple:0.5", "random:0.5"] + BOTS
        + [f"zoo.param:{s}:0.2" for s in range(40)]
        + ["noisy:0.1:zoo.ta_smart_a:1", "noisy:0.1:zoo.ta_final_a:1", "noisy:0.1:zoo.ta_final_b:1", "noisy:0.2:zoo.evader:0.5"]
        + ["self:3", "past:2"])

    LEAGUE_V2 = ",".join(
        ["simple:0.3", "random:0.3"] + BOTS + ["zoo.x_solver:2"]
        + [f"zoo.param:{s}:0.1" for s in range(200)]
        + [f"noisy:0.1:{b}:0.7" for b in ("zoo.ta_smart_a", "zoo.ta_final_a", "zoo.ta_final_b", "zoo.x_solver")]
        + ["noisy:0.2:zoo.evader:0.5", "noisy:0.25:zoo.ta_smart_a:0.5"]
        + [f"net:runs/students/{s}:0.4" for s in STUDENTS]
        + ["self:3", "past:3"])

    # graded after every EVAL_EVERY updates (argmax policy, 100 games each); never trained against
    EVAL_OPP = "zoo.val_smart,zoo.val_hunter,zoo.val_tricky,zoo.param:9000,zoo.param:9001,zoo.param:9002,zoo.param:9003"

    print(f"league v1: {LEAGUE_V1.count(',') + 1} opponents | league v2: {LEAGUE_V2.count(',') + 1} opponents"
          + ("" if STUDENTS else "  (WARNING: no student networks found, league v2 runs without them)"))
    '''))

    cells.append(md("""
    ## 5. Training: two PPO stages

    The trainer is `bomb_repro/arena/rl/ppo.py` (`python -m arena.rl.ppo --help` lists every option). In short:

    - **Environments.** 2048 games in parallel on the reference engine (64 worker processes x 32 games), the learner's
      side chosen at random, a new random map every game. The learner sees `preprocess(turn(obs))` and samples from
      softmax(scores); a blocked move simply acts as STAY (no action masking, because the server takes a plain argmax).
    - **PPO.** 64 steps per environment per update (131,072 steps per update), GAE (gamma 0.995, lambda 0.95),
      2 epochs, minibatches of 8192, clip 0.2, Adam with learning rate 3e-4 (5 warm-up updates, cosine decay to 10 %),
      entropy bonus 0.02 falling to 0.005, gradient clipping 1.0.
    - **Reward.** Only at the end of a game: win +1; draw and loss as given per stage. No shaping.
    - **Checkpoints.** Every 40 updates the network is exported and its argmax policy is graded on the validation bots
      (100 games each) in the background.

    | | Stage 1 | Stage 2 |
    |---|---|---|
    | start | random weights, seed 0 | the stage-1 network after 200 updates, seed 3 |
    | league | v1 | v2 |
    | reward for draw / loss | -0.1 / -0.4 | -0.3 / -0.5 |
    | schedules run over | 300 M steps | 120 M steps |
    | stopped after | 200 updates (26.2 M steps) | 628 updates (82.3 M steps) |
    | extra | | the first 3 updates train only the value head |

    The submitted network is the stage-2 network after its last update (628). Stage 2 ran under a wall-clock limit in
    the original run, which ended it there; `--stop-update` reproduces the same stopping point exactly.
    """))
    cells.append(code(f'''
    import torch

    MODE = "restore"   # "restore": the trained weights from the assets file | "train": run both stages from scratch

    # the original run (change these only for a quick test)
    STAGE1_UPDATES, STAGE2_UPDATES = 200, 628
    EVAL_EVERY, EVAL_GAMES = 40, 100
    N_ENVS = 2048
    GRADE_GAMES = 100                                   # games per opponent in section 6

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    WORKERS = min(64, max(1, CPUS - 2))                 # 64 in the original run
    PROCS = max(1, min(96, CPUS - 1))                   # processes for grading
    EXPECTED_MD5 = "{md5}"       # the submitted weights

    PPO = ["--rollout", 64, "--epochs", 2, "--minibatch", 8192, "--gamma", 0.995, "--lam", 0.95, "--clip", 0.2,
           "--vf-coef", 0.5, "--max-grad-norm", 1.0, "--target-kl", 0.03, "--lr", 3e-4, "--lr-final", 0.1, "--lr-warmup", 5,
           "--ent-coef", 0.02, "--ent-coef-final", 0.005, "--pfsp-power", 1, "--pfsp-floor", 0.1, "--pfsp-decay", 0.9,
           "--snapshots", 4, "--snap-every", 25, "--snap-greedy", 0.5, "--reward-win", 1]


    def evals(out):
        f = Path(out) / "evals.jsonl"
        return [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []


    def run_stage(name, args, stop_update):
        """Run the trainer for exactly `stop_update` updates; it then exports <out>/final and grades it."""
        out = WORK / "runs" / "rl" / name
        meta = out / "final" / "meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("update") == stop_update:
            print(f"{{name}}: already done ({{out}})")
            return out
        shutil.rmtree(out, ignore_errors=True)
        cmd = [sys.executable, "-u", "-m", "arena.rl.ppo", "--out", f"runs/rl/{{name}}", "--device", DEVICE,
               "--workers", WORKERS, "--envs-per-worker", max(1, N_ENVS // WORKERS), "--groups", 2 if WORKERS > 1 else 1,
               "--eval-opp", EVAL_OPP, "--eval-every", EVAL_EVERY, "--eval-games", EVAL_GAMES,
               "--eval-procs", max(1, min(12, CPUS // 4)), "--stop-update", stop_update, *PPO, *args]
        proc = subprocess.Popen(list(map(str, cmd)), cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        try:
            for line in proc.stdout:
                if not line.startswith("[ppo] u") or int(line.split()[1][1:]) % 20 == 0:
                    print(line.rstrip()[:260 if line.startswith("[eval]") else 150], flush=True)
            if proc.wait() != 0 or not meta.exists():
                raise RuntimeError(f"the trainer failed (exit code {{proc.returncode}})")
        finally:
            if proc.poll() is None:  # interrupted: stop the trainer and its workers (they form one process group)
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                        proc.wait(timeout=30)
                        break
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
        return out


    print(f"MODE = {{MODE}} | device {{DEVICE}} | {{CPUS}} CPUs -> {{WORKERS}} workers x {{max(1, N_ENVS // WORKERS)}} environments")
    '''))
    cells.append(code('''
    # Stage 1: from random weights against league v1 (draw -0.1, loss -0.4).
    if MODE == "train":
        STAGE1 = run_stage("stage1", ["--league", LEAGUE_V1, "--channels", 64, "--blocks", 6, "--norm", "gn", "--head", "gather",
                                      "--head-hidden", 256, "--reward-draw", -0.1, "--reward-loss", -0.4,
                                      "--total-steps", 300_000_000, "--seed", 0], STAGE1_UPDATES)
    else:
        print("MODE = restore: nothing to train")
    '''))
    cells.append(code('''
    # Stage 2: continue from the stage-1 network against league v2 (draw -0.3, loss -0.5).
    if MODE == "train":
        STAGE2 = run_stage("stage2", ["--league", LEAGUE_V2, "--init", "runs/rl/stage1/final", "--value-warmup", 3,
                                      "--reward-draw", -0.3, "--reward-loss", -0.5, "--total-steps", 120_000_000, "--seed", 3],
                           STAGE2_UPDATES)
        for e in evals(STAGE2):  # the validation history (100 games per opponent, so it is noisy)
            print(f"update {e['update']:>4}: validation mean win rate {e['mean_win_rate']:.3f}, loss rate {e['mean_loss_rate']:.3f}")
        FINAL = STAGE2 / "final"  # the network after the last update
    else:
        assert HAVE_BEST, "MODE = restore needs bomb_best_assets.tar.gz next to the notebook (see section 2)"
        FINAL = WORK / "assets" / "best"
    print("final network:", FINAL)
    '''))
    cells.append(code('''
    # Load the final weights into the Net of the submission cell (strict: the names and shapes must match).
    import importlib

    import submission
    importlib.reload(submission)  # picks up the submission cell
    from submission import Net

    net = Net()
    net.load_state_dict(torch.load(FINAL / "submission.pth", map_location="cpu"))
    net.eval()
    md5 = hashlib.md5((FINAL / "submission.pth").read_bytes()).hexdigest()
    print(f"{sum(p.numel() for p in net.parameters()):,} parameters | weights md5 {md5}"
          + (" (= the submitted network)" if md5 == EXPECTED_MD5 else " (a retrained network)"))
    if MODE == "restore":
        assert md5 == EXPECTED_MD5, "the assets file does not hold the submitted weights"
    # the trainer exported the same class: the two files must be the same network definition
    same = (FINAL / "submission.py").read_text().strip() == Path("submission.py").read_text().strip()
    print("submission cell == the trainer's exported submission.py:", same)
    '''))

    cells.append(md("""
    ## 6. Local evaluation

    First the course's check against SimpleBot, then our own grading, which plays the network exactly as the server
    does (`NeuralPolicy`, argmax) against the held-out bots: `GRADE_GAMES` games per opponent, every map from both
    sides, maps that were never used in training. Columns: W D L, `self` = games we lost to our own bomb,
    `oppself` = opponent self-bombs, `t/o` = draws by the 300-step limit.
    """))
    cells.append(code('''
    # The course's local evaluation (section 6 of the template), against SimpleBot.
    random.seed(0)
    total = {"win": 0, "draw": 0, "loss": 0}
    for i in range(20):
        side = i % 2
        mine = lambda: NeuralPolicy(net)  # exactly how the server plays the network
        pair = (mine, SimpleBot) if side == 0 else (SimpleBot, mine)
        game, actions, bots = play_game(*pair, random_rules(seed=1000 + i // 2))
        w = game.winner()
        total["win" if w == side else "draw" if w is None else "loss"] += 1
    print(f"vs SimpleBot: {total['win']}W {total['draw']}D {total['loss']}L")
    '''))
    cells.append(code('''
    # Held-out validation bots (used to choose the checkpoint) and test bots (not used for any decision).
    arena("arena.grade", f"net:{FINAL}", "--panel", "val", "--games", GRADE_GAMES, "--procs", PROCS)
    arena("arena.grade", f"net:{FINAL}", "--panel", "test", "--games", GRADE_GAMES, "--procs", PROCS)
    '''))

    cells.append(md("""
    ## 7. Make the submission

    The course's last cell, unchanged: it saves `submission.pth` from `net`, plays one game in a fresh Python process in
    a folder that holds only the two files, and builds `submission.zip`.
    """))
    cells.append(new_code_cell(last_src))
    cells.append(code('''
    # The zip holds exactly the network that was graded above.
    import zipfile

    with zipfile.ZipFile("submission.zip") as zf:
        state = torch.load(io.BytesIO(zf.read("submission.pth")), map_location="cpu")
    ref = torch.load(FINAL / "submission.pth", map_location="cpu")
    assert state.keys() == ref.keys() and all(torch.equal(state[k], ref[k]) for k in ref)
    print("submission.zip is ready:", Path("submission.zip").resolve())
    '''))

    nb = new_notebook(cells=cells, metadata={
        "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.validate(nb)
    nbformat.write(nb, OUT_NB)

    # the league strings written in the notebook must be the ones the original runs used
    ns: dict = {"STUDENTS": train_students}
    exec(cells[8].source, ns)  # noqa: S102
    assert ns["LEAGUE_V1"] == league_v1, "league v1 differs from runs/rl/league_v1.txt"
    assert ns["LEAGUE_V2"] == league_v2, "league v2 differs from runs/rl/league_v2.txt"
    print(f"wrote {OUT_NB} ({OUT_NB.stat().st_size / 1e6:.2f} MB, {len(cells)} cells)")
    print(f"wrote {OUT_ASSETS} ({OUT_ASSETS.stat().st_size / 1e6:.1f} MB): best weights md5 {md5}, students {len(train_students)} train + {len(test_students)} test")


if __name__ == "__main__":
    main()
