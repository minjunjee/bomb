"""PPO league training of the BOMB network (the submission is the ARGMAX of this network's 6 scores).

    nice -n 10 .venv/bin/python -m arena.rl.ppo --out runs/rl/test1 --device cuda:1 --workers 64 --envs-per-worker 32 \
        --league "simple:1,random:1,zoo.base_cautious:4,self:2" --eval-opp simple,zoo.base_cautious --eval-every 20 \
        --total-steps 20000000

What it does
    rollout   W x E environments on the reference engine (arena.rl.vecenv), batched GPU inference in this process;
              the learner SAMPLES from softmax(scores) (no action masking: a blocked move is a legal STAY, exactly
              what the exported argmax network can express).
    update    PPO-clip + GAE on the GPU, large minibatches, entropy bonus, optional KL penalty toward the frozen
              initial network, optional random board-symmetry augmentation (arena.nets.augment, see --aug).
    league    every episode draws its opponent from the league; bot entries are re-weighted toward opponents the
              sampled policy does not beat yet (PFSP); "self" = current weights on both sides (both sides give
              training data), "past" = a frozen snapshot of the learner on the other side.
    eval      every --eval-every updates the network is exported to <out>/ckpt_<update>/ and graded in the background
              like the server ("python -m arena.core leaderboard", ARGMAX policy, evaluation maps >= 1_000_000);
              results go to <out>/evals.jsonl and <out>/best/ is the checkpoint with the best mean win rate
              (ties: lower loss rate).

Files in <out> (keep it under runs/): log.jsonl (one line per update), evals.jsonl, ckpt_<n>/, best/, final/,
state.pt (for --resume), args.json, eval_logs/.

Options
  run
    --out DIR             output folder, e.g. runs/rl/<name> (required)
    --device DEV          torch device of the learner (default cuda:1)
    --seed N              seeds torch, numpy and every environment stream (default 0)
    --resume              continue from <out>/state.pt (network, optimizer, schedules, league statistics, snapshots,
                          best evaluation); the environments start new games. Without it an existing state.pt is an error.
    --total-steps N       environment steps to train for (one step = one environment transition of the learner's
                          side; self-play second sides come on top) (default 20e6). The schedules run over this.
    --time-limit SEC      stop (cleanly, with the final export and evaluation) after SEC seconds of wall time in this
                          session even if --total-steps is not reached (default 0 = no limit)
    --stop-update N       stop (cleanly, with the final export and evaluation) once update N is done (default 0 = off).
                          Unlike --total-steps it does not change the schedules: use it to end a run at an exact update.
  environments
    --workers W           worker processes (default 64)
    --envs-per-worker E   environments per worker (default 32)
    --groups G            worker groups stepped alternately while the GPU serves the other group (default 2; 1 = synchronous)
    --league TEXT         "spec:weight,..." (default "simple:1,random:1,zoo.base_cautious:4,self:2"). spec = any harness
                          spec (random, simple, zoo.x, zoo.x:arg, noisy:<eps>:<spec>, net:<folder>, file.py), "self"
                          or "past". The last ":<number>" is the weight.
    --reward-win / --reward-draw / --reward-loss   terminal rewards (default 1 / 0 / -0.5); no shaping
    --pfsp-power P        bot entries are sampled with weight * (floor + (1-floor) * (1 - winrate)^P), renormalised so
                          that bots keep their total share against self/past (default 1; 0 = fixed weights)
    --pfsp-floor F        see above (default 0.1)
    --pfsp-decay D        decay per update of the win/episode counters behind `winrate` (default 0.9)
    --snapshots K         size of the pool of frozen past networks for "past" (default 4)
    --snap-every N        every N updates the oldest snapshot is replaced by the current weights (default 25)
    --snap-greedy P       a snapshot plays argmax with probability P per step, else it samples (default 0.5)
  network
    --init FOLDER         start from an exported network (arena.nets.load_submission; its cfg wins)
    --channels / --blocks / --norm / --head / --head-hidden    arena.nets.build_net cfg for a network from scratch
                          (default 64 / 4 / gn / gather / 256)
  PPO
    --rollout T           steps per environment per update (default 64)
    --epochs K            passes over the batch per update (default 2)
    --minibatch B         minibatch size (default 8192)
    --gamma / --lam       discount and GAE lambda (default 0.995 / 0.95); episode ends (hit or 300 steps) are true ends
    --clip C              PPO clip range (default 0.2)
    --vf-coef C           value loss weight (default 0.5)
    --max-grad-norm G     gradient clipping (default 1.0)
    --target-kl K         stop the epochs of an update when the approximate KL exceeds 1.5 K (default 0.03; 0 = off)
    --lr LR               Adam learning rate (default 3e-4)
    --lr-final F          final learning rate as a fraction of --lr, cosine decay over the run (default 0.1)
    --lr-warmup N         linear warm-up over the first N updates (default 5)
    --ent-coef / --ent-coef-final   entropy bonus, linear schedule over the run (default 0.02 -> 0.005)
    --kl-coef / --kl-coef-final / --kl-decay-steps   penalty KL(anchor || policy) toward the frozen INITIAL network,
                          linear from --kl-coef to --kl-coef-final over --kl-decay-steps environment steps
                          (default 0 / 0 / total steps; 0 = no anchor)
    --value-warmup K      for the first K updates only the value head is trained (the policy and the tower are
                          frozen), so that a pretrained policy is not wrecked by an untrained critic (default 0)
    --aug N               random board symmetries: 1 (off, default), 2 (identity + transpose) or 8. Every episode
                          gets one random symmetry k; the learner sees augment(x, k), acts there, and the action is
                          mapped back for the engine. The update batch is therefore made of randomly turned boards
                          WITH the exact behaviour log-probabilities (plain PPO on the symmetrised game). Augmenting
                          only inside the update is not sound for PPO: the stored log-probability belongs to the
                          un-turned board, and recomputing it on the turned board gives ratios against tiny
                          probabilities - that variant diverged within 20 updates here.
                          Only the identity view is graded (sym = 1 networks), so 8 spends 7/8 of the experience
                          on views the server never shows: use it for networks that must stay symmetric.
    --amp MODE            none (fp32 + TF32, default) | bf16 (autocast: about 1.3x faster updates, coarser log-probs)
  evaluation and output
    --eval-opp LIST       comma-separated harness specs graded against (default simple,zoo.base_cautious; "" = no evaluation)
    --eval-every N        export + evaluate every N updates (default 20); one evaluation runs at a time, a newer
                          checkpoint replaces a waiting one
    --eval-games G        games per opponent (default 100, like the server)
    --eval-procs P        processes of the evaluation subprocess (default 8). CPU budget = workers + eval-procs + 2.
    --save-every N        write state.pt every N updates (default 10)
    --keep-ckpts N        keep only the newest N evaluated ckpt_<n> folders besides best/ (default 0 = keep all)
    --quiet               no progress lines on stdout
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.nets import SYM_ACT_INV, augment, build_net, export_submission, load_submission  # noqa: E402
from arena.rl.vecenv import DRAW, KIND_BOT, KIND_PAST, KIND_SELF, LOSS, WIN, VecEnv, parse_league  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--total-steps", type=float, default=20e6)
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--stop-update", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--envs-per-worker", type=int, default=32)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--league", default="simple:1,random:1,zoo.base_cautious:4,self:2")
    ap.add_argument("--reward-win", type=float, default=1.0)
    ap.add_argument("--reward-draw", type=float, default=0.0)
    ap.add_argument("--reward-loss", type=float, default=-0.5)
    ap.add_argument("--pfsp-power", type=float, default=1.0)
    ap.add_argument("--pfsp-floor", type=float, default=0.1)
    ap.add_argument("--pfsp-decay", type=float, default=0.9)
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--snap-every", type=int, default=25)
    ap.add_argument("--snap-greedy", type=float, default=0.5)
    ap.add_argument("--init", default=None)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--norm", default="gn")
    ap.add_argument("--head", default="gather")
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final", type=float, default=0.1)
    ap.add_argument("--lr-warmup", type=int, default=5)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--ent-coef-final", type=float, default=0.005)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--kl-coef-final", type=float, default=0.0)
    ap.add_argument("--kl-decay-steps", type=float, default=0.0)
    ap.add_argument("--value-warmup", type=int, default=0)
    ap.add_argument("--aug", type=int, default=1, choices=(1, 2, 8))
    ap.add_argument("--amp", default="none", choices=("bf16", "none"))
    ap.add_argument("--eval-opp", default="simple,zoo.base_cautious")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-procs", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--keep-ckpts", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap


def gae(rew, done, val, last_val, gamma: float, lam: float):
    """rew, done, val: (T, N); last_val: (N,) = V(observation after the last step). done[t] = 1: step t ended the
    episode (a true end: nothing is bootstrapped across it). Returns (advantages, returns)."""
    adv = torch.zeros_like(rew)
    run = torch.zeros_like(last_val)
    nxt = last_val
    for t in range(rew.shape[0] - 1, -1, -1):
        live = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * live - val[t]
        run = delta + gamma * lam * live * run
        adv[t] = run
        nxt = val[t]
    return adv, adv + val


# ----------------------------------------------------------------------------------------------------------------
# background evaluation of the ARGMAX policy with the harness
# ----------------------------------------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, out: Path, opps: list[str], games: int, procs: int, keep: int = 0):
        self.out, self.opps, self.games, self.procs, self.keep = out, opps, games, procs, keep
        self.proc, self.job, self.waiting = None, None, None
        self.best = None
        self.done_folders: list[Path] = []
        if (out / "best" / "eval.json").exists():
            self.best = json.loads((out / "best" / "eval.json").read_text())
        (out / "eval_logs").mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self):
        return bool(self.opps)

    def submit(self, folder: Path, meta: dict):
        """Evaluate `folder` (an exported checkpoint) as soon as the evaluator is free; a newer submission replaces a
        waiting one (whose folder is removed)."""
        if not self.enabled:
            return
        if self.waiting is not None and self.waiting[0] != folder:
            shutil.rmtree(self.waiting[0], ignore_errors=True)
        self.waiting = (folder, meta)
        self.poll()

    def _start(self, folder: Path, meta: dict):
        js = self.out / "eval_logs" / f"{folder.name}.json"
        cmd = [sys.executable, "-m", "arena.core", "leaderboard", "--me", f"net:{folder.resolve()}", "--opp",
               ",".join(self.opps), "--games", str(self.games), "--procs", str(self.procs), "--json", str(js)]
        delta = max(0, 10 - os.nice(0))
        if delta:
            cmd = ["nice", "-n", str(delta)] + cmd
        log = open(self.out / "eval_logs" / f"{folder.name}.log", "w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}  # the graded policy runs on one CPU core, like the server
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        log.close()
        self.job = (folder, meta, js, time.time())

    def poll(self) -> dict | None:
        """Collect a finished evaluation (returns its record) and start the waiting one."""
        rec = None
        if self.proc is not None and self.proc.poll() is not None:
            rec = self._collect()
        if self.proc is None and self.waiting is not None:
            (folder, meta), self.waiting = self.waiting, None
            self._start(folder, meta)
        return rec

    def _collect(self) -> dict | None:
        folder, meta, js, t0 = self.job
        code, self.proc, self.job = self.proc.returncode, None, None
        if code != 0 or not js.exists():
            print(f"[eval] FAILED for {folder.name} (exit {code}); see {self.out / 'eval_logs' / (folder.name + '.log')}", flush=True)
            return None
        res = json.loads(js.read_text())
        rec = {**meta, "ckpt": folder.name, "mean_win_rate": res["mean_win_rate"], "min_win_rate": res["min_win_rate"],
               "mean_loss_rate": res["mean_loss_rate"], "mean_draw_rate": res["mean_draw_rate"], "faults": res["faults"],
               "eval_seconds": round(time.time() - t0, 1), "games": res["games"],
               "per_opp": {o: {k: t[k] for k in ("win", "draw", "loss", "self_bomb", "timeout_draw", "win_rate",
                                                  "loss_rate", "avg_steps")} for o, t in res["per_opp"].items()}}
        with open(self.out / "evals.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        key = (rec["mean_win_rate"], -rec["mean_loss_rate"])
        if self.best is None or key > (self.best["mean_win_rate"], -self.best["mean_loss_rate"]):
            self.best = rec
            best = self.out / "best"
            best.mkdir(exist_ok=True)
            for name in ("submission.pth", "submission.py", "meta.json"):
                shutil.copy2(folder / name, best / (name + ".tmp"))
                os.replace(best / (name + ".tmp"), best / name)
            (best / "eval.json").write_text(json.dumps(rec, indent=1))
        self.done_folders.append(folder)
        if self.keep > 0:
            while len(self.done_folders) > self.keep:
                shutil.rmtree(self.done_folders.pop(0), ignore_errors=True)
        return rec

    def finish(self, timeout: float = 1800.0) -> list[dict]:
        """Wait for the running and the waiting evaluation."""
        recs, t0 = [], time.time()
        while (self.proc is not None or self.waiting is not None) and time.time() - t0 < timeout:
            rec = self.poll()
            if rec:
                recs.append(rec)
            if self.proc is not None:
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        return recs

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        self.proc = None


# ----------------------------------------------------------------------------------------------------------------
# the trainer
# ----------------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, args, env: VecEnv, state: dict | None = None):
        self.args, self.env = args, env
        self.dev = torch.device(args.device)
        self.amp = args.amp == "bf16" and self.dev.type == "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if state is not None:
            net = build_net(state["cfg"])
            net.load_state_dict(state["net"])
        elif args.init:
            folder = Path(args.init)
            net = load_submission(folder if folder.is_absolute() else ROOT / folder)
        else:
            net = build_net({"channels": args.channels, "blocks": args.blocks, "norm": args.norm, "head": args.head,
                             "head_hidden": args.head_hidden, "sym": 1})
        self.cfg = dict(net.cfg)
        self.net = net.to(self.dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, eps=1e-5)
        self.update_idx, self.steps, self.elapsed = 0, 0, 0.0
        self.kinds = env.kinds
        self.has_self = bool((self.kinds == KIND_SELF).any())
        self.has_past = bool((self.kinds == KIND_PAST).any())
        self.is_bot = self.kinds == KIND_BOT
        S = len(env.specs)
        self.cnt_w, self.cnt_n = np.zeros(S), np.zeros(S)  # decayed wins / episodes per league entry
        self.rng = np.random.default_rng(args.seed + 17)
        # frozen networks
        self.anchor = None
        if args.kl_coef > 0 or args.kl_coef_final > 0:
            self.anchor = self._frozen(self.net)
            if state is not None and state.get("anchor") is not None:
                self.anchor.load_state_dict(state["anchor"])
        self.snaps, self.snap_ptr = [], 0
        if self.has_past:
            self.snaps = [self._frozen(self.net) for _ in range(max(1, args.snapshots))]
            if state is not None:
                for s, sd in zip(self.snaps, state.get("snaps", [])):
                    s.load_state_dict(sd)
                self.snap_ptr = state.get("snap_ptr", 0)
        self.snap_id = self.rng.integers(0, max(1, len(self.snaps)), env.N)
        self.aug = int(args.aug)
        self.sym_np = self.rng.integers(0, self.aug, env.N)  # the board symmetry of every environment's episode
        self.sym_act_inv = torch.as_tensor(SYM_ACT_INV, device=self.dev)  # [k, action in view k] -> real action
        if state is not None:
            self.opt.load_state_dict(state["opt"])
            self.update_idx, self.steps, self.elapsed = state["update"], state["steps"], state["elapsed"]
            self.cnt_w, self.cnt_n = np.array(state["cnt_w"]), np.array(state["cnt_n"])
            if len(self.cnt_w) != S:  # the league changed: start its statistics again
                self.cnt_w, self.cnt_n = np.zeros(S), np.zeros(S)
            torch.set_rng_state(state["torch_rng"])
            if self.dev.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], self.dev)
            self.rng.bit_generator.state = state["np_rng"]
        self._frozen_policy = None
        # rollout storage
        T, N, dev = args.rollout, env.N, self.dev
        self.obs = torch.zeros((T, N, 13, 9, 9), device=dev)
        self.act = torch.zeros((T, N), dtype=torch.long, device=dev)
        self.logp = torch.zeros((T, N), device=dev)
        self.val = torch.zeros((T, N), device=dev)
        self.rew_np, self.done_np = np.zeros((T, N), np.float32), np.zeros((T, N), np.float32)
        if self.has_self:
            self.obs2 = torch.zeros((T, N, 13, 9, 9), device=dev)
            self.act2 = torch.zeros((T, N), dtype=torch.long, device=dev)
            self.logp2 = torch.zeros((T, N), device=dev)
            self.val2 = torch.zeros((T, N), device=dev)
            self.rew2_np, self.valid2_np = np.zeros((T, N), np.float32), np.zeros((T, N), bool)
        self.ready = [True] * env.G  # the group's observations are in shared memory (nothing pending)
        self.counts = np.zeros((S, 4), np.int64)
        self.len_sum, self.t_wait = 0, 0.0
        self._apply_pfsp()

    # -- helpers --------------------------------------------------------------------------------------------
    def _frozen(self, net):
        f = copy.deepcopy(net).eval()
        for p in f.parameters():
            p.requires_grad_(False)
        return f

    def _forward(self, net, x):
        with torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
            logits, value = net(x)
        return logits.float(), value.float()

    def _apply_pfsp(self):
        """League distribution: bots re-weighted toward the ones the sampled policy does not beat yet."""
        a, env = self.args, self.env
        w = env.weights.copy()
        if a.pfsp_power > 0 and self.is_bot.any():
            wr = (self.cnt_w + 0.0) / np.maximum(self.cnt_n, 1e-9)
            wr = np.where(self.cnt_n > 1.0, wr, 0.0)  # unknown yet = not beaten
            f = a.pfsp_floor + (1 - a.pfsp_floor) * np.clip(1 - wr, 0, 1) ** a.pfsp_power
            bots = w * f * self.is_bot
            if bots.sum() > 0:
                bots *= (w * self.is_bot).sum() / bots.sum()  # bots keep their total share against self / past
                w = np.where(self.is_bot, bots, w)
        env.set_league(w)

    def _sched(self):
        a = self.args
        frac = min(1.0, self.steps / max(1.0, a.total_steps))
        lr = a.lr * (a.lr_final + (1 - a.lr_final) * 0.5 * (1 + math.cos(math.pi * frac)))
        if a.lr_warmup > 0:
            lr *= min(1.0, (self.update_idx + 1) / a.lr_warmup)
        ent = a.ent_coef + (a.ent_coef_final - a.ent_coef) * frac
        kd = a.kl_decay_steps or a.total_steps
        kl = a.kl_coef + (a.kl_coef_final - a.kl_coef) * min(1.0, self.steps / max(1.0, kd))
        return lr, ent, (kl if self.anchor is not None else 0.0)

    # -- rollout --------------------------------------------------------------------------------------------
    def _record(self, t: int, g: int):
        """Results of the actions taken at time t by group g (it has just finished stepping)."""
        env = self.env
        lo, hi = env.group_rows[g]
        d = env.done[lo:hi]
        self.rew_np[t, lo:hi] = env.rew0[lo:hi]
        self.done_np[t, lo:hi] = d
        if self.has_self:
            self.rew2_np[t, lo:hi] = env.rew1[lo:hi]
        fin = np.nonzero(d)[0]
        if len(fin):
            np.add.at(self.counts, (env.ep_opp[lo:hi][fin], env.result[lo:hi][fin]), 1)
            self.len_sum += int(env.ep_len[lo:hi][fin].sum())
            if self.has_past:
                self.snap_id[lo + fin] = self.rng.integers(0, len(self.snaps), len(fin))
            if self.aug > 1:  # the next episode of these environments is seen through a new symmetry
                self.sym_np[lo + fin] = self.rng.integers(0, self.aug, len(fin))

    def _act(self, t: int, g: int):
        env, dev = self.env, self.dev
        lo, hi = env.group_rows[g]
        n = hi - lo
        x0 = torch.from_numpy(env.obs0[lo:hi]).to(dev)
        k = None
        if self.aug > 1:
            k = torch.from_numpy(self.sym_np[lo:hi]).to(dev)
            x0 = augment(x0, k=k)[0]
        kind = self.kinds[env.cur_opp[lo:hi]]
        rows, x = None, x0
        if self.has_self:
            rows = np.nonzero(kind == KIND_SELF)[0]
            if len(rows):
                x1 = torch.from_numpy(env.obs1[lo:hi][rows]).to(dev)
                rows_t = torch.from_numpy(rows).to(dev)
                if k is not None:
                    x1 = augment(x1, k=k[rows_t])[0]
                    k = torch.cat([k, k[rows_t]])
                x = torch.cat([x0, x1])
            else:
                rows = None
        logits, value = self._forward(self.net, x)
        lp_all = F.log_softmax(logits, -1)
        a = torch.multinomial(lp_all.exp(), 1).squeeze(1)  # an action of the board as the network saw it
        lp = lp_all.gather(1, a[:, None]).squeeze(1)
        self.obs[t, lo:hi], self.act[t, lo:hi], self.logp[t, lo:hi], self.val[t, lo:hi] = x0, a[:n], lp[:n], value[:n]
        a_np = (a if k is None else self.sym_act_inv[k, a]).cpu().numpy()  # back to the un-augmented turned frame
        env.act0[lo:hi] = a_np[:n]
        if rows is not None:
            r = rows_t + lo
            self.obs2[t, r], self.act2[t, r], self.logp2[t, r], self.val2[t, r] = x1, a[n:], lp[n:], value[n:]
            self.valid2_np[t, lo + rows] = True
            env.act1[lo + rows] = a_np[n:]
        if self.has_past:
            prow = np.nonzero(kind == KIND_PAST)[0]
            if len(prow):
                sid = self.snap_id[lo + prow]
                for k, snap in enumerate(self.snaps):
                    rk = prow[sid == k]
                    if len(rk):
                        lg, _ = self._forward(snap, torch.from_numpy(env.obs1[lo:hi][rk]).to(dev))
                        ak = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(1)
                        if self.args.snap_greedy > 0:
                            greedy = torch.rand(len(rk), device=dev) < self.args.snap_greedy
                            ak = torch.where(greedy, lg.argmax(1), ak)
                        env.act1[lo + rk] = ak.cpu().numpy()

    @torch.no_grad()
    def collect(self):
        """One rollout of T steps for every environment. Groups are stepped alternately: while the workers of one
        group play, the network answers the other group."""
        env, T = self.env, self.args.rollout
        self.counts[:] = 0
        self.len_sum, self.t_wait = 0, 0.0
        if self.has_self:
            self.valid2_np[:] = False
        busy0 = int(env.busy_ns.sum())
        t0 = time.perf_counter()
        for t in range(T):
            for g in range(env.G):
                if not self.ready[g]:
                    tw = time.perf_counter()
                    env.wait(g)
                    self.t_wait += time.perf_counter() - tw
                    self._record(t - 1, g)
                self._act(t, g)
                env.send(g)
                self.ready[g] = False
        for g in range(env.G):
            tw = time.perf_counter()
            env.wait(g)
            self.t_wait += time.perf_counter() - tw
            self._record(T - 1, g)
            self.ready[g] = True
        # values of the observations after the last step (bootstrap for unfinished episodes)
        k = torch.from_numpy(self.sym_np).to(self.dev) if self.aug > 1 else None
        x = torch.from_numpy(env.obs0).to(self.dev)
        _, self.last_val = self._forward(self.net, x if k is None else augment(x, k=k)[0])
        if self.has_self:
            self.last_val2 = torch.zeros(env.N, device=self.dev)
            rows = np.nonzero(self.kinds[env.cur_opp] == KIND_SELF)[0]
            if len(rows):
                rows_t = torch.from_numpy(rows).to(self.dev)
                x = torch.from_numpy(env.obs1[rows]).to(self.dev)
                _, v = self._forward(self.net, x if k is None else augment(x, k=k[rows_t])[0])
                self.last_val2[rows_t] = v
        dt = time.perf_counter() - t0
        return {"t_rollout": dt, "sps_rollout": env.N * T / dt, "wait_frac": self.t_wait / dt,
                "worker_busy": (int(env.busy_ns.sum()) - busy0) / 1e9 / (dt * env.W)}

    # -- update ---------------------------------------------------------------------------------------------
    def update(self) -> dict:
        a, dev = self.args, self.dev
        T, N = a.rollout, self.env.N
        lr, ent_coef, kl_coef = self._sched()
        for group in self.opt.param_groups:
            group["lr"] = lr
        rew, done = torch.from_numpy(self.rew_np).to(dev), torch.from_numpy(self.done_np).to(dev)
        adv, ret = gae(rew, done, self.val, self.last_val, a.gamma, a.lam)
        obs, act, logp_old = self.obs.view(T * N, 13, 9, 9), self.act.view(-1), self.logp.view(-1)
        adv, ret, val_old = adv.view(-1), ret.view(-1), self.val.view(-1)
        n_second = 0
        if self.has_self and self.valid2_np.any():
            valid = torch.from_numpy(self.valid2_np).to(dev)
            vf = valid.float()
            # outside "self" episodes the second side is not ours: cut the recursion there (done = 1, value = 0)
            adv2, ret2 = gae(torch.from_numpy(self.rew2_np).to(dev) * vf, torch.maximum(done, 1 - vf),
                             self.val2 * vf, self.last_val2, a.gamma, a.lam)
            idx = valid.view(-1).nonzero().squeeze(1)
            n_second = int(idx.numel())
            obs = torch.cat([obs, self.obs2.view(T * N, 13, 9, 9)[idx]])
            act = torch.cat([act, self.act2.view(-1)[idx]])
            logp_old = torch.cat([logp_old, self.logp2.view(-1)[idx]])
            adv, ret = torch.cat([adv, adv2.view(-1)[idx]]), torch.cat([ret, ret2.view(-1)[idx]])
            val_old = torch.cat([val_old, (self.val2 * vf).view(-1)[idx]])
        M = obs.shape[0]
        ev = float(1 - (ret - val_old).var() / (ret.var() + 1e-8))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        warm = self.update_idx < a.value_warmup
        if warm != (self._frozen_policy is True):
            for name, p in self.net.named_parameters():
                p.requires_grad_(name.startswith("v.") or not warm)
            self._frozen_policy = warm

        stats = {k: 0.0 for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "kl_anchor", "grad_norm")}
        n_mb, stop, skipped = 0, False, 0
        params = [p for p in self.net.parameters() if p.requires_grad]
        for _ in range(a.epochs):
            perm = torch.randperm(M, device=dev)
            for mb in torch.tensor_split(perm, max(1, round(M / a.minibatch))):
                x, y = obs[mb], act[mb]  # boards and actions exactly as the behaviour policy saw / chose them
                logits, value = self._forward(self.net, x)
                lp_all = F.log_softmax(logits, -1)
                lp = lp_all.gather(1, y[:, None]).squeeze(1)
                v_loss = 0.5 * (value - ret[mb]).pow(2).mean()
                entropy = -(lp_all.exp() * lp_all).sum(-1).mean()
                log_ratio = (lp - logp_old[mb]).clamp(-10.0, 10.0)
                ratio = log_ratio.exp()
                am = adv[mb]
                pg_loss = -torch.min(ratio * am, ratio.clamp(1 - a.clip, 1 + a.clip) * am).mean()
                kl_anchor = torch.zeros((), device=dev)
                if self.anchor is not None:
                    with torch.no_grad():
                        la = F.log_softmax(self._forward(self.anchor, x)[0], -1)
                    kl_anchor = (la.exp() * (la - lp_all)).sum(-1).mean()
                if warm:
                    loss = a.vf_coef * v_loss
                else:
                    loss = pg_loss + a.vf_coef * v_loss - ent_coef * entropy + kl_coef * kl_anchor
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
                if not (torch.isfinite(loss) and torch.isfinite(gn)):  # never write NaN into the weights
                    skipped += 1
                    continue
                self.opt.step()
                with torch.no_grad():
                    akl = float(((ratio - 1) - log_ratio).mean())
                    stats["pg_loss"] += float(pg_loss)
                    stats["v_loss"] += float(v_loss)
                    stats["entropy"] += float(entropy)
                    stats["approx_kl"] += akl
                    stats["clipfrac"] += float(((ratio - 1).abs() > a.clip).float().mean())
                    stats["kl_anchor"] += float(kl_anchor)
                    stats["grad_norm"] += float(gn)
                n_mb += 1
                if a.target_kl > 0 and not warm and akl > 1.5 * a.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: v / max(1, n_mb) for k, v in stats.items()}
        out.update(explained_var=ev, lr=lr, ent_coef=ent_coef, kl_coef=kl_coef, samples=M, samples_second_side=n_second,
                   minibatches=n_mb, skipped_minibatches=skipped, value_only=bool(warm))
        self.update_idx += 1
        self.steps += T * N
        if self.has_past and self.update_idx % max(1, a.snap_every) == 0:
            self.snaps[self.snap_ptr].load_state_dict(self.net.state_dict())
            self.snap_ptr = (self.snap_ptr + 1) % len(self.snaps)
        return out

    def league_stats(self) -> dict:
        """Per-opponent results of the SAMPLED policy in the last rollout; updates the PFSP statistics."""
        a, env, c = self.args, self.env, self.counts
        n = c[:, WIN] + c[:, DRAW] + c[:, LOSS]
        self.cnt_w = a.pfsp_decay * self.cnt_w + c[:, WIN]
        self.cnt_n = a.pfsp_decay * self.cnt_n + n
        self._apply_pfsp()
        per = {}
        for i, spec in enumerate(env.specs):
            m = max(1, int(n[i]))
            per[spec] = {"n": int(n[i]), "win": round(c[i, WIN] / m, 4), "draw": round(c[i, DRAW] / m, 4),
                         "loss": round(c[i, LOSS] / m, 4), "win_ema": round(float(self.cnt_w[i] / max(self.cnt_n[i], 1e-9)), 4),
                         "p": round(float(env.league_p[i]), 4)}
        tot = int(n.sum())
        return {"league": per, "episodes": tot, "ep_len": self.len_sum / max(1, tot), "bot_faults": int(env.faults.sum())}

    def state(self) -> dict:
        return {"cfg": self.cfg, "net": self.net.state_dict(), "opt": self.opt.state_dict(), "update": self.update_idx,
                "steps": self.steps, "elapsed": self.elapsed, "cnt_w": self.cnt_w.tolist(), "cnt_n": self.cnt_n.tolist(),
                "anchor": None if self.anchor is None else self.anchor.state_dict(),
                "snaps": [s.state_dict() for s in self.snaps], "snap_ptr": self.snap_ptr,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.dev) if self.dev.type == "cuda" else None,
                "np_rng": self.rng.bit_generator.state, "args": vars(self.args)}


def _save_state(trainer: Trainer, out: Path, resumes: int):
    st = trainer.state()
    st["resumes"] = resumes
    tmp = out / "state.pt.tmp"
    torch.save(st, tmp)
    os.replace(tmp, out / "state.pt")


def _fmt_eval(rec: dict) -> str:
    per = " | ".join(f"{o} {t['win']}-{t['draw']}-{t['loss']}" for o, t in rec["per_opp"].items())
    return (f"[eval] update {rec['update']} steps {rec['steps'] / 1e6:.2f}M  mean win {rec['mean_win_rate']:.3f} "
            f"loss {rec['mean_loss_rate']:.3f}  W-D-L: {per}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.total_steps = float(args.total_steps)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    state, resumes = None, 0
    if (out / "state.pt").exists():
        if not args.resume:
            raise SystemExit(f"{out}/state.pt exists: use --resume or another --out")
        state = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        resumes = state.get("resumes", 0) + 1
    elif args.resume:
        print(f"[ppo] nothing to resume in {out}: starting a new run", flush=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    torch.manual_seed(args.seed + 1000 * resumes)
    np.random.seed(args.seed + 1000 * resumes)
    torch.set_num_threads(2)

    def on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    league = parse_league(args.league)
    eval_opps = [s for s in args.eval_opp.split(",") if s.strip()]
    evaluator = Evaluator(out, eval_opps, args.eval_games, args.eval_procs, args.keep_ckpts)
    # The workers are started BEFORE this process touches CUDA (fork after CUDA initialisation breaks them).
    env = VecEnv(league, workers=args.workers, envs_per_worker=args.envs_per_worker, groups=args.groups,
                 seed=args.seed + 7919 * resumes, reward_win=args.reward_win, reward_draw=args.reward_draw,
                 reward_loss=args.reward_loss)
    interrupted = False
    try:
        trainer = Trainer(args, env, state)
        del state
        say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
        say(f"[ppo] out={out} device={args.device} net={trainer.cfg} params={sum(p.numel() for p in trainer.net.parameters())} "
            f"envs={env.N} ({env.W}x{env.E}, {env.G} groups, {env.start_method}) league={league} "
            f"start: update {trainer.update_idx}, steps {trainer.steps}")
        log = open(out / "log.jsonl", "a")

        def export(tag: str):
            folder = out / tag
            export_submission(trainer.net, folder, extra={"update": trainer.update_idx, "steps": trainer.steps})
            return folder

        if evaluator.enabled and trainer.update_idx == 0:
            evaluator.submit(export("ckpt_0"), {"update": 0, "steps": 0, "time": 0.0})
        t_session = time.time()
        while trainer.steps < args.total_steps:
            if args.time_limit > 0 and time.time() - t_session >= args.time_limit:
                say(f"[ppo] time limit of {args.time_limit:.0f}s reached at {trainer.steps} steps")
                break
            if args.stop_update > 0 and trainer.update_idx >= args.stop_update:
                say(f"[ppo] stop update {args.stop_update} reached at {trainer.steps} steps")
                break
            t0 = time.time()
            roll = trainer.collect()
            lg = trainer.league_stats()
            t1 = time.time()
            upd = trainer.update()
            if trainer.dev.type == "cuda":
                torch.cuda.synchronize(trainer.dev)
            t2 = time.time()
            trainer.elapsed += t2 - t0
            rec = {"update": trainer.update_idx, "steps": trainer.steps, "time": round(trainer.elapsed, 2),
                   "sps": round(env.N * args.rollout / (t2 - t0)), "sps_rollout": round(roll["sps_rollout"]),
                   "t_rollout": round(roll["t_rollout"], 3), "t_update": round(t2 - t1, 3),
                   "wait_frac": round(roll["wait_frac"], 3), "worker_busy": round(roll["worker_busy"], 3),
                   **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in upd.items()},
                   "episodes": lg["episodes"], "ep_len": round(lg["ep_len"], 2), "bot_faults": lg["bot_faults"],
                   "league": lg["league"]}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            u = trainer.update_idx
            if u % 5 == 0 or u == 1:
                wr = " ".join(f"{s.replace('zoo.', '')[:14]}:{v['win']:.2f}/{v['draw']:.2f}/{v['loss']:.2f}"
                              for s, v in lg["league"].items())
                say(f"[ppo] u{u} {trainer.steps / 1e6:.2f}M steps {trainer.elapsed:.0f}s sps {rec['sps']} (rollout "
                    f"{rec['sps_rollout']}) ent {upd['entropy']:.3f} v {upd['v_loss']:.4f} kl {upd['approx_kl']:.4f} "
                    f"ev {upd['explained_var']:.2f} len {lg['ep_len']:.0f} | W/D/L {wr}")
            if evaluator.enabled:
                if u % max(1, args.eval_every) == 0:
                    evaluator.submit(export(f"ckpt_{u}"), {"update": u, "steps": trainer.steps, "time": round(trainer.elapsed, 1)})
                r = evaluator.poll()
                if r:
                    say(_fmt_eval(r))
            if u % max(1, args.save_every) == 0:
                _save_state(trainer, out, resumes)
        _save_state(trainer, out, resumes)
        final = export("final")
        env.close()  # free the CPUs for the last evaluation
        if evaluator.enabled:
            evaluator.submit(final, {"update": trainer.update_idx, "steps": trainer.steps, "time": round(trainer.elapsed, 1)})
            for r in evaluator.finish():
                say(_fmt_eval(r))
            if evaluator.best:
                say(f"[ppo] best: {evaluator.best['ckpt']} mean win {evaluator.best['mean_win_rate']:.3f} -> {out / 'best'}")
        log.close()
    except KeyboardInterrupt:
        interrupted = True
        print("[ppo] interrupted: saving state.pt", flush=True)
        try:
            _save_state(trainer, out, resumes)
        except Exception as e:  # noqa: BLE001
            print(f"[ppo] could not save: {e!r}", flush=True)
    finally:
        env.close()
        evaluator.kill()
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
