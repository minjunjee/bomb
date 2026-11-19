"""arena.bc.dagger - the DAgger loop: one resumable CLI, everything under runs/bc/<name>/.

    nice -n 10 .venv/bin/python -m arena.bc.dagger --name <name> --teacher zoo.T --iters K \
        [--pairs0 "T|T*3, T|noisy:0.25:T*3, T|simple*2, T|random*2"] [--games0 3000] [--epochs0 30] \
        [--pairs "+S|T*3, +noisy:0.15:S|T*2, +S@sample|T, +S|+noisy:0.15:S, +S|simple, +S|random"] [--games 1500] \
        [--epochs 15] [--cfg '{"channels":96,"blocks":8}'] [--eval-opp random,simple,zoo.base_cautious] \
        [--eval-games 200] [--procs 32] [--device cuda:0] [--train-args "--amp --value-weight 0.05"]

Iteration 0    teacher-driven games (--pairs0: teacher vs zoo bots, teacher vs noisy teacher, ...) -> data/it00,
               a network trained from scratch -> it00/
Iteration i>0  the current student S = it<i-1>/<--student-export> plays --pairs (argmax "S", sampled "S@sample",
               eps-noisy "noisy:<eps>:S") against a mix of opponents (zoo bots, the teacher T, itself); the states of
               the "+" actors are labelled by the teacher -> data/it<i>; then a network is trained on ALL data so far,
               warm-started from the previous student (--fresh: from scratch) -> it<i>/
Every iteration is evaluated with the harness (train.py --eval-opp); the teacher itself is evaluated once
(teacher_eval.json) for comparison.

Arguments
    --name NAME            the run folder runs/bc/NAME (everything is written there)
    --teacher SPEC         the labelling bot
    --iters K              DAgger iterations after iteration 0 (raise it and run again to continue)
    --pairs0 / --pairs     the pair mix of iteration 0 / of the later iterations (syntax: arena.bc.common)
    --games0 / --games     games of iteration 0 / per later iteration; map seeds: --seed0 .. (never reused)
    --cfg JSON             the network (arena.nets.build_net)
    --epochs0 / --epochs   epochs of iteration 0 / of the later iterations;  --lr0 / --lr the same for the peak lr
    --batch B, --sym K     passed to train.py;  --train-args / --gen-args "...": any other train.py / gen.py options
    --eval-opp A,B --eval-games N --eval-procs P    the harness evaluation of every student (and of the teacher)
    --student-export best|final    which export of an iteration is "the student" (default best)
    --procs P              generation processes;  --device cuda:0

Files
    data/itNN/             the dataset of iteration NN (manifest.json + shards)
    itNN/                  train.py output: log.jsonl, best/, final/, eval_*.json, result.json
    teacher_eval.json      the teacher's own harness results
    summary.json           one row per iteration: samples, how often the student played the teacher's action in its
                           own games, holdout agreement, harness results
    commands.log           every gen / train / eval command that was run
Resuming: a stage is skipped when its output exists (data/itNN/manifest.json, itNN/result.json); the same command
continues where it stopped. The arguments that define the data are frozen in dagger.json (--allow-change to override).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .common import EVAL_SEED0, ROOT, parse_pairs, read_manifest, write_json_atomic

FROZEN = ("teacher", "pairs0", "pairs", "games0", "games", "seed0", "cfg")


def run(cmd: list[str], log: Path):
    line = shlex.join(cmd)
    print(f"[dagger] $ {line}", flush=True)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} $ {line}\n")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[dagger] the command failed (exit {r.returncode}); fix it and run the same dagger command again")


def short_eval(res: dict) -> dict:
    return {opp: {k: t[k] for k in ("win", "draw", "loss", "self_bomb", "timeout_draw", "faults")}
            for opp, t in res["per_opp"].items()}


def wdl(d: dict | None) -> str:
    return "-" if not d else f"{d['win']}/{d['draw']}/{d['loss']}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--pairs0", default="T|T*3, T|noisy:0.25:T*3, T|simple*2, T|random*2")
    ap.add_argument("--pairs", default="+S|T*3, +noisy:0.15:S|T*2, +S@sample|T, +S|+noisy:0.15:S, +S|simple, +S|random")
    ap.add_argument("--games0", type=int, default=3000)
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--cfg", default='{"channels":96,"blocks":8}')
    ap.add_argument("--epochs0", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr0", type=float, default=2e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--sym", type=int, default=8, choices=(1, 2, 8))
    ap.add_argument("--fresh", action="store_true", help="train every iteration from scratch (no warm start)")
    ap.add_argument("--student-export", default="best", choices=("best", "final"))
    ap.add_argument("--eval-opp", default="random,simple,zoo.base_cautious")
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-procs", type=int, default=32)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-args", default="")
    ap.add_argument("--gen-args", default="")
    ap.add_argument("--allow-change", action="store_true")
    args = ap.parse_args(argv)

    last_seed = args.seed0 + args.games0 + args.iters * args.games
    if args.seed0 < 0 or last_seed > EVAL_SEED0:
        raise SystemExit(f"map seeds {args.seed0}..{last_seed - 1} must stay below {EVAL_SEED0}")
    rel = Path("runs") / "bc" / args.name  # relative to the project root: also the form of the net:<folder> specs
    root = ROOT / rel
    root.mkdir(parents=True, exist_ok=True)
    parse_pairs(args.pairs0, args.teacher, "x")  # syntax errors now, not after an hour
    parse_pairs(args.pairs, args.teacher, "x")
    frozen = {k: getattr(args, k) for k in FROZEN}
    cfg_file = root / "dagger.json"
    if cfg_file.exists():
        old = json.loads(cfg_file.read_text())["frozen"]
        diff = {k: (old.get(k), v) for k, v in frozen.items() if old.get(k) != v}
        if diff and not args.allow_change:
            raise SystemExit(f"[dagger] {rel} was started with other arguments {diff}; use another --name or --allow-change")
    write_json_atomic(cfg_file, {"frozen": frozen, "args": vars(args)})
    log, py = root / "commands.log", sys.executable

    # ---- the teacher's own results against the evaluation opponents
    teacher_eval = None
    if args.eval_opp:
        tfile = root / "teacher_eval.json"
        if tfile.exists():
            t = json.loads(tfile.read_text())
            if t.get("games") != args.eval_games or list(t["per_opp"]) != args.eval_opp.split(","):
                tfile.unlink()
        if not tfile.exists():
            run([py, "-m", "arena.core", "leaderboard", "--me", args.teacher, "--opp", args.eval_opp, "--games",
                 str(args.eval_games), "--procs", str(args.eval_procs), "--json", str(tfile)], log)
        teacher_eval = short_eval(json.loads(tfile.read_text()))

    rows = []
    for it in range(args.iters + 1):
        tag = f"it{it:02d}"
        data_dir, train_dir = rel / "data" / tag, rel / tag
        student = None if it == 0 else str(rel / f"it{it - 1:02d}" / args.student_export)
        # ---- 1. data: iteration 0 = teacher-driven games, later = the student's games labelled by the teacher
        if not (ROOT / data_dir / "manifest.json").exists():
            cmd = [py, "-m", "arena.bc.gen", "--teacher", args.teacher,
                   "--pairs", args.pairs0 if it == 0 else args.pairs,
                   "--games", str(args.games0 if it == 0 else args.games),
                   "--seed0", str(args.seed0 if it == 0 else args.seed0 + args.games0 + (it - 1) * args.games),
                   "--procs", str(args.procs), "--out", str(data_dir)]
            if student:
                cmd += ["--student", student]
            run(cmd + shlex.split(args.gen_args), log)
        else:
            print(f"[dagger] {tag}: data exists, skipping the generation", flush=True)
        man = read_manifest(ROOT / data_dir)
        # ---- 2. training on everything so far (+ export, check, harness evaluation)
        if not (ROOT / train_dir / "result.json").exists():
            cmd = [py, "-m", "arena.bc.train", "--data", *[str(rel / "data" / f"it{k:02d}") for k in range(it + 1)],
                   "--cfg", args.cfg, "--out", str(train_dir), "--device", args.device,
                   "--epochs", str(args.epochs0 if it == 0 else args.epochs),
                   "--lr", str(args.lr0 if it == 0 or args.fresh else args.lr),
                   "--batch", str(args.batch), "--sym", str(args.sym)]
            if student and not args.fresh:
                cmd += ["--init", student]
            if args.eval_opp:
                cmd += ["--eval-opp", args.eval_opp, "--eval-games", str(args.eval_games), "--eval-procs",
                        str(args.eval_procs), "--eval-which", args.student_export]
            run(cmd + shlex.split(args.train_args), log)
        else:
            print(f"[dagger] {tag}: training exists, skipping it", flush=True)
        res = json.loads((ROOT / train_dir / "result.json").read_text())
        # ---- 3. the summary row
        per_pair = man["counts"]["per_pair"]
        own = [d for k, d in per_pair.items() if student and k.split("|")[0] == f"+net:{student}" and not k.split("|")[1].startswith("+")]
        own_n = sum(d["samples"] for d in own)
        rows.append({
            "iter": it, "student": str(train_dir / args.student_export), "games": man["games"],
            "new_samples": man["samples"], "total_samples": res["data"]["samples"],
            "teacher_errors": man["counts"]["teacher_errors"], "actor_faults": man["counts"]["actor_faults"],
            "gen_samples_per_s_per_proc": man["timing"]["samples_per_s_per_proc"],
            # in the games of this iteration: how often the actors played the teacher's action
            "label_eq_played": man["counts"]["label_eq_played"],
            "label_eq_played_per_pair": {k: round(d["label_eq_played"], 4) for k, d in per_pair.items()},
            # ... the argmax student of the previous iteration alone (pairs "+S|<opponent>")
            "prev_student_on_policy_agreement": sum(d["label_eq_played"] * d["samples"] for d in own) / own_n if own_n else None,
            "holdout_top1_best": res["holdout_top1_best"], "holdout_top1_final": res["holdout_top1_final"],
            "holdout_per_dataset": (res.get("holdout_best") or {}).get("per_dataset"),
            "holdout_samples": res["data"]["holdout_samples"], "best_epoch": res["best_epoch"],
            "train_samples_per_s": res["train_samples_per_s"], "train_seconds": res["train_seconds"],
            "check": res.get(f"check_{args.student_export}"),
            "eval": res.get(f"eval_{args.student_export}"),
            "eval_mean_win_rate": res.get(f"eval_{args.student_export}_mean_win_rate"),
        })
        write_json_atomic(root / "summary.json", {"teacher": args.teacher, "teacher_eval": teacher_eval, "iterations": rows,
                                                  "args": vars(args)})

    # ---- the table
    opps = args.eval_opp.split(",") if args.eval_opp else []
    print(f"\n[dagger] {rel}: teacher {args.teacher}, W/D/L in {args.eval_games} games per opponent")
    head = f"{'':<9}{'games':>6}{'samples':>9}{'total':>9}{'on-policy':>10}{'holdout':>9}  " + "  ".join(f"{o[:20]:>20}" for o in opps)
    print(head)
    if teacher_eval:
        print(f"{'teacher':<9}{'':>6}{'':>9}{'':>9}{'':>10}{'':>9}  " + "  ".join(f"{wdl(teacher_eval.get(o)):>20}" for o in opps))
    for r in rows:
        on = r["prev_student_on_policy_agreement"]
        ev = r["eval"] or {}
        print(f"{'it%02d' % r['iter']:<9}{r['games']:>6}{r['new_samples']:>9}{r['total_samples']:>9}"
              f"{(f'{on:.4f}' if on is not None else '-'):>10}{r['holdout_top1_best']:>9.4f}  "
              + "  ".join(f"{wdl(ev.get(o)):>20}" for o in opps))
    print("on-policy = how often the previous student (argmax, pairs '+S|opponent') played the teacher's action in the "
          "games of this iteration; holdout = top-1 agreement on held-out games of all data so far")
    print(f"[dagger] summary: {rel / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
