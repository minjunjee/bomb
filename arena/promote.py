"""Keep-or-discard step of the research loop: grade a candidate network on the validation panel and, if it beats
the incumbent in best/, make it the new best (submission.py + submission.pth + submission.zip).

    nice -n 10 .venv/bin/python -m arena.promote runs/exp/r1/a/best --name r1-a --desc "what was tried"
    ... --dry-run     grade and log only, never replace best/
    ... --games 200   games per validation opponent (default 200)

Every call appends one line to runs/results.jsonl (the experiment log). The decision metric is the validation
score = mean wins per 100 games over the VAL panel (bots never trained against); ties within 1 point go to the
candidate with fewer losses. The TEST panel is not touched here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from arena.core import DEFAULT_PROCS, EVAL_SEED0, ROOT
from arena.grade import grade
from arena.nets import check_submission, make_zip
from arena.panel import panel

BEST = ROOT / "best"
LOG = ROOT / "runs" / "results.jsonl"


def summary(g: dict) -> dict:
    return {"score": round(g["score"], 2), "worst": round(g["worst_score"], 1), "worst_opp": g["worst_opp"],
            "loss": round(100 * g["mean_loss_rate"], 2), "draw": round(100 * g["mean_draw_rate"], 2),
            "self_bomb": round(100 * g["self_bomb_rate"], 2), "timeout": round(100 * g["timeout_rate"], 2),
            "per_opp": {o: [t["win"], t["draw"], t["loss"]] for o, t in g["per_opp"].items()}}


def better(new: dict, old: dict | None) -> bool:
    if old is None:
        return True
    if abs(new["score"] - old["score"]) <= 1.0:
        return (new["loss"], -new["score"]) < (old["loss"], -old["score"])
    return new["score"] > old["score"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="candidate folder with submission.py + submission.pth")
    ap.add_argument("--name", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    ap.add_argument("--panel", default="val")
    ap.add_argument("--also", default="train", help="extra panel graded for information only ('' = none)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    folder = folder if folder.is_absolute() else ROOT / folder
    spec = f"net:{folder}"
    chk = check_submission(folder, verbose=False)
    if not chk.get("ok"):
        raise SystemExit(f"check_submission failed: {chk}")
    print(f"check ok: {chk['ms_per_act']} ms/act, {chk['pth_bytes'] / 1e6:.1f} MB")
    val = grade(spec, panel(args.panel), args.games, args.seed0, args.procs)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": args.name, "desc": args.desc, "folder": str(folder),
           "games": args.games, "seed0": args.seed0, "ms_per_act": chk["ms_per_act"], "val": summary(val)}
    if args.also:
        extra = grade(spec, panel(args.also), max(50, args.games // 2), args.seed0, args.procs)
        rec[args.also] = summary(extra)
    (folder / "grade.json").write_text(json.dumps(rec, indent=1))

    old = None
    if (BEST / "grade.json").exists():
        old = json.loads((BEST / "grade.json").read_text())
        # the panel may have grown since the incumbent was graded: compare on the same opponents and seeds
        if old.get("folder") and (set(old["val"]["per_opp"]) != set(rec["val"]["per_opp"]) or old.get("games") != args.games
                                  or old.get("seed0") != args.seed0):
            if (BEST / "submission.pth").exists():
                print("re-grading the incumbent on the current panel ...")
                old_val = grade(f"net:{BEST}", panel(args.panel), args.games, args.seed0, args.procs, verbose=False)
                old = {**old, "val": summary(old_val), "games": args.games, "seed0": args.seed0}
                (BEST / "grade.json").write_text(json.dumps(old, indent=1))
    keep = better(rec["val"], old["val"] if old else None)
    rec["status"] = "dry-run" if args.dry_run else ("keep" if keep else "discard")
    rec["incumbent"] = None if old is None else {"name": old.get("name"), "score": old["val"]["score"], "loss": old["val"]["loss"]}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n{args.name}: val score {rec['val']['score']} (worst {rec['val']['worst']} vs {rec['val']['worst_opp']}, "
          f"losses {rec['val']['loss']}) | incumbent: {rec['incumbent']} -> {rec['status'].upper()}")
    if keep and not args.dry_run:
        BEST.mkdir(exist_ok=True)
        for f in ("submission.py", "submission.pth"):
            shutil.copy(folder / f, BEST / f)
        make_zip(BEST)
        (BEST / "grade.json").write_text(json.dumps(rec, indent=1))
        (BEST / "NOTES.txt").write_text(f"{rec['time']}  {args.name}: {args.desc}\nfrom {folder}\nval score {rec['val']['score']} "
                                        f"wins/100, losses {rec['val']['loss']}/100, {chk['ms_per_act']} ms/act\n")
        print(f"new best -> {BEST / 'submission.zip'}")


if __name__ == "__main__":
    main()
