"""Compact status of PPO runs: python -m arena.mini.status [run names...]  (default: every folder in runs/rl)."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent / "runs" / "rl"
MAIN = ["zoo.base_cautious", "zoo.ta_smart_a", "zoo.ta_smart_b", "zoo.evader", "zoo.ta_final_a", "zoo.ta_final_b", "zoo.param", "self"]
runs = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if (p / "log.jsonl").exists())
for r in runs:
    d = ROOT / r
    try:
        last = json.loads((d / "log.jsonl").read_text().strip().splitlines()[-1])
    except Exception as e:
        print(r, "no log", e); continue
    age = time.time() - (d / "log.jsonl").stat().st_mtime
    lg = last["league"]
    tr = " ".join(f"{k.split('.')[-1][:9]}={lg[k]['win_ema']:.2f}" for k in MAIN if k in lg)
    bots = [v["win_ema"] for k, v in lg.items() if k not in ("self", "past")]
    print(f"{r}: u{last['update']} {last['steps'] / 1e6:.1f}M sps {last['sps'] / 1e3:.0f}k ent {last['entropy']:.2f} len {last['ep_len']:.0f} "
          f"log-age {age:.0f}s | train(sampled) mean {sum(bots) / len(bots):.2f} | {tr}")
    ev = d / "evals.jsonl"
    if ev.exists():
        E = [json.loads(l) for l in ev.read_text().strip().splitlines()]
        best = max(E, key=lambda e: e.get("mean_win_rate", 0))
        for tag, e in (("last", E[-1]), ("best", best)):
            per = e.get("per_opp", {})
            s = " ".join(f"{k.split('.')[-1][:10]} {v['win']}-{v['draw']}-{v['loss']}" for k, v in list(per.items())[:9])
            print(f"   eval {tag} u{e.get('update')} mean_win {e.get('mean_win_rate', 0):.3f} loss {e.get('mean_loss_rate', 0):.3f} | {s}")
