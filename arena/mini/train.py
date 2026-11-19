"""Minimal, verified behaviour-cloning trainer for shards written by arena.mini.gen.

    nice -n 10 .venv/bin/python -m arena.mini.train --data runs/mini/data/xm0 --out runs/mini/bc_xm0 --device cuda:1 --epochs 10

Loss = cross-entropy to the teacher's action + vw * MSE(value, game result z). Random board symmetries (all 8).
Holdout = every 20th game (by map seed). Exports <out>/submission.{py,pth} (arena.nets.export_submission) and checks it.
"""
import argparse, glob, json, time
import numpy as np, torch, torch.nn.functional as F
from arena.nets import augment, build_net, check_submission, count_params, export_submission, load_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1"); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--cfg", default='{"channels": 96, "blocks": 8}'); ap.add_argument("--init", default=None)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--vw", type=float, default=0.5); ap.add_argument("--sym", type=int, default=8)
    args = ap.parse_args()
    dev = args.device
    xs, ys, zs, ms = [], [], [], []
    for d in args.data:
        for f in sorted(glob.glob(f"{d}/shard_*.npz")):
            s = np.load(f)
            xs.append(s["x"]); ys.append(s["y"]); ms.append(s["meta"][:, 0])
            zs.append(s["z"] if "z" in s else np.zeros(len(s["y"]), np.int8))
    x = torch.from_numpy(np.concatenate(xs)).to(dev); y = torch.from_numpy(np.concatenate(ys)).long().to(dev)
    z = torch.from_numpy(np.concatenate(zs)).float().to(dev)
    hold = torch.from_numpy(np.concatenate(ms) % 20 == 0).to(dev)
    tr = (~hold).nonzero().squeeze(1); ho = hold.nonzero().squeeze(1)
    print(f"train {len(tr)} holdout {len(ho)} labels {torch.bincount(y, minlength=6).tolist()} z mean {z.mean():.3f}", flush=True)
    torch.manual_seed(0)
    net = (load_submission(args.init) if args.init else build_net(json.loads(args.cfg))).to(dev).to(memory_format=torch.channels_last)
    print("params", count_params(net), net.cfg, flush=True)
    B = args.batch; steps = args.epochs * (len(tr) // B)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    def evaluate():
        net.eval(); c = 0; pa = torch.zeros(6); na = torch.zeros(6)
        with torch.no_grad():
            for i in range(0, len(ho), 8192):
                idx = ho[i:i + 8192]
                ok = (net(x[idx])[0].argmax(1) == y[idx])
                c += ok.sum().item()
                pa += torch.bincount(y[idx], weights=ok.float(), minlength=6).cpu(); na += torch.bincount(y[idx], minlength=6).cpu()
        net.train(); return c / len(ho), (pa / na.clamp(min=1)).tolist()

    t0 = time.time(); step = 0
    for ep in range(args.epochs):
        perm = tr[torch.randperm(len(tr), device=dev)]; tot = totv = 0.0; nb = 0
        for i in range(0, len(perm) - B + 1, B):
            idx = perm[i:i + B]
            xb, yb, _ = augment(x[idx], y[idx], n_sym=args.sym)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, v = net(xb)
            lp = F.cross_entropy(logits.float(), yb); lv = F.mse_loss(v.float(), z[idx])
            loss = lp + args.vw * lv
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += lp.item(); totv += lv.item(); nb += 1; step += 1
        acc, per = evaluate()
        print(f"epoch {ep + 1}/{args.epochs} ce {tot / nb:.4f} v {totv / nb:.4f} holdout acc {acc:.4f} per-action {[round(p, 3) for p in per]} "
              f"({time.time() - t0:.0f}s)", flush=True)
    export_submission(net, args.out, extra={"holdout_acc": acc, "data": args.data})
    check_submission(args.out)


if __name__ == "__main__":
    main()
