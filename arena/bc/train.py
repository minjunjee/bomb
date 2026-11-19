"""arena.bc.train - supervised (behaviour-cloning) training on one GPU, export and harness check.

    nice -n 10 .venv/bin/python -m arena.bc.train --data <dir> [<dir> ...] --cfg '{"channels":96,"blocks":8}' \
        --out runs/bc/<run> --device cuda:0 --epochs E [--init <submission folder>] \
        [--eval-opp simple,zoo.base_cautious --eval-games 200]

Data
    --data DIR [DIR ...]   datasets written by arena.bc.gen (all of them are loaded into GPU memory as uint8 codes,
                           1053 bytes per sample, and decoded to the exact float32 planes on the GPU)
    --holdout F            share of the GAMES held out (default 0.05). The split is by MAP SEED with a fixed table
                           (--split-seed), so a game is on the same side in every run: a warm-started student of a
                           later DAgger iteration has never trained on any holdout game.
Network
    --cfg JSON             arena.nets.build_net(cfg) (default: the cfg of --init, else arena.nets.DEFAULT_CFG)
    --init FOLDER          warm start from an exported submission folder (same architecture)
Loss = cross-entropy(scores, y)
    --q-weight W --q-temp T        + W * cross-entropy to softmax(q / T)   (only samples that have teacher values)
    --value-weight W --value-gamma G   + W * MSE(value head, outcome * G^togo)      (default W = 0: off)
    --label-smoothing S
Optimisation
    --epochs E --batch B (2048) --lr LR (2e-3) --wd WD (0.01) --warmup (0.03 of the steps, or a step count)
    --lr-final F (0.02)    AdamW, linear warm-up then cosine down to F * LR
    --sym {8,2,1}          random board symmetries on the GPU (arena.nets.augment): all 8 | identity + transpose | off
    --amp                  bf16 autocast (the holdout is always scored in float32, as the server runs the network)
    --seed N
Output (all in --out)
    log.jsonl              {"type": "train"}: step, loss parts, lr, samples/s | {"type": "epoch"}: holdout top-1
                           agreement with the teacher, per-action agreement, confusion (rows = teacher action,
                           columns = network action), per-dataset and per-player agreement, the same on a train subset
    best/ final/           arena.nets.export_submission (best = highest holdout agreement; --best-by ce = lowest
                           holdout cross-entropy, the steadier choice for a teacher with random tie-breaks), checked
                           with arena.nets.check_submission (--no-check to skip)
    --export-sym K         export with test-time symmetry averaging (cfg "sym" = K) on the same weights
    --eval-opp A,B,...     run `arena.core leaderboard` in a subprocess (--eval-games 200, --eval-procs 32,
                           --eval-seed0 1000000, --eval-which best|final|both) -> eval_best.json / eval_final.json
    result.json            the summary (written last: its existence means the run finished)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import ACTION_NAMES, EVAL_SEED0, ROOT, iter_shards, read_manifest, torch_lut, write_json_atomic

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.nets import DEFAULT_CFG, augment, build_net, check_submission, count_params, export_submission  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
def holdout_table(frac: float, split_seed: int = 0) -> np.ndarray:
    """is_holdout[map_seed] for every training map seed: fixed, so the split never changes between runs."""
    return np.random.default_rng(split_seed).random(EVAL_SEED0) < frac


def load_data(dirs, device, holdout: float, split_seed: int) -> dict:
    """All datasets in GPU memory. Returns a dict of tensors (X stays uint8 codes) + the train / holdout indices."""
    mans = [read_manifest(d) for d in dirs]
    n = sum(m["samples"] for m in mans)
    if n == 0:
        raise SystemExit("no samples in --data")
    any_q = any(m["has_q"] for m in mans)
    D = {
        "X": torch.empty((n, 13, 9, 9), dtype=torch.uint8, device=device),
        "Y": torch.empty(n, dtype=torch.long, device=device),
        "OUT": torch.empty(n, dtype=torch.float32, device=device),
        "TOGO": torch.empty(n, dtype=torch.float32, device=device),
        "PLAYER": torch.empty(n, dtype=torch.uint8, device=device),
        "DS": torch.empty(n, dtype=torch.long, device=device),
        "HOLD": torch.empty(n, dtype=torch.bool, device=device),
        "Q": torch.zeros((n, 6), dtype=torch.float32, device=device) if any_q else None,
        "HASQ": torch.zeros(n, dtype=torch.bool, device=device) if any_q else None,
    }
    table = holdout_table(holdout, split_seed)
    o = games = hold_games = 0
    for di, (d, m) in enumerate(zip(dirs, mans)):
        for sh in iter_shards(d):
            k = len(sh["y"])
            seeds = m["seed0"] + sh["game"].astype(np.int64)
            if k and (seeds.min() < 0 or seeds.max() >= EVAL_SEED0):
                raise SystemExit(f"{d}: map seeds outside [0, {EVAL_SEED0}) - evaluation maps must not be trained on")
            games += len(sh["g_seed"])
            hold_games += int(table[sh["g_seed"]].sum())
            if not k:
                continue
            s = slice(o, o + k)
            D["X"][s] = torch.from_numpy(sh["x"]).to(device)
            D["Y"][s] = torch.from_numpy(sh["y"].astype(np.int64)).to(device)
            D["OUT"][s] = torch.from_numpy(sh["outcome"].astype(np.float32)).to(device)
            D["TOGO"][s] = torch.from_numpy(sh["togo"].astype(np.float32)).to(device)
            D["PLAYER"][s] = torch.from_numpy(sh["player"]).to(device)
            D["DS"][s] = di
            D["HOLD"][s] = torch.from_numpy(table[seeds]).to(device)
            if "q" in sh:
                q = sh["q"]
                ok = ~(np.isnan(q).any(1) | np.isposinf(q).any(1) | np.isneginf(q).all(1))
                D["Q"][s] = torch.from_numpy(np.where(ok[:, None], q, 0.0).astype(np.float32)).to(device)
                D["HASQ"][s] = torch.from_numpy(ok).to(device)
            o += k
    assert o == n, (o, n)
    D["train_idx"] = torch.nonzero(~D["HOLD"]).squeeze(1)
    D["hold_idx"] = torch.nonzero(D["HOLD"]).squeeze(1)
    D["info"] = {"dirs": [str(d) for d in dirs], "samples": n, "games": games, "holdout_games": hold_games,
                 "train_samples": int(D["train_idx"].numel()), "holdout_samples": int(D["hold_idx"].numel()),
                 "samples_per_dataset": [m["samples"] for m in mans], "teachers": sorted({m["teacher"] for m in mans}),
                 "has_q": [bool(m["has_q"]) for m in mans]}
    return D


# ----------------------------------------------------------------------------------------------------------------
# evaluation on stored samples (always float32, no augmentation)
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, D, idx, lut, n_ds: int, batch: int = 16384) -> dict:
    dev = idx.device
    n = int(idx.numel())
    if n == 0:
        return {"n": 0, "top1": float("nan")}
    conf = torch.zeros(36, dtype=torch.long, device=dev)
    ds_hit = torch.zeros(n_ds, dtype=torch.float64, device=dev)
    ds_cnt = torch.zeros(n_ds, dtype=torch.float64, device=dev)
    pl_hit = torch.zeros(2, dtype=torch.float64, device=dev)
    pl_cnt = torch.zeros(2, dtype=torch.float64, device=dev)
    ce = torch.zeros((), dtype=torch.float64, device=dev)
    vse = torch.zeros((), dtype=torch.float64, device=dev)
    for ii in idx.split(batch):
        x = lut[D["X"][ii].long()].contiguous(memory_format=torch.channels_last)
        logits, value = net(x)
        y = D["Y"][ii]
        pred = logits.argmax(1)
        hit = (pred == y).double()
        conf += torch.bincount(y * 6 + pred, minlength=36)
        ds_hit.index_add_(0, D["DS"][ii], hit)
        ds_cnt.index_add_(0, D["DS"][ii], torch.ones_like(hit))
        pl_hit.index_add_(0, D["PLAYER"][ii].long(), hit)
        pl_cnt.index_add_(0, D["PLAYER"][ii].long(), torch.ones_like(hit))
        ce += F.cross_entropy(logits.float(), y, reduction="sum").double()
        vse += ((value.float() - D["OUT"][ii]) ** 2).sum().double()
    conf = conf.view(6, 6).cpu().numpy()
    diag = np.diag(conf)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = diag / conf.sum(1)
        precision = diag / conf.sum(0)
        per_ds = (ds_hit / ds_cnt).cpu().numpy()
        per_pl = (pl_hit / pl_cnt).cpu().numpy()
    clean = lambda v: None if not np.isfinite(v) else round(float(v), 5)  # noqa: E731
    return {
        "n": n, "top1": float(diag.sum() / n), "ce": float(ce / n), "value_mse": float(vse / n),
        "per_action": {a: clean(recall[i]) for i, a in enumerate(ACTION_NAMES)},     # agreement given the teacher's action
        "precision": {a: clean(precision[i]) for i, a in enumerate(ACTION_NAMES)},   # ... given the network's action
        "confusion": conf.tolist(),                                                   # rows teacher, columns network
        "per_dataset": [clean(v) for v in per_ds], "per_player": [clean(v) for v in per_pl],
    }


def harness_eval(folder: Path, opps: str, games: int, procs: int, seed0: int, json_path: Path) -> dict | None:
    """`arena.core leaderboard` for an exported folder in a subprocess (CPU workers, plays exactly like the server)."""
    cmd = [sys.executable, "-m", "arena.core", "leaderboard", "--me", f"net:{Path(folder).resolve()}", "--opp", opps,
           "--games", str(games), "--seed0", str(seed0), "--procs", str(procs), "--json", str(Path(json_path).resolve())]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.rstrip()[-4000:], flush=True)
    if r.returncode != 0:
        print(f"[train] harness evaluation failed:\n{r.stderr[-3000:]}", flush=True)
        return None
    return json.loads(Path(json_path).read_text())


def short_eval(res: dict | None) -> dict | None:
    if not res:
        return None
    return {opp: {k: t[k] for k in ("win", "draw", "loss", "self_bomb", "timeout_draw", "faults", "win_as_A", "win_as_B")}
            | {"avg_steps": round(t["avg_steps"], 1), "ms_per_act": round(t["ms_per_act"], 2)}
            for opp, t in res["per_opp"].items()}


# ----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--cfg", default=None, help="json for arena.nets.build_net")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--init", default=None)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lr-final", type=float, default=0.02)
    ap.add_argument("--sym", type=int, default=8, choices=(1, 2, 8))
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--q-weight", type=float, default=0.0)
    ap.add_argument("--q-temp", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=0.0)
    ap.add_argument("--value-gamma", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--best-by", default="top1", choices=("top1", "ce"))
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--export-sym", type=int, default=None, choices=(1, 2, 8))
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--eval-opp", default=None)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-procs", type=int, default=32)
    ap.add_argument("--eval-seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--eval-which", default="best", choices=("best", "final", "both"))
    args = ap.parse_args(argv)

    t_all = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").unlink(missing_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- network
    cfg = json.loads(args.cfg) if args.cfg else None
    if args.init:
        init = Path(args.init)
        init_cfg = json.loads((init / "meta.json").read_text())["cfg"]
        cfg = {**init_cfg, **cfg} if cfg else init_cfg
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    net = build_net(cfg)
    if args.init:
        net.load_state_dict(torch.load(Path(args.init) / "submission.pth", map_location="cpu"))
    net = net.to(dev).to(memory_format=torch.channels_last)
    export_cfg = {"sym": args.export_sym} if args.export_sym else None

    # ---- data
    t0 = time.time()
    D = load_data(args.data, dev, args.holdout, args.split_seed)
    info = D["info"]
    lut = torch_lut(dev)
    train_idx, hold_idx = D["train_idx"], D["hold_idx"]
    n_train, n_ds = int(train_idx.numel()), len(args.data)
    g = torch.Generator(device=dev).manual_seed(args.seed + 1)
    sub = train_idx[torch.randperm(n_train, device=dev, generator=g)[: max(20_000, int(hold_idx.numel()))]]
    use_q = args.q_weight > 0 and D["Q"] is not None and bool(D["HASQ"].any())
    if args.q_weight > 0 and not use_q:
        print("[train] WARNING: --q-weight given but no dataset has teacher action values: soft targets are off")
    print(f"[train] {info['samples']} samples / {info['games']} games from {n_ds} dataset(s) on {dev} in "
          f"{time.time() - t0:.1f}s | train {n_train} | holdout {info['holdout_samples']} samples of "
          f"{info['holdout_games']} games | net {cfg['channels']}x{cfg['blocks']} ({count_params(net) / 1e6:.2f}M params)"
          f"{' | init ' + args.init if args.init else ''}", flush=True)

    # ---- optimiser and schedule
    batch = min(args.batch, n_train)
    steps_per_epoch = max(1, n_train // batch)
    total = steps_per_epoch * args.epochs
    warm = int(args.warmup * total) if args.warmup < 1 else int(args.warmup)
    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, fused=dev.type == "cuda")

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return args.lr * (args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * t)))

    log = open(out / "log.jsonl", "w")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    write_json_atomic(out / "args.json", {"args": vars(args), "cfg": cfg, "data": info})
    emit({"type": "start", "args": vars(args), "cfg": cfg, "data": info, "steps": total})

    best = {"top1": -1.0, "rank": -float("inf"), "epoch": 0}
    acc = torch.zeros(4, device=dev)  # sums of loss, ce, soft, value loss since the last log line
    step = seen = acc_n = 0
    last_sps = 0.0
    speeds, epochs_log = [], []
    t_train = 0.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = train_idx[torch.randperm(n_train, device=dev)]
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        t_seg, seg_samples, t_epoch = time.perf_counter(), 0, time.perf_counter()
        for i in range(steps_per_epoch):
            ii = perm[i * batch:(i + 1) * batch]
            xb, yb = D["X"][ii], D["Y"][ii]
            qb = D["Q"][ii] if use_q else None
            if args.sym > 1:
                xb, yb, qb = augment(xb, yb, None, qb, n_sym=args.sym)  # on the uint8 codes: exact
            x = lut[xb.long()].contiguous(memory_format=torch.channels_last)
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.amp):
                logits, value = net(x)
            logits, value = logits.float(), value.float()
            ce = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            loss, soft, vloss = ce, acc.new_zeros(()), acc.new_zeros(())
            if use_q:
                has = D["HASQ"][ii].float()
                per = -(F.softmax(qb / args.q_temp, 1) * F.log_softmax(logits, 1)).sum(1)
                soft = (per * has).sum() / has.sum().clamp(min=1)
                loss = loss + args.q_weight * soft
            if args.value_weight > 0:
                target = D["OUT"][ii] * (args.value_gamma ** D["TOGO"][ii]) if args.value_gamma < 1 else D["OUT"][ii]
                vloss = F.mse_loss(value, target)
                loss = loss + args.value_weight * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc += torch.stack([loss.detach(), ce.detach(), soft.detach(), vloss.detach()])
            acc_n += 1
            step += 1
            seen += int(ii.numel())
            seg_samples += int(ii.numel())
            if step % args.log_every == 0 or i == steps_per_epoch - 1:
                vals = (acc / acc_n).tolist()  # .tolist() waits for the GPU: the timing below is real
                acc.zero_()
                acc_n = 0
                dt = time.perf_counter() - t_seg
                sps = seg_samples / max(1e-9, dt)
                last_sps = sps
                if step > args.log_every:  # the first interval includes the cuDNN warm-up
                    speeds.append(sps)
                emit({"type": "train", "step": step, "epoch": epoch, "loss": vals[0], "ce": vals[1], "soft": vals[2],
                      "value_loss": vals[3], "lr": lr_at(step - 1), "samples_per_s": sps})
                t_seg, seg_samples = time.perf_counter(), 0
        t_train += time.perf_counter() - t_epoch
        net.eval()
        hold = evaluate(net, D, hold_idx, lut, n_ds)
        tr = evaluate(net, D, sub, lut, n_ds)
        score = hold["top1"] if hold["n"] else tr["top1"]
        rank = -(hold["ce"] if hold["n"] else tr["ce"]) if args.best_by == "ce" else score
        is_best = rank > best["rank"]
        if is_best:
            best = {"top1": score, "rank": rank, "epoch": epoch, "holdout": hold}
            export_submission(net, out / "best", cfg=export_cfg, extra={"epoch": epoch, "holdout_top1": score})
        rec = {"type": "epoch", "epoch": epoch, "step": step, "holdout": hold, "train_subset": tr,
               "epoch_seconds": time.perf_counter() - t_epoch, "best": is_best}
        emit(rec)
        epochs_log.append({"epoch": epoch, "holdout_top1": hold["top1"], "train_top1": tr["top1"], "holdout_ce": hold.get("ce")})
        print(f"[train] epoch {epoch:>3}/{args.epochs} | holdout top-1 {hold['top1']:.4f} ce {hold.get('ce', float('nan')):.4f} | "
              f"train-subset top-1 {tr['top1']:.4f} | {last_sps:.0f} samples/s"
              f"{' *' if is_best else ''}", flush=True)

    # ---- export, check, harness
    final_hold = evaluate(net.eval(), D, hold_idx, lut, n_ds)
    export_submission(net, out / "final", cfg=export_cfg, extra={"epoch": args.epochs, "holdout_top1": final_hold["top1"]})
    result = {
        "out": str(out), "cfg": cfg, "data": info, "args": vars(args), "params": count_params(net),
        "epochs": epochs_log, "best_epoch": best["epoch"], "holdout_top1_best": best["top1"],
        "holdout_top1_final": final_hold["top1"], "holdout_best": best.get("holdout"), "holdout_final": final_hold,
        "train_samples_per_s": float(np.median(speeds)) if speeds else seen / max(1e-9, t_train),  # median of the log intervals
        "train_samples_per_s_mean": seen / max(1e-9, t_train), "train_seconds": t_train, "steps": step,
    }
    if not args.no_check:
        for which in ("best", "final"):
            result[f"check_{which}"] = check_submission(out / which, verbose=False)
            print(f"[train] check_submission {which}: {result[f'check_{which}']}", flush=True)
    if args.eval_opp:
        for which in (("best", "final") if args.eval_which == "both" else (args.eval_which,)):
            res = harness_eval(out / which, args.eval_opp, args.eval_games, args.eval_procs, args.eval_seed0,
                               out / f"eval_{which}.json")
            result[f"eval_{which}"] = short_eval(res)
            result[f"eval_{which}_mean_win_rate"] = res["mean_win_rate"] if res else None
    result["seconds"] = time.time() - t_all
    emit({"type": "final", **{k: v for k, v in result.items() if k not in ("args", "data")}})
    log.close()
    write_json_atomic(out / "result.json", result)
    print(f"[train] done in {result['seconds']:.0f}s: holdout top-1 best {best['top1']:.4f} (epoch {best['epoch']}), "
          f"final {final_hold['top1']:.4f} | {result['train_samples_per_s'] or 0:.0f} samples/s | {out}", flush=True)
    return result


if __name__ == "__main__":
    main()
