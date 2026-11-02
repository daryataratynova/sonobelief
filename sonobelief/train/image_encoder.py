"""Train the plane autoencoder: M_P -> E_img -> e_P -> D_img -> M_hat_P.

Loss, weighted per channel (config has lambda_liver):
  L_AE = lambda_liver * (BCE_liver + lambda_dice * Dice_liver) + lambda_fov * BCE_fov
  Dice_liver is the soft Dice loss per plane (eps = 1, so empty-liver planes are pushed to
  predict no liver), averaged over planes.
Legacy loss (config without lambda_liver, e.g. ae_liver.json):
  L_img = BCE over both channels + lambda_dice * soft Dice over `dice_channels`
Metrics (val, hard threshold 0.5): liver Dice over planes that contain liver, overall and per
window (general, subxiphoid, off-target); FOV Dice; on planes with no liver, the mean number
of false-positive liver pixels.
Training planes: window types in cfg["windows"] (0 general, 1 subxiphoid, 2 off-target).

Run dir runs/<name>_<timestamp>/: config.json, log.csv, best.pt (by val liver Dice),
last.pt, recon_val.png, and with --export embeddings_<split>.npz (e_P for every plane,
with study index and pose) for downstream use.

Usage: python -m sonobelief.train.image_encoder [--config configs/image_encoder.json] [--epochs N] [--export]
"""
import argparse
import csv
import json
import math
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sonobelief import paths
from sonobelief.datasets.planes import PlaneSet
from sonobelief.image_ae import PlaneAE

matplotlib.use("Agg")


def soft_dice_loss(logits, target, channels, eps=1.0):
    p = torch.sigmoid(logits[:, channels])
    t = target[:, channels]
    inter = (p * t).sum((2, 3))
    dice = (2 * inter + eps) / (p.sum((2, 3)) + t.sum((2, 3)) + eps)
    return 1 - dice.mean()


def loss_fn(logits, x, cfg):
    """Returns total loss and a dict of its parts."""
    if "lambda_liver" not in cfg:  # legacy: both channels together
        bce = F.binary_cross_entropy_with_logits(logits, x)
        dice = soft_dice_loss(logits, x, cfg["dice_channels"])
        return bce + cfg["lambda_dice"] * dice, {"bce": bce, "dice_loss": dice}
    liver_bce = F.binary_cross_entropy_with_logits(logits[:, 0], x[:, 0])
    liver_dice = soft_dice_loss(logits, x, [0])
    fov_bce = F.binary_cross_entropy_with_logits(logits[:, 1], x[:, 1])
    total = cfg["lambda_liver"] * (liver_bce + cfg["lambda_dice"] * liver_dice) + cfg["lambda_fov"] * fov_bce
    return total, {"liver_bce": liver_bce, "liver_dice_loss": liver_dice, "fov_bce": fov_bce}


@torch.no_grad()
def evaluate(model, planes, cfg, bs=1024):
    model.eval()
    tot, n = {}, 0
    dice = {0: [], 1: []}  # per channel: per-plane Dice (nan where the target channel is empty)
    fp_empty = []          # predicted liver pixels on planes with no liver
    for x, idx in planes.batches(bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        logits = logits.float()
        loss, parts = loss_fn(logits, x, cfg)
        for k, v in {"loss": loss, **parts}.items():
            tot[k] = tot.get(k, 0.0) + v.item() * len(x)
        n += len(x)
        pred = logits > 0
        for c in dice:
            t = x[:, c] > 0.5
            inter = (pred[:, c] & t).sum((1, 2)).float()
            s = (pred[:, c].sum((1, 2)) + t.sum((1, 2))).float()
            d = 2 * inter / s.clamp(min=1)
            d[~t.any(1).any(1)] = float("nan")
            dice[c].append(d)
        empty = ~(x[:, 0] > 0.5).any(1).any(1)
        fp_empty.append(pred[empty, 0].sum((1, 2)).float())
    res = {k: v / n for k, v in tot.items()}
    liver, fov = torch.cat(dice[0]), torch.cat(dice[1])
    win = torch.from_numpy(planes.poses["window"]).to(liver.device)
    res["liver_dice"] = liver.nanmean().item()
    for w, name in [(0, "general"), (1, "subxiphoid"), (2, "off_target")]:
        res[f"liver_dice_{name}"] = liver[win == w].nanmean().item() if (win == w).any() else float("nan")
    res["fov_dice"] = fov.nanmean().item()
    fp = torch.cat(fp_empty)
    res["fp_px_empty"] = fp.mean().item() if len(fp) else float("nan")
    res["n_empty"] = len(fp)
    model.train()
    return res


@torch.no_grad()
def save_recon(model, planes, path, k=8):
    model.eval()
    idx = torch.linspace(0, len(planes) - 1, k, device=planes.device).long()
    from sonobelief.datasets.planes import unpack
    x = unpack(planes.packed[idx])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    p = torch.sigmoid(logits.float()).cpu().numpy()
    x = x.cpu().numpy()
    fig, axs = plt.subplots(3, k, figsize=(2 * k, 6.3))
    for j in range(k):
        show = lambda m: m[1] * 0.35 + m[0] * 0.65
        axs[0, j].imshow(show(x[j]), cmap="magma", vmin=0, vmax=1)
        axs[1, j].imshow(show(p[j]), cmap="magma", vmin=0, vmax=1)
        axs[2, j].imshow(np.abs((p[j, 0] > 0.5) - x[j, 0]), cmap="Reds", vmin=0, vmax=1)
        for a in axs[:, j]:
            a.set_axis_off()
    for a, t in zip(axs[:, 0], ["input", "reconstruction", "liver error"]):
        a.set_title(t, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
    model.train()


@torch.no_grad()
def export_embeddings(model, run_dir, device, min_component_px=0):
    model.eval()
    for split in ["train", "val", "test"]:
        planes = PlaneSet(split, device, min_component_px=min_component_px)  # all windows
        zs = []
        for x, _ in planes.batches(2048):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zs.append(model.enc(x).float().cpu())
        np.savez_compressed(os.path.join(run_dir, f"embeddings_{split}.npz"),
                            z=torch.cat(zs).numpy().astype(np.float16),
                            study=np.array(planes.studies)[planes.study_idx.cpu().numpy()],
                            **planes.poses)
        print(f"exported {split}: {len(planes)} embeddings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(paths.CONFIGS, "image_encoder.json"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--export", action="store_true", help="encode all planes with best.pt after training")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.epochs:
        cfg["epochs"] = args.epochs

    torch.manual_seed(cfg["seed"])
    device = "cuda"
    run_dir = os.path.join(paths.RUNS, f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)

    w = tuple(cfg.get("windows", (0, 1, 2)))
    mc = cfg.get("min_component_px", 0)  # runs before the small-component filter used raw masks
    train, val = (PlaneSet(sp, device, windows=w, min_component_px=mc) for sp in ("train", "val"))
    print(f"train {len(train)} planes ({len(train.studies)} studies), val {len(val)} ({len(val.studies)})")
    model = PlaneAE(2, cfg["latent_dim"]).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(train) / cfg["batch_size"])
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    best = -1.0
    log_keys = None
    with open(os.path.join(run_dir, "log.csv"), "w", newline="") as f:
        log = csv.writer(f)
        for ep in range(1, cfg["epochs"] + 1):
            t0, run, n = time.time(), 0.0, 0
            for x, _ in train.batches(cfg["batch_size"], shuffle=True, generator=gen):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(x)
                loss, _ = loss_fn(logits.float(), x, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                sched.step()
                run += loss.item() * len(x)
                n += len(x)
            v = evaluate(model, val, cfg)
            dt = time.time() - t0
            if log_keys is None:
                log_keys = list(v)
                log.writerow(["epoch", "train_loss", *[f"val_{k}" for k in log_keys], "sec"])
            log.writerow([ep, f"{run / n:.4f}", *[f"{v[k]:.4f}" for k in log_keys], f"{dt:.0f}"])
            f.flush()
            print(f"ep {ep:3d}  train {run / n:.4f}  val {v['loss']:.4f}  liver Dice {v['liver_dice']:.4f} "
                  f"(gen {v['liver_dice_general']:.3f} subx {v['liver_dice_subxiphoid']:.3f} "
                  f"off {v['liver_dice_off_target']:.3f})  fov Dice {v['fov_dice']:.4f}  "
                  f"FP px on empty {v['fp_px_empty']:.1f}  ({dt:.0f}s)", flush=True)
            state = {"model": model.state_dict(), "config": cfg, "epoch": ep, "val": v}
            torch.save(state, os.path.join(run_dir, "last.pt"))
            if v["liver_dice"] > best:
                best = v["liver_dice"]
                torch.save(state, os.path.join(run_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"))["model"])
    save_recon(model, val, os.path.join(run_dir, "recon_val.png"))
    print(f"best val liver Dice {best:.4f}  -> {run_dir}")
    if args.export:
        export_embeddings(model, run_dir, device, cfg.get("min_component_px", 0))


if __name__ == "__main__":
    main()
