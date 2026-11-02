"""Step 1b: pose-independent liver-shape latent z_L (liver only; GB is never an input).

    L_centered (liver occupancy, centred on its centroid, CT RAS axes, 64^3 at 5 mm)
        --E_L--> z_L --D_occ--> liver occupancy logits (64^3)
    loss = BCE + soft Dice.  Optional mild augmentation (config "aug"): the same random
    rotation / anisotropic scale is applied to input and target, so z_L still describes
    the shape it sees; only real, unaugmented livers are used for evaluation and probes.
The teacher state is then z_world = [p_L, z_L, z_GB] with p_L = liver placement relative to
P_ref (offset + rotation, explicit) and z_GB = r_GB = c_GB - c_L (explicit, CT RAS mm).

Frozen evaluation on val patients (fit on train z_L):
  recon Dice / IoU                 vs. the population-mean liver (train mean occupancy,
                                   threshold chosen on train)
  probes (ridge and MLP on z_L)    liver volume (ml), RAS extents x/y/z (mm): R^2 and MAE vs
                                   predicting the train mean
  cross-organ (information only)   z_L -> r_GB error vs. the train-mean r_GB
Run dir runs/<name>_<timestamp>/: config.json, log.csv, best.pt (by val recon Dice),
probes.json, recon_val.png, z_L_<split>.npz (z_L, study, r_GB) for downstream use.
Usage: python -m sonobelief.train.liver_shape --config configs/liver_shape.json
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
from torch import nn

from sonobelief import paths
from sonobelief.nets import block3d
from sonobelief.paths import ANATOMY
from sonobelief.regression import fit_mlp, ridge
from sonobelief.simulate.anatomy import VOX_MM
from sonobelief.table import load_table

matplotlib.use("Agg")


class LiverSet:
    """Liver occupancy (S, 1, 64, 64, 64) float16 in [0, 1] on the GPU + per-patient targets."""

    def __init__(self, split, device="cuda"):
        tab = load_table(split)
        self.studies = list(tab.index)
        vol, ext, rgb = [], [], []
        for s in self.studies:
            z = np.load(os.path.join(ANATOMY, f"{s}.npz"))
            liv = z["vol"][1].astype(np.float32) / 255  # 128^3 at 2.5 mm
            m = liv > 0.5
            ext.append([(np.ptp(np.flatnonzero(m.any(axis=tuple(a for a in range(3) if a != k)))) + 1) * VOX_MM
                        for k in range(3)])
            vol.append(torch.from_numpy(liv)[None, None])
            rgb.append(z["gb_c"] - z["liver_c"])
        v = torch.cat(vol).to(device)
        self.x = F.avg_pool3d(v, 2).half()                                   # 64^3 at 5 mm
        self.volume = torch.tensor(tab["liver_ml"].values, device=device).float()
        self.extent = torch.tensor(np.array(ext), device=device).float()      # mm, RAS x/y/z
        self.r_gb = torch.tensor(np.array(rgb), device=device).float()        # mm, CT RAS

    def __len__(self):
        return len(self.studies)


def augment(x, gen, rot_deg, scale):
    """Same random rotation (about a random axis, <= rot_deg) and anisotropic scale for a batch."""
    B, dev = len(x), x.device
    ax = torch.randn(B, 3, device=dev, generator=gen)
    ax = ax / ax.norm(dim=1, keepdim=True)
    ang = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * math.radians(rot_deg)
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -ax[:, 2], ax[:, 1], -ax[:, 0]
    K = K - K.transpose(1, 2)
    R = torch.eye(3, device=dev) + torch.sin(ang)[:, None, None] * K + (1 - torch.cos(ang))[:, None, None] * K @ K
    S = 1 + (torch.rand(B, 3, device=dev, generator=gen) * 2 - 1) * scale
    A = torch.cat([R / S[:, None, :], torch.zeros(B, 3, 1, device=dev)], 2)  # output -> input coordinates
    grid = F.affine_grid(A, list(x.shape), align_corners=False)
    return F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class ShapeAE(nn.Module):
    def __init__(self, z_dim=32, widths=(32, 64, 128, 256)):
        super().__init__()
        chans = [1, *widths]
        self.enc = nn.Sequential(*[block3d(a, b) for a, b in zip(chans[:-1], chans[1:])], nn.Flatten(),
                                 nn.Linear(widths[-1] * 4**3, 512), nn.SiLU(), nn.Linear(512, z_dim))
        self.fc = nn.Linear(z_dim, 256 * 4**3)
        up = [(256, 128), (128, 64), (64, 32), (32, 32)]  # 4 -> 64
        self.dec = nn.Sequential(*[nn.Sequential(nn.ConvTranspose3d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU())
                                   for a, b in up], nn.Conv3d(32, 1, 3, 1, 1))

    def decode(self, z):
        return self.dec(self.fc(z).view(-1, 256, 4, 4, 4))

    def forward(self, x, z_noise=0.0):
        z = self.enc(x)
        if z_noise and self.training:
            z = z + z_noise * z.detach().std(0, keepdim=True) * torch.randn_like(z)
        return self.decode(z), z


def loss_fn(logits, t):
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = torch.sigmoid(logits)
    dice = (2 * (p * t).sum((1, 2, 3, 4)) + 1) / (p.sum((1, 2, 3, 4)) + t.sum((1, 2, 3, 4)) + 1)
    return bce + (1 - dice.mean())


def hard_scores(pred, t):
    """pred, t bool (N, ...) -> Dice, IoU per item."""
    p, t = pred.flatten(1), t.flatten(1)
    i = (p & t).sum(1).float()
    s = (p.sum(1) + t.sum(1)).float()
    u = (p | t).sum(1).float()
    return 2 * i / s.clamp(min=1), i / u.clamp(min=1)


@torch.no_grad()
def encode_eval(model, L, bs=32):
    model.eval()
    zs, dice, iou = [], [], []
    for i in range(0, len(L), bs):
        x = L.x[i:i + bs].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, z = model(x)
        d, u = hard_scores(logits.float() > 0, x > 0.5)
        zs.append(z.float()); dice.append(d); iou.append(u)
    model.train()
    return torch.cat(zs), torch.cat(dice), torch.cat(iou)


def r2(p, y):
    return (1 - ((p - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).mean().item()


def probes(ztr, zva, Ltr, Lva):
    """Frozen probes on z_L: fit on train patients, evaluate on val."""
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-6
    a, b = (ztr - mu) / sd, (zva - mu) / sd
    one = lambda z: torch.cat([z, torch.ones(len(z), 1, device=z.device)], 1)
    out = {}
    for name, ytr, yva in [("volume_ml", Ltr.volume[:, None], Lva.volume[:, None]),
                           ("extent_mm", Ltr.extent, Lva.extent),
                           ("r_gb_mm", Ltr.r_gb, Lva.r_gb)]:
        p_lin = one(b) @ ridge(one(a), ytr, lam=0.1)
        p_mlp = fit_mlp(a, ytr, b, steps=2000, wd=1e-2)
        base = ytr.mean(0).expand_as(yva)
        if name == "r_gb_mm":
            err = lambda p: (p - yva).norm(dim=1).mean().item()
        else:
            err = lambda p: (p - yva).abs().mean().item()
        out[name] = {"train_mean_err": err(base), "ridge_err": err(p_lin), "mlp_err": err(p_mlp),
                     "ridge_r2": r2(p_lin, yva), "mlp_r2": r2(p_mlp, yva)}
    return out


def save_recon(model, L, path, k=6):
    model.eval()
    idx = torch.linspace(0, len(L) - 1, k).long()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.sigmoid(model(L.x[idx].float())[0].float()).cpu().numpy()
    x = L.x[idx].float().cpu().numpy()
    fig, axs = plt.subplots(4, k, figsize=(2.2 * k, 8.8))
    for j in range(k):
        for r, (im, ax_) in enumerate([(x[j, 0], 2), (p[j, 0], 2), (x[j, 0], 0), (p[j, 0], 0)]):
            axs[r, j].imshow(im.max(ax_).T if ax_ == 2 else im.max(0).T, origin="lower", cmap="magma", vmin=0, vmax=1)
            axs[r, j].set_axis_off()
    for a, t in zip(axs[:, 0], ["liver (MIP, axial)", "D_occ(z_L)", "liver (MIP, sagittal)", "D_occ(z_L)"]):
        a.set_title(t, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(paths.CONFIGS, "liver_shape.json"))
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    torch.manual_seed(cfg["seed"])
    run_dir = os.path.join(paths.RUNS, f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    L = {sp: LiverSet(sp) for sp in ("train", "val", "test")}
    tr, va = L["train"], L["val"]

    # population-mean liver baseline (threshold chosen on train)
    mean_occ = tr.x.float().mean(0, keepdim=True)
    th = max(np.linspace(0.1, 0.7, 13), key=lambda t: hard_scores((mean_occ > t).expand_as(tr.x), tr.x > 0.5)[0].mean().item())
    md, mi = hard_scores((mean_occ > th).expand_as(va.x), va.x > 0.5)
    base = {"mean_liver_dice": md.mean().item(), "mean_liver_iou": mi.mean().item()}
    print(f"train {len(tr)} livers, val {len(va)}; population-mean liver on val: Dice {base['mean_liver_dice']:.3f}")

    model = ShapeAE(cfg["z_dim"]).cuda()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    T, W = cfg["steps"], cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, T - W))))
    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    best, keys, run = -1.0, None, 0.0
    with open(os.path.join(run_dir, "log.csv"), "w", newline="") as f:
        log = csv.writer(f)
        t0 = time.time()
        for step in range(1, T + 1):
            x = tr.x[torch.randint(len(tr), (cfg["batch"],), device="cuda", generator=gen)].float()
            if cfg.get("aug"):
                x = augment(x, gen, cfg["aug_rot_deg"], cfg["aug_scale"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, cfg.get("z_noise", 0.0))
            loss = loss_fn(logits.float(), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            if step % cfg["eval_every"] == 0 or step == T:
                _, dtr, _ = encode_eval(model, tr)
                _, dva, iva = encode_eval(model, va)
                row = {"step": step, "train_loss": run / cfg["eval_every"], "train_dice": dtr.mean().item(),
                       "val_dice": dva.mean().item(), "val_iou": iva.mean().item(), "min": (time.time() - t0) / 60}
                run = 0.0
                if keys is None:
                    keys = list(row)
                    log.writerow(keys)
                log.writerow([f"{row[k]:.4f}" if isinstance(row[k], float) else row[k] for k in keys])
                f.flush()
                print(f"step {step:6d} loss {row['train_loss']:.4f} | recon Dice train {row['train_dice']:.3f} "
                      f"val {row['val_dice']:.3f} (IoU {row['val_iou']:.3f}; mean liver {base['mean_liver_dice']:.3f}) "
                      f"| {row['min']:.1f} min", flush=True)
                state = {"model": model.state_dict(), "config": cfg, "step": step}
                torch.save(state, os.path.join(run_dir, "last.pt"))
                if row["val_dice"] > best:
                    best = row["val_dice"]
                    torch.save(state, os.path.join(run_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)["model"])
    Z = {}
    for sp in L:
        z, d, u = encode_eval(model, L[sp])
        Z[sp] = (z, d, u)
        np.savez(os.path.join(run_dir, f"z_L_{sp}.npz"), z_L=z.cpu().numpy(), study=np.array(L[sp].studies),
                 r_gb=L[sp].r_gb.cpu().numpy(), recon_dice=d.cpu().numpy())
    zt = Z["train"][0]
    ev = torch.linalg.eigvalsh(torch.cov((zt - zt.mean(0)).T)).flip(0)
    res = {"best_step": torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)["step"],
           "recon_dice_train": Z["train"][1].mean().item(), "recon_dice_val": Z["val"][1].mean().item(),
           "recon_iou_val": Z["val"][2].mean().item(), **base,
           "z_participation_ratio": (1 / ((ev / ev.sum()) ** 2).sum()).item(),
           "probes_val": probes(zt, Z["val"][0], tr, va)}
    json.dump(res, open(os.path.join(run_dir, "probes.json"), "w"), indent=1)
    save_recon(model, va, os.path.join(run_dir, "recon_val.png"))
    print(json.dumps(res, indent=1))
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
