"""Oracle target-pose model g(state, P) = P(P in T_subcostal-LA | state) (train / val only).

    g_theta(s, P^L) -> logit P(P in T_subcostal-LA),  s = cfg "state":
      "zl"      true z_L (standardized, as train.observation_model.Views)                         main: liver shape alone
      "zl_rgb"  [z_L, r_GB / 100] (true GB offset from the liver centroid)      reference: shape + GB position
      "gb"      [r_GB / 100, a_GB, l_GB / 100] without z_L (true GB geometry only)
      "zl_gb"   [z_L, r_GB / 100, a_GB, l_GB / 100]                              reference: shape + GB geometry
      "none"    no state (a constant)                                            pose prior in the liver frame
      a_GB = true GB long axis (PCA, gb_target), encoded sign-invariantly as the 6 unique entries of a a^T;
      l_GB = GB length (2-98% extent along a, mm); both from the dense bank file (gb_axes, gb_length).
P^L = probe pose in the liver-centred CT frame [(apex - c_L) / 100 mm, beam, lateral], lateral in the canonical
roll (theta in [0, pi); labels are invariant to the 180 deg flip about the beam).  Model: latent_models.DObs with
1 output; Gaussian noise state_noise on the state during training (as D_obs).
Data (frozen T_subcostal-LA labels, applicable studies):
  train  both banks of the train patients: the GB-aimed dense bank (gb_target_subcostal, 4000 poses) and the
         GB-blind bank (target_bank_blind, 8000 poses)
  val    model selection on the val GB-blind bank: mean per-patient AUPRC (the ranking endpoint; BCE is not used
         because the train prevalence (both banks, 2.0%) differs from the blind bank's (~0.8%)); evaluation on the val GB-blind bank only, the candidate
         set that does not encode the GB position.
Evaluation per val patient (applicable and available; ranking metrics on patients with >= 1 positive in the bank):
  auroc, auprc (average precision), recall@1 / @10 (top-N contains a pose of T), top-1 distance to T = skin mm +
  rotation deg (mod the beam flip) from the top-scored candidate to the nearest known T pose (dense + blind banks),
  and the same for the best of the top 10.  References: random ranking (prevalence-based expectations), and the
  main model's controls "mean_state" (z_L = 0) and "other_patient" (another val patient's z_L).
Output: runs/<name>_<ts>/{config.json, log.csv, best.pt, results.json, per_patient.csv}
Usage: python -m sonobelief.train.pose_head --config configs/pose_head.json        (exact-geometry head: configs/exact_geometry_head.json)
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sonobelief import paths
from sonobelief.datasets.views import Views
from sonobelief.nets import DObs
from sonobelief.target_poses import PoseSet, evaluate, evaluate_fast, fmt, predict, random_dist, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    run_dir = os.path.join(paths.RUNS, f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    zl_run = os.path.join(paths.REPO, cfg["zl_run"])
    tr_v = Views("train", zl_run)
    va_v = Views("val", zl_run, tr_v.stats)
    tr = PoseSet(tr_v, ("dense", "blind"), cfg["state"])
    va = PoseSet(va_v, ("blind",), cfg["state"])
    print(f"train {len(tr)} poses ({tr.y.mean():.4f} in T, {len(tr.patients)} pts) | val blind {len(va)} "
          f"({va.y.mean():.4f}, {len(va.patients)} pts)", flush=True)
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    model = DObs(tr.state.shape[1], e_dim=1, width=cfg["width"], depth=cfg["depth"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    T, W = cfg["steps"], cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, T - W))))
    log = csv.writer(open(os.path.join(run_dir, "log.csv"), "w", newline=""))
    log.writerow(["step", "train_bce", "val_bce", "val_auroc", "val_auprc"])
    noise = cfg["state_noise"] if cfg["state"] != "none" else 0.0
    best, since, t0, run = -1.0, 0, time.time(), None
    for step in range(1, T + 1):
        i = torch.randint(len(tr), (cfg["batch"],), device="cuda", generator=gen)
        st = tr.state[tr.pat[i]]
        st = st + noise * torch.randn(st.shape, device="cuda", generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(st, tr.pose[i, None])[:, 0, 0].float()
        loss = F.binary_cross_entropy_with_logits(logit, tr.y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % cfg["eval_every"] == 0 or step == T:
            lv = predict(model, va, va.state)
            vb = F.binary_cross_entropy_with_logits(lv, va.y).item()
            fast = evaluate_fast(lv, va)
            va_auc, va_ap = (float(np.mean([r[k] for r in fast])) for k in ("auroc", "auprc"))
            log.writerow([step, run, vb, va_auc, va_ap])
            print(f"step {step:6d} | train bce {run:.4f} | val bce {vb:.4f} auroc {va_auc:.3f} auprc {va_ap:.3f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if va_ap > best:
                best, since = va_ap, 0
                torch.save({"model": model.state_dict(), "config": cfg, "step": step, "val_bce": vb, "val_auprc": va_ap},
                           os.path.join(run_dir, "best.pt"))
            else:
                since += 1
                if since >= cfg["patience"]:
                    print(f"early stop at {step}", flush=True)
                    break

    ck = torch.load(os.path.join(run_dir, "best.pt"))
    model.load_state_dict(ck["model"])
    res = {"best_step": ck["step"], "val_bce": ck["val_bce"], "val_auprc_select": ck["val_auprc"], "random_top1_dist_median": random_dist(va, gen)}
    controls = {"model": va.state}
    if cfg["state"] != "none":
        controls["mean_state"] = torch.zeros_like(va.state)
        controls["other_patient"] = va.state[torch.tensor(np.roll(np.arange(len(va.state)), 1), device="cuda")]
    per = []
    for name, st in controls.items():
        rows = evaluate(predict(model, va, st), va)
        res[name] = summarise(rows)
        per += [{"variant": name, **r} for r in rows]
        print(fmt(name, res[name]), flush=True)
    print(f"random candidate: expected distance to T, median over patients {res['random_top1_dist_median']:.1f}")
    pd.DataFrame(per).to_csv(os.path.join(run_dir, "per_patient.csv"), index=False)
    json.dump(res, open(os.path.join(run_dir, "results.json"), "w"), indent=1, default=float)
    print(f"done -> {run_dir}")
