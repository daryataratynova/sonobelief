"""Target-pose head trained on INFERRED GB geometry (train / val only): the last two checks of the shape branch.

    z_L -> p(r_GB, a_GB, l_GB | z_L) -> g(P | r_GB, a_GB, l_GB),  g trained on samples of the conditional

Conditional (gb_geometry.py): r_GB ridge (lambda 0.1), log l ridge (lambda 1, train OOF), Q = a a^T by
  --axis ridge   linear ridge (lambda 0.1, train OOF)
  --axis krr     RBF kernel ridge on standardized z_L; bandwidth (x median pairwise distance) and lambda chosen by
                 10-fold out-of-fold mean axis angle on train
  noise: joint out-of-fold residual tuples (r, Q, log l) of the train patients.
Training states (train patients): out-of-fold mean of the patient + the residual tuple of ANOTHER train patient,
  drawn afresh per example, so the head sees geometry with the deployment-time uncertainty (never the exact one).
Val: per patient the in-sample prediction from z_L + M residual tuples; s(P) = mean_m sigmoid g(state_m, P).
Model selection: mean per-patient AUPRC on the val GB-blind bank (M_SEL samples); final evaluation with M samples.
References (paired patient bootstrap): pose-only head (target_pose_none), exact-geometry head with predicted
  geometry (gb_geometry.py "pred_all").
Output: runs/target_pose_inferred_<axis>_<ts>/{config.json, log.csv, best.pt, results.json, per_patient.csv}
Usage: python -m sonobelief.train.geometry_head --axis krr        (frozen head; --axis ridge is the ridge ablation)
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
from sonobelief.gb_geometry import geometry, quality, scores, state_of
from sonobelief.geometry_conditional import BLOCKS, oof, predictors
from sonobelief.nets import DObs
from sonobelief.target_poses import PoseSet, evaluate, evaluate_fast, fmt, summarise

M_SEL, M = 8, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["ridge", "krr"], default="ridge")
    ap.add_argument("--config", default=os.path.join(paths.CONFIGS, "exact_geometry_head.json"))
    ap.add_argument("--pose_only", default=os.path.join(paths.WEIGHTS, "pose_head"))
    ap.add_argument("--exact_head", default=os.path.join(paths.WEIGHTS, "exact_geometry_head"))
    args = ap.parse_args()
    cfg = dict(json.load(open(args.config)), name=f"target_pose_inferred_{args.axis}", state="gb_inferred", axis=args.axis)
    run_dir = os.path.join(paths.RUNS, f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    zl_run = os.path.join(paths.REPO, cfg["zl_run"])
    tr_v = Views("train", zl_run)
    va_v = Views("val", zl_run, tr_v.stats)
    tr = PoseSet(tr_v, ("dense", "blind"), "gb")
    va = PoseSet(va_v, ("blind",), "gb")
    ti, vi = torch.tensor(tr.patients, device="cuda"), torch.tensor(va.patients, device="cuda")
    Xtr, Ytr, Xva, Yva = tr_v.zl[ti], geometry(tr_v, ti), va_v.zl[vi], geometry(va_v, vi)

    # ---- conditional p(r, a, l | z_L)
    fns, info = predictors(args.axis, Xtr, Ytr)
    O = torch.cat([oof(fns[k], Xtr, Ytr[:, b]) for k, b in BLOCKS.items()], 1)
    res = Ytr - O
    Pva = torch.cat([fns[k](Xtr, Ytr[:, b], Xva) for k, b in BLOCKS.items()], 1)
    qual = {"train_oof": quality(Ytr, O), "val": quality(Yva, Pva)}
    for k, q in qual.items():
        print(f"{k:10s} r_GB {q['r_gb_err_mm']:5.1f} mm | axis {q['axis_err_deg']:5.1f} deg (med {q['axis_err_deg_median']:5.1f}) "
              f"| length {q['length_err_mm']:5.1f} mm", flush=True)

    # ---- per-patient lookup (patient index of Views -> row of O / Pva)
    row_tr = torch.full((len(tr_v.studies),), -1, dtype=torch.long, device="cuda")
    row_tr[ti] = torch.arange(len(ti), device="cuda")
    n_res = len(res)
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])

    def val_states(m, gen):
        j = torch.randint(n_res, (len(vi), m), device="cuda", generator=gen)
        st = torch.zeros(len(va_v.studies), m, 10, device="cuda")
        st[vi] = state_of(Pva[:, None] + res[j])
        return st

    S_sel = val_states(M_SEL, torch.Generator(device="cuda").manual_seed(1))
    torch.manual_seed(cfg["seed"])
    head = DObs(10, e_dim=1, width=cfg["width"], depth=cfg["depth"]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    T, W = cfg["steps"], cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, T - W))))
    log = csv.writer(open(os.path.join(run_dir, "log.csv"), "w", newline=""))
    log.writerow(["step", "train_bce", "val_auroc", "val_auprc"])
    best, since, t0, run = -1.0, 0, time.time(), None
    for step in range(1, T + 1):
        i = torch.randint(len(tr), (cfg["batch"],), device="cuda", generator=g)
        r = row_tr[tr.pat[i]]
        j = torch.randint(n_res - 1, (len(i),), device="cuda", generator=g)
        j = j + (j >= r).long()                                     # another patient's residual tuple
        st = state_of(O[r] + res[j])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = head(st, tr.pose[i, None])[:, 0, 0].float()
        loss = F.binary_cross_entropy_with_logits(logit, tr.y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % cfg["eval_every"] == 0 or step == T:
            fast = evaluate_fast(scores(head, va, S_sel), va)
            head.train()
            va_auc, va_ap = (float(np.mean([x[k] for x in fast])) for k in ("auroc", "auprc"))
            log.writerow([step, run, va_auc, va_ap])
            print(f"step {step:6d} | train bce {run:.4f} | val auroc {va_auc:.3f} auprc {va_ap:.3f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if va_ap > best:
                best, since = va_ap, 0
                torch.save({"model": head.state_dict(), "config": cfg, "step": step, "val_auprc": va_ap},
                           os.path.join(run_dir, "best.pt"))
            else:
                since += 1
                if since >= cfg["patience"]:
                    print(f"early stop at {step}", flush=True)
                    break

    ck = torch.load(os.path.join(run_dir, "best.pt"))
    head.load_state_dict(ck["model"])
    rows = evaluate(scores(head, va, val_states(M, torch.Generator(device="cuda").manual_seed(2))), va)
    out = {"best_step": ck["step"], "conditional": info, "quality": qual, "inferred_head": summarise(rows)}
    print(fmt("inferred_head", out["inferred_head"]), flush=True)
    df = pd.DataFrame(rows).set_index("study")
    refs = {"pose_only": pd.read_csv(os.path.join(args.pose_only, "per_patient.csv")).query("variant == 'model'").set_index("study")}
    p = os.path.join(args.exact_head, "geometry_ceiling_per_patient.csv")
    if os.path.exists(p):
        refs["exact_head_pred_all"] = pd.read_csv(p).query("variant == 'pred_all'").set_index("study")
    rng = np.random.default_rng(0)
    out["bootstrap"] = {}
    for rn, ref in refs.items():
        for k in ("auprc", "auroc", "recall1", "recall10", "top1_dist"):
            d = (df[k] - ref.loc[df.index, k]).dropna().values
            bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            out["bootstrap"][f"{rn}:{k}"] = [d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)]
            print(f"  inferred_head - {rn:20s} {k:9s} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    df.reset_index().to_csv(os.path.join(run_dir, "per_patient.csv"), index=False)
    json.dump(out, open(os.path.join(run_dir, "results.json"), "w"), indent=1, default=float)
    print(f"done -> {run_dir}")
