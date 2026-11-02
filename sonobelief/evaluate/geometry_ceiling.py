"""GB geometry | z_L, and the target-pose ceiling with inferred geometry (train / val only).

    z_L -> p(r_GB, a_GB, l_GB | z_L) -> g(P | r_GB, a_GB, l_GB)   (g = frozen geometry head, train_target_pose "gb")

Conditional (train patients with an applicable GB target, i.e. those g was trained on):
  targets   r_GB (mm, liver frame), Q = a a^T (6 unique entries; identical for a and -a), log l_GB
  mean      ridge on standardized z_L per block: r_GB with the frozen lambda 0.1 (belief.rgb_conditional);
            Q and log l with lambda chosen by 10-fold out-of-fold MSE on train (grid LAMS)
  noise     joint out-of-fold residual tuples (r, Q, log l) of the train patients, resampled as whole tuples
            (keeps e.g. position / orientation deviations together)
  sample m  y_m = mean(z_L) + res_j(m);  a_m = principal eigenvector of sym(Q_m);  l_m = exp(log l_m)
Prediction quality (train out-of-fold, val): r_GB error (mm), axis angle (deg, sign-invariant), length error (mm),
  each against the population mean (axis: principal eigenvector of the mean Q).
Target-pose scores (val GB-blind bank, true z_L): s(P) = mean_m sigmoid g(state_m, P), M samples. Rows:
  true_geometry       all true (the oracle "gb" result)
  pred_r              r_GB predicted, a / l true
  pred_r_a            r_GB, a_GB predicted, l true
  pred_r_l            r_GB, l_GB predicted, a true
  pred_all            all predicted from the true z_L            <- the ceiling with shape known, geometry inferred
  population          geometry of random train patients (no anatomy information through the same head)
  --all_masks (post hoc, 2026-10-30): also pred_a, pred_l, pred_a_l (axis / length / both predicted, the rest true),
  appended after the rows above so their numbers and CIs are unchanged; written to geometry_ceiling_val_full.json
Metrics: train_target_pose.evaluate / summarise; paired patient bootstrap vs pose-only and true geometry.
Output: <gb_head_run>/geometry_ceiling_val.json, geometry_ceiling_per_patient.csv
Usage: python -m sonobelief.evaluate.geometry_ceiling --head weights/exact_geometry_head [--pose_only weights/pose_head]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from sonobelief import paths
from sonobelief.datasets.views import Views
from sonobelief.gb_geometry import LAM_R, LAMS, M, geometry, oof, quality, scores, state_of
from sonobelief.nets import DObs
from sonobelief.regression import ridge_pred
from sonobelief.target_poses import PoseSet, evaluate, fmt, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--all_masks", action="store_true", help="add pred_a, pred_l, pred_a_l (post hoc)")
    ap.add_argument("--pose_only", default=os.path.join(paths.WEIGHTS, "pose_head"))
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.head, "best.pt"))
    cfg = ck["config"]
    assert cfg["state"] == "gb"
    zl_run = os.path.join(paths.REPO, cfg["zl_run"])
    tr_v = Views("train", zl_run)
    va_v = Views("val", zl_run, tr_v.stats)
    tr = PoseSet(tr_v, ("blind",), "gb")                         # patient list (the applicable train patients)
    va = PoseSet(va_v, ("blind",), "gb")
    head = DObs(10, e_dim=1, width=cfg["width"], depth=cfg["depth"]).cuda()
    head.load_state_dict(ck["model"])
    ti = torch.tensor(tr.patients, device="cuda")
    vi = torch.tensor(va.patients, device="cuda")
    Xtr, Ytr, Xva, Yva = tr_v.zl[ti], geometry(tr_v, ti), va_v.zl[vi], geometry(va_v, vi)
    blocks = {"r": slice(0, 3), "Q": slice(3, 9), "logl": slice(9, 10)}
    lam = {"r": LAM_R}
    for k in ("Q", "logl"):
        err = {l: ((oof(Xtr, Ytr[:, blocks[k]], l) - Ytr[:, blocks[k]]) ** 2).mean().item() for l in LAMS}
        lam[k] = min(err, key=err.get)
        print(f"lambda {k}: train OOF MSE " + ", ".join(f"{l:g} {e:.4f}" for l, e in err.items()) + f" -> {lam[k]:g}")
    O = torch.cat([oof(Xtr, Ytr[:, b], lam[k]) for k, b in blocks.items()], 1)
    res = Ytr - O                                                                    # joint OOF residual tuples
    Pva = torch.cat([ridge_pred(Xtr, Ytr[:, b], Xva, lam[k]) for k, b in blocks.items()], 1)
    pop = Ytr.mean(0, keepdim=True)
    out = {"lambda": lam, "n_train": len(ti), "n_val": len(vi),
           "quality": {"train_oof": quality(Ytr, O), "train_population": quality(Ytr, pop.expand_as(Ytr)),
                       "val": quality(Yva, Pva), "val_population": quality(Yva, pop.expand_as(Yva))}}
    for k, q in out["quality"].items():
        print(f"{k:17s} r_GB {q['r_gb_err_mm']:5.1f} mm | axis {q['axis_err_deg']:5.1f} deg (med {q['axis_err_deg_median']:5.1f}) "
              f"| length {q['length_err_mm']:5.1f} mm", flush=True)

    g = torch.Generator(device="cuda").manual_seed(0)
    S = len(va_v.studies)
    full = lambda Y: torch.zeros(S, *Y.shape[1:], device="cuda").index_copy_(0, vi, Y)   # per val patient index
    j = torch.randint(len(res), (len(vi), M), device="cuda", generator=g)
    pred = Pva[:, None] + res[j]                                                     # (n_val, M, 10)
    true = Yva[:, None].expand(-1, M, -1)
    mix = lambda keep_pred: torch.where(keep_pred, pred, true)
    sel = lambda *bl: torch.zeros(10, dtype=torch.bool, device="cuda").index_fill_(
        0, torch.cat([torch.arange(10, device="cuda")[blocks[b]] for b in bl]), True)
    variants = {"true_geometry": true[:, :1], "pred_r": mix(sel("r")), "pred_r_a": mix(sel("r", "Q")),
                "pred_r_l": mix(sel("r", "logl")), "pred_all": pred,
                "population": Ytr[torch.randint(len(Ytr), (len(vi), M), device="cuda", generator=g)]}
    if args.all_masks:
        variants.update({"pred_a": mix(sel("Q")), "pred_l": mix(sel("logl")), "pred_a_l": mix(sel("Q", "logl"))})
    per = []
    for name, Y in variants.items():
        rows = evaluate(scores(head, va, full(state_of(Y))), va)
        out[name] = summarise(rows)
        per += [{"variant": name, **r} for r in rows]
        print(fmt(name, out[name]), flush=True)
    po = pd.read_csv(os.path.join(args.pose_only, "per_patient.csv")).query("variant == 'model'").set_index("study")
    df = pd.DataFrame(per)
    rng = np.random.default_rng(0)
    out["bootstrap"] = {}
    for name in variants:
        a = df[df.variant == name].set_index("study")
        for ref_name, ref in (("pose_only", po), ("true_geometry", df[df.variant == "true_geometry"].set_index("study"))):
            if name == ref_name:
                continue
            for k in ("auprc", "recall1", "top1_dist"):
                d = (a[k] - ref.loc[a.index, k]).dropna().values
                bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
                out["bootstrap"][f"{name}-{ref_name}:{k}"] = [d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)]
    for k, v in out["bootstrap"].items():
        if k.startswith("pred_all") or k.startswith("population"):
            print(f"  {k:32s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
    sfx = "_full" if args.all_masks else ""
    df.to_csv(os.path.join(args.head, f"geometry_ceiling_per_patient{sfx}.csv"), index=False)
    json.dump(out, open(os.path.join(args.head, f"geometry_ceiling_val{sfx}.json"), "w"), indent=1, default=float)
    print(f"done -> {args.head}")
