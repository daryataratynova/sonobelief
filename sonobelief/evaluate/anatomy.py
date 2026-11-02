"""Hybrid shape inference p(z_L | p_L, O_1:K) on the source trajectories, with shadow as occlusion (train/val only).

Question: given realistic, correlated pre-target-acquisition views, can the frozen chain still recover the
GB-relevant liver shape?  O_1:K -> p_L -> z_L -> r_GB -> GB centroid in the P_ref frame.

Conditions: every val trajectory with >= K pre-acquisition views, prefix O_1:K (K in K_MAIN), P_ref = first view.
Frozen: E_L (z_L), D_obs (obs_run), E_img / D_img, r_GB | z_L (belief.rgb_conditional).
Likelihoods for e_i (standardized E_img embedding of observed view i):
  occl   occlusion-aware forward model (main): f(z, P, valid) = E_img([sigmoid(D_img(D_obs(z, P)))_liver * valid,
         valid]), valid = FOV & ~shadow of the observed frame (the shadow is observed; the liver behind it is not),
         target e_occl.  MAP by Levenberg-Marquardt (Jacobian by forward-mode AD, 32 tangents per view), then
         Laplace z ~ N(z_hat, H^-1), H = J^T J / sigma^2 + I.
  naive  D_obs(z, P) vs e_occl directly (shadow ignored), MAP + Laplace as belief.z_posterior.
  geom   D_obs(z, P) vs e_geom (geometric view, liver behind the shadow included) - not observable in practice,
         reference only.
  sigma^2 per likelihood = mean squared residual with the true z_L and pose over val pre-acquisition frames (as the
  original train.observation_model sigma^2); the lag-1 residual correlation between consecutive frames is reported (correlated views
  make the product likelihood overconfident).
References on the same conditions: prior_z (z ~ N(0, I), no shape information), true_z (true z_L; r_GB = mu_r(z)).
Placement:
  oracle   true p_L (one placement per condition) - isolates shape recovery
  cfm      the retrained placement flow (--cfm_run), M_p placement samples x n_lap Laplace draws each;
           occl warm-starts LM from the naive MAP of the same placement sample
  old      the frozen hybrid as is: frozen placement CFM (hybrid_frozen.json) + naive D_obs with the old sigma^2
Metrics per K: placement error, z_L error ||mean z - z_true|| (standardized), r_GB error (mm), GB centroid in the
P_ref frame: error of the sample mean, spread, 50/90% coverage (rank of the truth among sample distances).
Output: <cfm_run>/hybrid_traj_val.json
--sigma_scale s: occl likelihood with sigma^2 x s (tempering for correlated consecutive views; sensitivity only).
LM start (--occl_init): the naive MAP of the same placement (default) or z = 0.
Usage: python -m sonobelief.evaluate.anatomy --cfm_run weights/flow [--Mp 8] [--n_lap 8] [--lm_iters 8]
       external (frozen sigma^2): ... --split external --sigma_json runs/cfm_traj_mix_ft_20261023_193721/oracle_s1.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from sonobelief import paths
from sonobelief.belief import Forward, laplace_sample, lm_fit, rgb_conditional, z_posterior
from sonobelief.datasets.trajectories import K_MAIN, TrajSet, build, traj_frames
from sonobelief.flow import load_flow
from sonobelief.geometry import gram_schmidt, placement, proj_mean, relative, rot_err_deg, to_liver
from sonobelief.image_ae import PlaneAE
from sonobelief.nets import DObs
from sonobelief.paths import EMB_RUN, FROZEN


def gb_metrics(t_s, R_s, z_s, r_s, t_true, R_true, z_true, r_true):
    """Samples (N, M, ...) vs the truth (N, ...): placement, z_L, r_GB, GB centroid in the P_ref frame."""
    gb_true = t_true * 100 + (R_true @ r_true[..., None])[..., 0]
    gb_s = t_s * 100 + (R_s @ r_s[..., None])[..., 0]
    gb_hat = gb_s.mean(1)
    d_true = (gb_true - gb_hat).norm(dim=-1)
    d_s = (gb_s - gb_hat[:, None]).norm(dim=-1)
    rank = (d_s < d_true[:, None]).float().mean(1)
    m = lambda x: x.mean().item()
    return {"n": len(t_true), "t_err_mm": m((t_s.mean(1) - t_true).norm(dim=-1) * 100),
            "rot_err_deg": m(rot_err_deg(proj_mean(R_s), R_true)),
            "z_err": m((z_s.mean(1) - z_true).norm(dim=-1)), "z_spread": m((z_s - z_s.mean(1, keepdim=True)).norm(dim=-1)),
            "r_gb_err_mm": m((r_s.mean(1) - r_true).norm(dim=-1)),
            "gb_err_mm": m(d_true), "gb_err_mm_median": d_true.median().item(), "gb_spread_mm": m(d_s),
            "gb_cover50": m((rank <= 0.5).float()), "gb_cover90": m((rank <= 0.9).float()),
            "_per": {"r_gb_err_mm": (r_s.mean(1) - r_true).norm(dim=-1).tolist(), "gb_err_mm": d_true.tolist(),
                     "z_err": (z_s.mean(1) - z_true).norm(dim=-1).tolist(),
                     **{f"r_hat_{a}": r_s.mean(1)[:, i].tolist() for i, a in enumerate("xyz")},
                     **{f"r_true_{a}": r_true[:, i].tolist() for i, a in enumerate("xyz")},
                     **{f"gb_hat_{a}": gb_hat[:, i].tolist() for i, a in enumerate("xyz")},
                     **{f"gb_true_{a}": gb_true[:, i].tolist() for i, a in enumerate("xyz")}}}


def fmt(name, K, r):
    return (f"K={K} {name:13s} t {r['t_err_mm']:5.1f} mm rot {r['rot_err_deg']:5.1f} | z {r['z_err']:.2f} "
            f"(spread {r['z_spread']:.2f}) | r_GB {r['r_gb_err_mm']:5.1f} | GB {r['gb_err_mm']:5.1f} mm "
            f"(med {r['gb_err_mm_median']:5.1f}, spread {r['gb_spread_mm']:5.1f}, cov50/90 {r['gb_cover50']:.2f}/"
            f"{r['gb_cover90']:.2f})" + (f" | E {r['lm_final_energy']:.0f}" if "lm_final_energy" in r else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfm_run", required=True)
    ap.add_argument("--Mp", type=int, default=8, help="placement samples per condition")
    ap.add_argument("--n_lap", type=int, default=8, help="Laplace z draws per placement sample")
    ap.add_argument("--lm_iters", type=int, default=8)
    ap.add_argument("--no_cfm_occl", action="store_true", help="skip the (expensive) cfm x occl variant")
    ap.add_argument("--sigma_scale", type=float, default=1.0, help="occl likelihood temperature (sigma^2 x)")
    ap.add_argument("--occl_init", default="naive", choices=["naive", "zero"], help="LM start for occl")
    ap.add_argument("--K", type=int, nargs="+", default=list(K_MAIN))
    ap.add_argument("--out", default="hybrid_traj_val.json")
    ap.add_argument("--split", default="val", choices=["val", "test", "external", "external2", "amos", "amos2", "amos_all"])
    ap.add_argument("--sigma_json", help="use the frozen sigma^2 of this file (required for external)")
    args = ap.parse_args()
    assert args.split == "val" or args.sigma_json, "external: sigma^2 must be the frozen one (--sigma_json)"
    t0 = time.time()
    flow, ck = load_flow(args.cfm_run)
    cfg = ck["config"]
    zl_run = os.path.join(paths.REPO, cfg["zl_run"])
    tr = TrajSet("train", zl_run, cfg["emb"])
    va = TrajSet(args.split, zl_run, cfg["emb"], tr.stats)
    vg = TrajSet(args.split, zl_run, "e_geom", tr.stats)
    V = va.views
    mu_r, cov_r, oof_err = rgb_conditional(tr.views)
    chol_r = torch.linalg.cholesky(cov_r)
    dobs = DObs(V.zl.shape[1]).cuda().eval()
    dobs.load_state_dict(torch.load(os.path.join(paths.REPO, FROZEN["obs_run"], "zL.pt"))["model"])
    ae = PlaneAE(2, 128).cuda().eval()
    ae.load_state_dict(torch.load(os.path.join(EMB_RUN, "best.pt"), weights_only=False)["model"])
    for p in list(dobs.parameters()) + list(ae.parameters()) + list(flow.parameters()):
        p.requires_grad_(False)
    F = Forward(dobs, ae, tr.stats)
    rows, obs = traj_frames(args.split, V.studies)
    assert (rows["study"][va.row0.cpu().numpy()] == np.array(V.studies)[va.study.cpu().numpy()]).all()
    assert (rows["step"][va.row0.cpu().numpy()] == 0).all()
    valid = obs[:, 1] & ~obs[:, 2]                                          # (n_frames, H, W) FOV & ~shadow
    del obs
    sig_old = json.load(open(os.path.join(paths.REPO, FROZEN["obs_run"], "results.json")))["sigma2"]

    # ---- sigma^2 per likelihood: true z_L, true pose, val pre-acquisition frames
    L = va.pose.shape[1]
    step = torch.arange(L, device="cuda")[None]
    pre = step < va.n_pre[:, None]
    tj, st = pre.nonzero(as_tuple=True)
    z_t = V.zl[va.study[tj]]
    pl_t = va.pose[tj, st]
    fr_t = va.row0[tj] + st
    with torch.no_grad():
        f_occl = F.value(z_t, pl_t, lambda s: valid[fr_t[s]].float())
        f_dobs = torch.cat([dobs(z_t[i:i + 4096], pl_t[i:i + 4096][:, None])[:, 0] for i in range(0, len(z_t), 4096)])
    res_l = {"occl": f_occl - va.e[tj, st], "naive": f_dobs - va.e[tj, st], "geom": f_dobs - vg.e[tj, st]}
    sigma2 = {k: (r ** 2).mean().item() for k, r in res_l.items()}
    if args.sigma_json:            # frozen sigma^2 (the split's own residuals are reported only, as sigma2_split)
        out_sig = {"sigma2_split": sigma2}
        sigma2 = json.load(open(args.sigma_json))["sigma2"]
    else:
        out_sig = {}
    nxt = (st[1:] == st[:-1] + 1) & (tj[1:] == tj[:-1])                    # consecutive frame pairs
    lag1 = {k: torch.nn.functional.cosine_similarity(r[:-1][nxt], r[1:][nxt], dim=1).mean().item() for k, r in res_l.items()}
    out = {"cfm_run": args.cfm_run, "split": args.split, "sigma2": sigma2, **out_sig, "sigma2_old": sig_old, "residual_lag1_cos": lag1,
           "rgb_oof_err_mm": oof_err, "Mp": args.Mp, "n_lap": args.n_lap, "lm_iters": args.lm_iters}
    print(f"sigma^2 ({'frozen, ' + args.sigma_json if args.sigma_json else args.split + ' pre-acquisition frames, true z_L / pose'}): " + ", ".join(f"{k} {v:.3f}" for k, v in sigma2.items())
          + f" (old {sig_old:.3f}) | lag-1 residual cos " + ", ".join(f"{k} {v:.2f}" for k, v in lag1.items())
          + f" | r_GB|z_L oof {oof_err:.1f} mm", flush=True)

    g = torch.Generator(device="cuda").manual_seed(7)
    old_flow, _ = load_flow(FROZEN["cfm_run"])
    for p in old_flow.parameters():
        p.requires_grad_(False)

    def problems(tjK, K, t_s, R_s):
        """Placements (N, M, ...) -> per (condition, sample) liver-frame poses (N*M, K, 9), e_occl / e_geom, frames."""
        N, M = t_s.shape[:2]
        rel = relative(va.pose[tjK, :K], va.pose[tjK, 0])
        pl = to_liver(R_s, t_s, rel[:, None].expand(-1, M, -1, -1)).reshape(N * M, K, 9)
        rep = lambda a: a[:, None].expand(-1, M, *a.shape[1:]).reshape(N * M, *a.shape[1:])
        fr = rep(va.row0[tjK, None] + torch.arange(K, device="cuda"))
        return pl, rep(va.e[tjK, :K]), rep(vg.e[tjK, :K]), fr

    def shape(pl, e, fr, kind, z0=None):
        """-> z_hat, Cholesky factor of the Laplace precision."""
        if kind == "occl":
            fr_f = fr.flatten()
            if z0 is None:
                z0 = torch.zeros(len(pl), V.zl.shape[1], device="cuda")
            z, Lc, En, frac = lm_fit(F, z0, pl, e, lambda idx: valid[fr_f[idx]].float(),
                                     sigma2["occl"] * args.sigma_scale, args.lm_iters)
            return z, Lc, {"lm_improved_frac": frac, "lm_final_energy": En.mean().item()}
        z, Lc, _ = z_posterior(dobs, pl, e, sigma2[kind] if kind != "old" else sig_old)
        return z, Lc, None

    def draws(z_hat, Lc, N, M, n):
        z = laplace_sample(z_hat.repeat_interleave(n, 0), Lc.repeat_interleave(n, 0), g)
        r = mu_r(z) + torch.randn(len(z), 3, device="cuda", generator=g) @ chol_r.T
        return z.view(N, M * n, -1), r.view(N, M * n, 3)

    per_rows = []
    for K in args.K:
        tjK = va.eligible(K)
        N = len(tjK)
        R_true, t_true = placement(va.pose[tjK, 0])
        z_true, r_true = V.zl[va.study[tjK]], V.r_gb[va.study[tjK]]
        truth = (t_true, R_true, z_true, r_true)
        res = {}
        # ---------------- oracle placement
        n_or = args.Mp * args.n_lap
        t_o, R_o = t_true[:, None], R_true[:, None]
        pl, e, eg, fr = problems(tjK, K, t_o, R_o)
        exp = lambda a, M: a.expand(-1, M, *a.shape[2:])
        z_nv = None
        for kind in ("naive", "occl", "geom"):
            z_hat, Lc, frac = shape(pl, eg if kind == "geom" else e, fr, kind,
                                    z0=z_nv if (kind == "occl" and args.occl_init == "naive") else None)
            if kind == "naive":
                z_nv = z_hat
            z_s, r_s = draws(z_hat, Lc, N, 1, n_or)
            res[f"oracle_{kind}"] = gb_metrics(exp(t_o, n_or), exp(R_o, n_or), z_s, r_s, *truth)
            if frac is not None:
                res[f"oracle_{kind}"].update(frac)
        zp = torch.randn(N, n_or, V.zl.shape[1], device="cuda", generator=g)
        rp = mu_r(zp.flatten(0, 1)).view(N, n_or, 3) + torch.randn(N, n_or, 3, device="cuda", generator=g) @ chol_r.T
        res["oracle_prior_z"] = gb_metrics(exp(t_o, n_or), exp(R_o, n_or), zp, rp, *truth)
        rt = (mu_r(z_true)[:, None] + torch.randn(N, n_or, 3, device="cuda", generator=g) @ chol_r.T)
        res["oracle_true_z"] = gb_metrics(exp(t_o, n_or), exp(R_o, n_or), z_true[:, None].expand(-1, n_or, -1), rt, *truth)
        # ---------------- placement flows
        tok, vmask, _ = build(va, tjK, torch.zeros_like(tjK), torch.full_like(tjK, K))
        for fname, fl in (("cfm", flow), ("old", old_flow)):
            with torch.no_grad():
                x = torch.cat([fl.sample(tok[i:i + 512], vmask[i:i + 512], args.Mp, 25, g) for i in range(0, N, 512)])
            t_s, R_s = x[..., :3], gram_schmidt(x[..., 3:6], x[..., 6:9])
            pl, e, eg, fr = problems(tjK, K, t_s, R_s)
            expM = lambda a: a.repeat_interleave(args.n_lap, 1)
            kinds = ("old",) if fname == "old" else ("naive",) + (() if args.no_cfm_occl else ("occl",))
            z_naive = None
            for kind in kinds:
                z_hat, Lc, frac = shape(pl, e, fr, kind, z0=z_naive if args.occl_init == "naive" else None)
                if kind == "naive":
                    z_naive = z_hat
                z_s, r_s = draws(z_hat, Lc, N, args.Mp, args.n_lap)
                name = "old_hybrid" if kind == "old" else f"cfm_{kind}"
                res[name] = gb_metrics(expM(t_s), expM(R_s), z_s, r_s, *truth)
                if frac is not None:
                    res[name].update(frac)
            if fname == "cfm":
                n = args.Mp * args.n_lap
                zp = torch.randn(N, n, V.zl.shape[1], device="cuda", generator=g)
                rp = mu_r(zp.flatten(0, 1)).view(N, n, 3) + torch.randn(N, n, 3, device="cuda", generator=g) @ chol_r.T
                res["cfm_prior_z"] = gb_metrics(expM(t_s), expM(R_s), zp, rp, *truth)
                rt = mu_r(z_true)[:, None] + torch.randn(N, n, 3, device="cuda", generator=g) @ chol_r.T
                res["cfm_true_z"] = gb_metrics(expM(t_s), expM(R_s), z_true[:, None].expand(-1, n, -1), rt, *truth)
        for name, r in res.items():                       # per-condition errors (patient-level bootstrap)
            per = r.pop("_per")
            per_rows.extend({"K": K, "variant": name, "study": V.studies[int(si)], "traj": int(tj), **{k: v[c] for k, v in per.items()}}
                            for c, (si, tj) in enumerate(zip(va.study[tjK].tolist(), tjK.tolist())))
        out[f"K{K}"] = res
        for name, r in res.items():
            print(fmt(name, K, r), flush=True)
        print(f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        json.dump(out, open(os.path.join(args.cfm_run, args.out), "w"), indent=1)
        pd.DataFrame(per_rows).to_csv(os.path.join(args.cfm_run, args.out.replace(".json", "_per_condition.csv")), index=False)
    print(f"done -> {os.path.join(args.cfm_run, args.out)}")
