"""Step 4 (val only): target-pose belief from realistic sweeps.

    O_1:K -> p(p_L | O_1:K) -> s(P) = E_{p_L ~ p(p_L | O)} [ sigmoid g_pose(P^L(p_L)) ]

Conditions: every val trajectory of an applicable, available patient; prefix O_1:K (K in K_LIST, K <= n_pre),
P_ref = the first pose of the trajectory. Candidates: the patient's GB-blind bank (target_bank_blind, frozen
T_subcostal-LA labels). A candidate's pose relative to P_ref is known (probe tracking); its liver-frame pose
depends on the placement p_L = (R_L, t): P^L = to_liver(R_L, t, rel), lateral re-canonicalized (labels and g are
invariant to the 180 deg flip about the beam).
Frozen parts: placement CFM (--cfm_run, the augmented trajectory flow), pose head g_pose (target_pose_none),
inferred-geometry head (target_pose_inferred_krr, with its z_L -> geometry conditional refit on train).
Variants:
  population   K = 0: M placements drawn from the train trajectories' true placements relative to their first
               pose (the probe position is known, no image)
  point        the posterior mean placement (t mean, projected-mean rotation) of the flow's M samples, one frame
  posterior    M flow samples (main method)
  posterior_geom  M flow samples; z_L ~ Laplace around the occlusion-aware MAP at the posterior-mean placement
               (belief.lm_fit, naive start, LM_ITERS, sigma^2 of evaluate.anatomy), one draw per sample ->
               geometry ~ p(r, a, l | z_L) (+ a train OOF residual tuple) -> inferred-geometry head
  oracle       the true placement (K-independent)
  posterior_popgeom  (--pop_geom, post hoc ablation) the posterior_geom flow samples and inferred-geometry head, but the
               geometry drawn as in population_geom (z ~ N(0, I) -> p(r, a, l | z) + residual tuple), own generator
               (seed 11), so the other variants are unchanged
Local correction: the top-ranked pose's correction to the known T pose nearest in the local-correction norm, split
into probe manoeuvres in its own frame (slide, sweep, depth, rock, fan, rotate; correction()), and reach@k = some
known T pose within k x the frozen per-manoeuvre widths LOCAL_W on every manoeuvre (top-1 and best of top-10);
plus a T sampling-density check (known T poses reaching another known T pose).
Metrics per condition, averaged per patient, then over patients: auroc, auprc, recall@1, recall@10, top-1 distance
to T (skin mm + rotation deg mod flip, nearest known T pose of the dense + blind banks), and the high-belief region
= the smallest candidate set holding 50% of the normalized belief mass: its size (share of candidates) and precision
(share in T). Paired patient bootstrap: posterior - point, posterior - oracle, posterior_geom - posterior, per K.
Output: runs/target_belief_<split>_<ts>/{results.json, per_condition.csv}
Test (once, frozen components only): --split test --reopen_test reads every component from
  configs/target_belief_frozen.json (the CLI defaults are ignored) and writes target_belief_test_<ts>.
External (ENHANCE.PET, once, docs/preregistration.md): --split external --external_once, the same frozen components.
Usage: python -m sonobelief.evaluate.target_belief [--M 32] [--K 0 1 2 4 8]
       python -m sonobelief.evaluate.target_belief --split test --reopen_test
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from sonobelief import paths
from sonobelief.belief import Forward, laplace_sample, lm_fit, z_posterior
from sonobelief.datasets.trajectories import TrajSet, build, traj_frames
from sonobelief.flow import load_flow
from sonobelief.gb_geometry import geometry, state_of
from sonobelief.geometry import canonical, gram_schmidt, placement, proj_mean, relative, to_liver
from sonobelief.geometry_conditional import BLOCKS, oof, predictors
from sonobelief.image_ae import PlaneAE
from sonobelief.metrics import auroc, avg_precision
from sonobelief.nets import DObs
from sonobelief.paths import EMB_RUN, FROZEN
from sonobelief.target_poses import PoseSet, dist_to_T

LM_ITERS = 30
# One local correction (frozen from the train perturbation characterisation, work/perturbation_retention_subcostal.csv,
# 99 train studies): per manoeuvre the magnitude at which the median retention in T falls to 0.5 (linear
# interpolation; in-plane slide still retains 0.65 at the largest tested 20 mm and is capped there).
LOCAL_W = {"slide_mm": 20.0, "sweep_mm": 6.9, "rock_deg": 15.4, "fan_deg": 4.4, "rotate_deg": 15.7}
REACH_K = (1, 2)


def correction(skin_h, R_h, skin_T, R_T):
    """Probe correction from one pose (skin (3,), R (3, 3) columns lateral, beam, normal) to poses T (n, ...), in the
    first pose's probe frame, as absolute manoeuvre magnitudes (n, 6): slide (along lateral), sweep (along the plane
    normal), depth (along the beam; not a manoeuvre, reported only), rock (beam tilt within the image plane), fan
    (beam tilt out of the plane), rotate (twist about the beam after the tilt, mod 180 deg: roll + 180 = same plane)."""
    lat, beam, nrm = R_h[:, 0], R_h[:, 1], R_h[:, 2]
    d = skin_T - skin_h
    b = R_T[:, :, 1] @ R_h                                       # target beam in the probe frame [lat, beam, nrm]
    rock = torch.rad2deg(torch.atan2(b[:, 0], b[:, 1])).abs()
    fan = torch.rad2deg(torch.atan2(b[:, 2], b[:, 1])).abs()
    bt = R_T[:, :, 1]
    k = torch.cross(beam.expand_as(bt), bt, dim=-1)
    sn = k.norm(dim=-1, keepdim=True)
    k = k / sn.clamp(min=1e-8)
    ang = torch.atan2(sn[:, 0], (bt * beam).sum(-1))
    l = lat.expand_as(bt)
    l_sw = l * torch.cos(ang)[:, None] + torch.cross(k, l, dim=-1) * torch.sin(ang)[:, None] + k * (k * l).sum(-1, keepdim=True) * (1 - torch.cos(ang))[:, None]
    tw = torch.rad2deg(torch.atan2((torch.cross(l_sw, R_T[:, :, 0], dim=-1) * bt).sum(-1), (l_sw * R_T[:, :, 0]).sum(-1)))
    rot = ((tw + 90) % 180 - 90).abs()
    return torch.stack([(d @ lat).abs(), (d @ nrm).abs(), (d @ beam).abs(), rock, fan, rot], 1)


def local_norm(c):
    """Correction (n, 6) -> max over manoeuvres of |c_i| / w_i (depth excluded)."""
    w = torch.tensor([LOCAL_W["slide_mm"], LOCAL_W["sweep_mm"], LOCAL_W["rock_deg"], LOCAL_W["fan_deg"],
                      LOCAL_W["rotate_deg"]], device=c.device)
    return (c[:, [0, 1, 3, 4, 5]] / w).max(1).values


def top_correction(D, p, j):
    """Candidate j -> correction to the T pose that is nearest in the local-correction norm, and that norm."""
    if len(D.T_skin[p]) == 0:
        return None, float("nan")
    c = correction(D.skin[j], D.rot[j], D.T_skin[p], D.T_rot[p])
    nrm = local_norm(c)
    i = int(torch.argmin(nrm))
    return c[i], nrm[i].item()


def t_density(D, p):
    """Sampling-density check: share of known T poses that reach ANOTHER known T pose within k local corrections."""
    n = len(D.T_skin[p])
    if n < 2:
        return {}
    best = []
    for i in range(n):
        c = correction(D.T_skin[p][i], D.T_rot[p][i], D.T_skin[p], D.T_rot[p])
        nm = local_norm(c)
        nm[i] = float("inf")
        best.append(nm.min())
    best = torch.stack(best)
    return {f"T_self_reach_k{k}": (best <= k).float().mean().item() for k in REACH_K}


def load_head(run, dim):
    ck = torch.load(os.path.join(run, "best.pt"))
    h = DObs(dim, e_dim=1, width=ck["config"]["width"], depth=ck["config"]["depth"]).cuda().eval()
    h.load_state_dict(ck["model"])
    for p in h.parameters():
        p.requires_grad_(False)
    return h


@torch.no_grad()
def belief(head, rel, R_s, t_s, states=None, chunk=1 << 18):
    """rel (n, 9) candidates in the P_ref frame; placements R_s (M, 3, 3), t_s (M, 3); states (M, d) or None
    -> mean over samples of sigmoid g (n,)."""
    M, n = len(R_s), len(rel)
    out = torch.zeros(n, device="cuda")
    per = max(1, chunk // n)
    for i in range(0, M, per):
        R, t = R_s[i:i + per], t_s[i:i + per]
        m = len(R)
        pl = to_liver(R, t, rel[None].expand(m, -1, -1)).reshape(m * n, 9)
        lat, _ = canonical(pl[:, 3:6], pl[:, 6:9])
        pl = torch.cat([pl[:, :6], lat], 1)
        st = (torch.zeros(m * n, 1, device="cuda") if states is None
              else states[i:i + per].repeat_interleave(n, 0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = head(st, pl[:, None])[:, 0, 0].float()
        out += torch.sigmoid(lo).view(m, n).sum(0)
    return out / M


def metrics(s, y, D, p, cand):
    """Per-condition ranking + region metrics; cand = global indices of the patient's candidates in D."""
    o = torch.argsort(s, descending=True)
    top = cand[o[:10]]
    dtop = dist_to_T(D, p, top)
    w = s[o] / s.sum()
    k50 = int(torch.searchsorted(torch.cumsum(w, 0), torch.tensor(0.5, device="cuda")).item()) + 1
    r = {"top1_dist": dtop[0].item(), "region50_size": k50 / len(s), "region50_precision": y[o[:k50]].mean().item(),
         "region50_recall": (y[o[:k50]].sum() / y.sum().clamp(min=1)).item()}
    c1, n1 = top_correction(D, p, top[0])
    if c1 is not None:
        r.update({f"top1_{k}": v for k, v in zip(("slide_mm", "sweep_mm", "depth_mm", "rock_deg", "fan_deg", "rotate_deg"), c1.tolist())})
        n10 = min(top_correction(D, p, j)[1] for j in top.tolist())
        r.update({"top1_local_norm": n1, **{f"top1_reach_k{k}": float(n1 <= k) for k in REACH_K},
                  **{f"top10_reach_k{k}": float(n10 <= k) for k in REACH_K}})
    if 0 < y.sum() < len(y):
        r.update(auroc=auroc(s, y), auprc=avg_precision(s, y), recall1=float(y[o[0]] > 0),
                 recall10=float(y[o[:10]].max() > 0))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfm_run", default=os.path.join(paths.WEIGHTS, "flow"))
    ap.add_argument("--pose_head", default=os.path.join(paths.WEIGHTS, "pose_head"))
    ap.add_argument("--geom_head", default=os.path.join(paths.WEIGHTS, "geometry_head"))
    ap.add_argument("--sigma_json", default=os.path.join(paths.REPO, "weights/observation_model/sigma2_trajectories.json"))
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--K", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--split", default="val", choices=["val", "test", "external", "external2", "amos", "amos2", "amos_all"])
    ap.add_argument("--reopen_test", action="store_true")
    ap.add_argument("--external_once", action="store_true", help="the single pre-registered external evaluation")
    ap.add_argument("--pop_geom", action="store_true", help="add the posterior_popgeom ablation (post hoc)")
    ap.add_argument("--dump", type=int, nargs="*", default=[], help="trajectory indices whose candidate scores are saved "
                    "(dump_scores.npz; case panels). Does not change any metric")
    args = ap.parse_args()
    if args.split in ("test", "external", "external2", "amos", "amos2", "amos_all"):
        assert args.reopen_test if args.split == "test" else args.external_once, \
            f"the {args.split} split is closed: needs {'--reopen_test' if args.split == 'test' else '--external_once'}"
        fz = json.load(open(os.path.join(paths.CONFIGS, "target_belief_frozen.json")))
        for k in ("cfm_run", "pose_head", "geom_head", "sigma_json", "M", "K"):
            setattr(args, k, os.path.join(paths.REPO, fz[k]) if isinstance(fz[k], str) else fz[k])
    t0 = time.time()
    run_dir = os.path.join(paths.RUNS, f"target_belief_{args.split}{'_popgeom' if args.pop_geom else ''}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    flow, ck = load_flow(args.cfm_run)
    for p in flow.parameters():
        p.requires_grad_(False)
    cfg = ck["config"]
    zl_run = os.path.join(paths.REPO, cfg["zl_run"])
    tr = TrajSet("train", zl_run, cfg["emb"])
    va = TrajSet(args.split, zl_run, cfg["emb"], tr.stats)
    V = va.views
    D = PoseSet(V, ("blind",), "none")
    pose_head, geom_head = load_head(args.pose_head, 1), load_head(args.geom_head, 10)
    g = torch.Generator(device="cuda").manual_seed(7)
    g_pop = torch.Generator(device="cuda").manual_seed(11)
    M = args.M

    # ---- z_L -> geometry conditional (refit exactly as train.geometry_head --axis krr) and z-inference models
    tr_g = PoseSet(tr.views, ("blind",), "none")
    ti = torch.tensor(tr_g.patients, device="cuda")
    Xtr, Ytr = tr.views.zl[ti], geometry(tr.views, ti)
    fns, _ = predictors("krr", Xtr, Ytr)
    res = Ytr - torch.cat([oof(fns[k], Xtr, Ytr[:, b]) for k, b in BLOCKS.items()], 1)
    geo_mean = lambda z: torch.cat([fns[k](Xtr, Ytr[:, b], z) for k, b in BLOCKS.items()], 1)
    del tr_g
    dobs = DObs(V.zl.shape[1]).cuda().eval()
    dobs.load_state_dict(torch.load(os.path.join(paths.REPO, FROZEN["obs_run"], "zL.pt"))["model"])
    ae = PlaneAE(2, 128).cuda().eval()
    ae.load_state_dict(torch.load(os.path.join(EMB_RUN, "best.pt"), weights_only=False)["model"])
    for p in list(dobs.parameters()) + list(ae.parameters()):
        p.requires_grad_(False)
    F = Forward(dobs, ae, tr.stats)
    sig = json.load(open(args.sigma_json))["sigma2"]
    _, obs = traj_frames(args.split, V.studies)
    valid = obs[:, 1] & ~obs[:, 2]
    del obs

    # ---- population placements (K = 0): true placement of train trajectories relative to their first pose
    off = tr.source == 0
    R_pop, t_pop = placement(tr.pose[off.nonzero()[:, 0], 0])

    # ---- conditions: val trajectories of applicable, available patients
    ok_pat = torch.tensor([p in D.patients and D.available[p] for p in range(len(V.studies))], device="cuda")
    cand = {p: (D.pat == p).nonzero()[:, 0] for p in D.patients if D.available[p]}
    rows = []
    oracle_cache = {}
    dumps = {}
    for K in args.K:
        tjK = va.eligible(K)
        tjK = tjK[ok_pat[va.study[tjK]]]
        N = len(tjK)
        if K > 0:
            tok, vmask, _ = build(va, tjK, torch.zeros_like(tjK), torch.full_like(tjK, K))
            with torch.no_grad():
                x = torch.cat([flow.sample(tok[i:i + 512], vmask[i:i + 512], M, 25, g) for i in range(0, N, 512)])
            t_s, R_s = x[..., :3], gram_schmidt(x[..., 3:6], x[..., 6:9])
            t_pt, R_pt = t_s.mean(1), proj_mean(R_s)
            # z_L at the posterior-mean placement (occlusion-aware MAP + Laplace), one draw per placement sample
            rel_o = relative(va.pose[tjK, :K], va.pose[tjK, 0])
            pl = to_liver(R_pt, t_pt, rel_o)
            e = va.e[tjK, :K]
            z0, _, _ = z_posterior(dobs, pl, e, sig["naive"])
            fr = (va.row0[tjK, None] + torch.arange(K, device="cuda")).flatten()
            z_hat, Lc, _, _ = lm_fit(F, z0, pl, e, lambda idx: valid[fr[idx]].float(), sig["occl"], LM_ITERS)
            z_s = laplace_sample(z_hat.repeat_interleave(M, 0), Lc.repeat_interleave(M, 0), g).view(N, M, -1)
        else:
            j = torch.randint(len(R_pop), (N, M), device="cuda", generator=g)
            R_s, t_s = R_pop[j], t_pop[j]
            z_s = torch.randn(N, M, V.zl.shape[1], device="cuda", generator=g)
        jr = torch.randint(len(res), (N, M), device="cuda", generator=g)
        geo_states = state_of(geo_mean(z_s.flatten(0, 1)).view(N, M, -1) + res[jr])
        if args.pop_geom and K > 0:
            z_p = torch.randn(N, M, V.zl.shape[1], device="cuda", generator=g_pop)
            jp = torch.randint(len(res), (N, M), device="cuda", generator=g_pop)
            geo_pop = state_of(geo_mean(z_p.flatten(0, 1)).view(N, M, -1) + res[jp])
        R_true, t_true = placement(va.pose[tjK, 0])
        for c in range(N):
            tj, p = int(tjK[c]), int(va.study[tjK[c]])
            cd = cand[p]
            rel = relative(D.pose[cd], va.pose[tj, 0])
            y = D.y[cd]
            base = {"K": K, "study": V.studies[p], "traj": tj, "n_pos": int(y.sum().item())}
            variants = {}
            if K == 0:
                variants["population"] = belief(pose_head, rel, R_s[c], t_s[c])
                variants["population_geom"] = belief(geom_head, rel, R_s[c], t_s[c], geo_states[c])
            else:
                variants["point"] = belief(pose_head, rel, R_pt[c:c + 1], t_pt[c:c + 1])
                variants["posterior"] = belief(pose_head, rel, R_s[c], t_s[c])
                variants["posterior_geom"] = belief(geom_head, rel, R_s[c], t_s[c], geo_states[c])
                if args.pop_geom:
                    variants["posterior_popgeom"] = belief(geom_head, rel, R_s[c], t_s[c], geo_pop[c])
            if tj not in oracle_cache:
                oracle_cache[tj] = metrics(belief(pose_head, rel, R_true[c:c + 1], t_true[c:c + 1]), y, D, p, cd)
            rows.append({**base, "variant": "oracle", **oracle_cache[tj]})
            for name, s in variants.items():
                rows.append({**base, "variant": name, **metrics(s, y, D, p, cd)})
            if tj in args.dump:
                variants["oracle"] = belief(pose_head, rel, R_true[c:c + 1], t_true[c:c + 1])
                dumps.update({f"K{K}_t{tj}_{n}": s.cpu().numpy() for n, s in variants.items()})
                dumps[f"K{K}_t{tj}_cand"] = cd.cpu().numpy()
        print(f"K={K}: {N} conditions | {(time.time() - t0) / 60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "per_condition.csv"), index=False)

    # ---- aggregate: per patient, then over patients
    df = pd.DataFrame(rows)
    cols = ["auroc", "auprc", "recall1", "recall10", "top1_dist", "region50_size", "region50_precision", "region50_recall",
            "top1_slide_mm", "top1_sweep_mm", "top1_depth_mm", "top1_rock_deg", "top1_fan_deg", "top1_rotate_deg",
            "top1_local_norm"] + [f"{t}_reach_k{k}" for t in ("top1", "top10") for k in REACH_K]
    pp = df.groupby(["K", "variant", "study"])[cols].mean().reset_index()
    summ = pp.groupby(["K", "variant"])[cols].agg(["mean", "median"])
    out = {"table": {f"K{k}:{v}": {f"{c}_{a}": summ.loc[(k, v), (c, a)] for c in cols for a in ("mean", "median")}
                     for k, v in summ.index}}
    print(f"\n{'':22s} {'AUROC':>6s} {'AUPRC':>6s} {'R@1':>5s} {'R@10':>5s} {'top1 d med':>10s} {'reg50 size':>10s} {'reg50 prec':>10s}")
    for (k, v) in summ.index:
        r = summ.loc[(k, v)]
        print(f"K={k} {v:18s} {r[('auroc', 'mean')]:6.3f} {r[('auprc', 'mean')]:6.3f} {r[('recall1', 'mean')]:5.2f} "
              f"{r[('recall10', 'mean')]:5.2f} {r[('top1_dist', 'median')]:10.1f} {r[('region50_size', 'mean')]:10.3f} "
              f"{r[('region50_precision', 'mean')]:10.3f}")
    print(f"\n{'top-1 correction, median':22s} {'slide':>6s} {'sweep':>6s} {'depth':>6s} {'rock':>6s} {'fan':>6s} {'rot':>6s} | "
          f"reach k=1 / k=2 (top-1)   reach k=1 / k=2 (best of top-10)")
    for (k, v) in summ.index:
        r = summ.loc[(k, v)]
        print(f"K={k} {v:18s} " + " ".join(f"{r[(c, 'median')]:6.1f}" for c in ("top1_slide_mm", "top1_sweep_mm", "top1_depth_mm",
              "top1_rock_deg", "top1_fan_deg", "top1_rotate_deg")) + f" | {r[('top1_reach_k1', 'mean')]:.2f} / "
              f"{r[('top1_reach_k2', 'mean')]:.2f}               {r[('top10_reach_k1', 'mean')]:.2f} / {r[('top10_reach_k2', 'mean')]:.2f}")
    dens = pd.DataFrame([t_density(D, p) for p in cand])
    out["local_correction"] = {"widths": LOCAL_W, "k": list(REACH_K),
                               "T_sampling_density": dens.mean().to_dict()}
    print("T sampling density (share of known T poses reaching another within k corrections, mean over patients): "
          + ", ".join(f"{c} {v:.2f}" for c, v in dens.mean().items()))
    rng = np.random.default_rng(0)
    out["bootstrap"] = {}
    print()
    for K in args.K:
        pairs = ([("population", "oracle"), ("population_geom", "population")] if K == 0 else
                 [("posterior", "point"), ("posterior", "oracle"), ("posterior_geom", "posterior")]
                 + ([("posterior_geom", "posterior_popgeom"), ("posterior_popgeom", "posterior")] if args.pop_geom else []))
        for a, b in pairs:
            A = pp[(pp.K == K) & (pp.variant == a)].set_index("study")
            B = pp[(pp.K == K) & (pp.variant == b)].set_index("study")
            for c in ("auprc", "recall1", "recall10", "top1_dist", "region50_precision", "top1_reach_k1", "top1_reach_k2",
                      "top10_reach_k1"):
                d = (A[c] - B.loc[A.index, c]).dropna().values
                bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                out["bootstrap"][f"K{K}:{a}-{b}:{c}"] = [d.mean(), lo, hi]
                print(f"K={K} {a:15s} - {b:10s} {c:18s} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    json.dump(out, open(os.path.join(run_dir, "results.json"), "w"), indent=1, default=float)
    if dumps:
        np.savez(os.path.join(run_dir, "dump_scores.npz"), **dumps)
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
