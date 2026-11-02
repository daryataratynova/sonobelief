"""Step 1c: can liver views recover z_L?  (E_L and E_img frozen)

Pose: p_L composed with P_q^ref is the query pose in the liver-centred CT RAS frame,
    P_q^L = [(apex_q - c_L) / 100 mm, beam_q, lateral_q]       (9 numbers)
so D_obs(p_L, z_L, P_q) is trained as D_obs(z_L, P_q^L); no P_ref is needed.

A. Train D_obs (MLP, latent_models.DObs) on train patients, two models:
     zL    the patient's frozen z_L (standardized with train statistics)
     mean  z_bar_L for everybody: an average liver at the right place
   Val metrics on held-out patients: e_q R^2 and liver Dice of the frozen E_img decoder on
   e_hat (over queries with liver), and for zL also with another patient's z_L.
B. Observability (zL model, val patients, placement p_L assumed known):
   K observed views (liver-aimed windows) -> MAP z_hat_L = argmin
       sum_i ||D_obs(z, P_i) - e_i||^2 / (2 sigma^2) + ||z||^2 / 2
   (sigma^2 = val per-dim MSE of the zL model with the true z_L; Adam from z = 0), then on 64
   held-out views of the same patient: R^2 / Dice, plus |z_hat - z_L| and the GB offset r_GB
   read out by a ridge probe z_L -> r_GB fitted on train patients.
   K = 0 is z_bar_L (the prior mean); "true" is the patient's own z_L.
Output: runs/<name>_<timestamp>/{zL,mean}.pt, results.json
Usage: python -m sonobelief.train.observation_model --zl_run weights/liver_shape
"""
import argparse
import json
import math
import os
import time

import torch

from sonobelief import paths
from sonobelief.datasets.planes import PlaneSet, unpack
from sonobelief.datasets.views import N_PLANES, Views, dice_of
from sonobelief.image_ae import PlaneAE
from sonobelief.nets import DObs
from sonobelief.paths import EMB_RUN
from sonobelief.regression import ridge_pred

CFG = dict(steps=20000, warmup=500, lr=1e-3, wd=0.05, n_pat=64, n_query=64, z_noise=0.1,
           eval_every=200, patience=15, eval_queries=128, seed=0,
           K=(1, 2, 4, 8, 16, 32), trials=8, heldout=64, map_steps=300, map_lr=0.05)


@torch.no_grad()
def evaluate(model, V, zin, query, dec, planes, shuffle=None):
    """zin (S, 32) inputs, query (S, Q) indices -> R^2, mean liver Dice (queries with liver)."""
    model.eval()
    S = torch.arange(len(V), device="cuda")[:, None].expand_as(query)
    z = zin if shuffle is None else zin.roll(shuffle, 0)
    p = model(z, V.pose[S, query])
    t = V.e[S, query]
    mse = ((p - t) ** 2).mean().item()
    r2 = 1 - mse / t.flatten(0, 1).var(0).mean().item()
    m = unpack(planes.packed[(S * N_PLANES + query).flatten()])[:, 0] > 0.5
    has = m.flatten(1).any(1)
    d = dice_of(dec, p.flatten(0, 1), V.stats["e"], m)[has].mean().item()
    model.train()
    return {"mse": mse, "r2": r2, "dice": d}


def train_dobs(mode, tr, va, query_va, dec, planes_va, run_dir, cfg):
    torch.manual_seed(cfg["seed"])
    model = DObs(tr.zl.shape[1]).cuda()
    zbar = torch.zeros_like(tr.zl[:1])  # standardized: the train mean is 0
    ztr = tr.zl if mode == "zL" else zbar.expand_as(tr.zl)
    zva = va.zl if mode == "zL" else zbar.expand_as(va.zl)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    T, W = cfg["steps"], cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, T - W))))
    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    best, since, path = float("inf"), 0, os.path.join(run_dir, f"{mode}.pt")
    t0 = time.time()
    for step in range(1, T + 1):
        s = torch.randint(len(tr), (cfg["n_pat"],), device="cuda", generator=gen)
        q = torch.randint(N_PLANES, (cfg["n_pat"], cfg["n_query"]), device="cuda", generator=gen)
        z = ztr[s]
        if mode == "zL" and cfg["z_noise"]:
            z = z + cfg["z_noise"] * torch.randn(z.shape, device="cuda", generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(z, tr.pose[s[:, None], q])
        loss = ((p.float() - tr.e[s[:, None], q]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == T:
            v = evaluate(model, va, zva, query_va, dec, planes_va)
            print(f"[{mode}] step {step:6d} train mse {loss.item():.3f} | val R2 {v['r2']:.3f} Dice {v['dice']:.3f} "
                  f"| {(time.time() - t0) / 60:.1f} min", flush=True)
            if v["mse"] < best:
                best, since = v["mse"], 0
                torch.save({"model": model.state_dict(), "step": step, "val": v}, path)
            else:
                since += 1
                if since >= cfg["patience"]:
                    print(f"[{mode}] early stop at {step}", flush=True)
                    break
    model.load_state_dict(torch.load(path)["model"])
    return model.eval()


def map_infer(model, pose, e, sigma2, steps, lr):
    """pose (N, K, 9), e (N, K, 128) -> MAP z (N, 32) under N(0, I) prior, from z = 0."""
    z = torch.zeros(len(pose), model.zin.in_features, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        nll = ((model(z, pose) - e) ** 2).sum((1, 2)) / (2 * sigma2) + 0.5 * (z ** 2).sum(1)
        opt.zero_grad()
        nll.sum().backward()
        opt.step()
    return z.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zl_run", required=True)
    ap.add_argument("--name", default="train.observation_model")
    args = ap.parse_args()
    cfg = dict(CFG)
    run_dir = os.path.join(paths.RUNS, f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir)
    json.dump({**cfg, "zl_run": args.zl_run}, open(os.path.join(run_dir, "config.json"), "w"), indent=1)

    tr = Views("train", args.zl_run)
    va = Views("val", args.zl_run, tr.stats)
    planes_va = PlaneSet("val")
    ae = PlaneAE(2, 128).cuda().eval()
    ae.load_state_dict(torch.load(os.path.join(EMB_RUN, "best.pt"), weights_only=False)["model"])
    dec = ae.dec
    g = torch.Generator(device="cuda").manual_seed(123)
    query_va = torch.randint(N_PLANES, (len(va), cfg["eval_queries"]), device="cuda", generator=g)

    res = {"A": {}, "B": {}}
    models = {m: train_dobs(m, tr, va, query_va, dec, planes_va, run_dir, cfg) for m in ("zL", "mean")}
    res["A"]["zL"] = evaluate(models["zL"], va, va.zl, query_va, dec, planes_va)
    res["A"]["zL_shuffled"] = evaluate(models["zL"], va, va.zl, query_va, dec, planes_va, shuffle=1)
    res["A"]["mean_zL"] = evaluate(models["mean"], va, torch.zeros_like(va.zl), query_va, dec, planes_va)
    res["A"]["zL_model_at_zbar"] = evaluate(models["zL"], va, torch.zeros_like(va.zl), query_va, dec, planes_va)
    sigma2 = res["A"]["zL"]["mse"]

    # B: observability. r_GB probe on train z_L (ridge as in regression.ridge_pred).
    rgb_probe = lambda z: ridge_pred(tr.zl, tr.r_gb, z, 0.1)  # standardized z, unpenalized intercept
    gb_err = lambda z, idx: (rgb_probe(z) - va.r_gb[idx]).norm(dim=1)
    model = models["zL"]
    S, T = len(va), cfg["trials"]
    pat = torch.arange(S, device="cuda").repeat_interleave(T)                     # (S*T,)
    aimed = (va.window < 2)
    def draw(k):
        """k observed (liver-aimed) + heldout (any, disjoint) query indices per (patient, trial)."""
        obs, held = [], []
        for p in pat.tolist():
            perm = torch.randperm(N_PLANES, device="cuda", generator=g)
            a = perm[aimed[p, perm]][:k]
            rest = perm[~torch.isin(perm, a)][:cfg["heldout"]]
            obs.append(a); held.append(rest)
        return torch.stack(obs), torch.stack(held)

    def score(z, held):
        P = pat[:, None].expand_as(held)
        with torch.no_grad():
            pr = model(z, va.pose[P, held])
        t = va.e[P, held]
        m = unpack(planes_va.packed[(P * N_PLANES + held).flatten()])[:, 0] > 0.5
        has = m.flatten(1).any(1)
        mse = ((pr - t) ** 2).mean().item()
        return {"r2": 1 - mse / t.flatten(0, 1).var(0).mean().item(),
                "dice": dice_of(dec, pr.flatten(0, 1), va.stats["e"], m)[has].mean().item(),
                "z_err": (z - va.zl[pat]).norm(dim=1).mean().item(),
                "r_gb_err_mm": gb_err(z, pat).mean().item()}

    _, held0 = draw(1)
    res["B"]["K0_prior_mean"] = score(torch.zeros(len(pat), va.zl.shape[1], device="cuda"), held0)
    res["B"]["true_zL"] = score(va.zl[pat], held0)
    for k in cfg["K"]:
        obs, held = draw(k)
        P = pat[:, None].expand_as(obs)
        zhat = map_infer(model, va.pose[P, obs], va.e[P, obs], sigma2, cfg["map_steps"], cfg["map_lr"])
        res["B"][f"K{k}"] = score(zhat, held)
        print(f"K={k:2d}: " + "  ".join(f"{a} {b:.3f}" for a, b in res["B"][f"K{k}"].items()), flush=True)
    res["B"]["note"] = "placement p_L known; observed views from windows 0/1; held-out views any window"
    res["sigma2"] = sigma2
    json.dump(res, open(os.path.join(run_dir, "results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"done -> {run_dir}")
