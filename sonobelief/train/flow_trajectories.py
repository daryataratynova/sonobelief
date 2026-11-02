"""Placement CFM p(p_L | O_1:K) retrained on the realistic source trajectories (train/val only).

Same model as the frozen hybrid's placement flow (flow.Posterior: Set Transformer + CFM over [t, R_L 6D]);
only the training data changes:
  conditions  O_1:K = the first K pre-target-acquisition observations of a frozen source trajectory
              (work/traj_emb; poses with canonical roll), K in K_SET; K = 16 only for
              trajectories with >= 16 pre-acquisition views. P_ref = the first view of the condition.
  tokens      [standardized frozen E_img embedding (cfg "emb": e_occl = shadow as occlusion, or e_geom), Delta P (9)]
  target      liver placement relative to P_ref (t / 100 mm, R_L 6D), from the liver centroid (work/anatomy)
cfg "windows": false = prefixes only; true = also any contiguous window of the pre-acquisition sequence
              (augmentation; validation and evaluation always use prefixes).
cfg "mix_planes": share of each batch drawn from the original random liver-aimed plane conditions (train_cfm.draw /
              build: frozen geometric embeddings of work/planes, K from K_SET), to keep multi-view aggregation learnt
              from many distinct view sets; 0 = trajectories only. Validation / evaluation use trajectories only.
cfg "aug": true = the train trajectory pool also contains the augmentation trajectories (work/
              traj_emb/train_aug.npz: 64 more per train patient, extra seeds of the frozen generator; train only).
              The fixed-train monitor uses the official trajectories only.
cfg "init": a placement CFM run to fine-tune from (keeps its state standardization), or null (from scratch).
Model selection: val CFM loss on fixed prefix conditions, mean over K_MAIN.
Evaluation (val prefixes, best checkpoint): placement of the point estimate (sample mean / projected mean
rotation), spread and 90% coverage, per K; the same for a baseline flow (the frozen hybrid's CFM, applied as is).
Output: runs/<name>_<timestamp>/{config.json, log.csv, best.pt, eval_val.json}
Usage: python -m sonobelief.train.flow_trajectories --config configs/flow.json
       python -m sonobelief.train.flow_trajectories --eval_only runs/<run> [--baseline weights/flow_planes]
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from sonobelief import paths
from sonobelief.datasets.trajectories import K_MAIN, TrajSet, build, draw
from sonobelief.datasets.views import build as build_planes
from sonobelief.datasets.views import draw as draw_planes
from sonobelief.flow import KMAX, Posterior, load_flow
from sonobelief.geometry import gram_schmidt, placement, proj_mean, rot_err_deg

K_SET = (1, 2, 4, 8, 16)


assert max(K_SET) <= KMAX


def fixed_set(D, seed, reps=4):
    """Every eligible trajectory x K (prefixes), reps noise draws each."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    tj, k = [], []
    for K in K_SET:
        el = D.eligible(K)
        el = el[D.source[el] == 0]                  # official trajectories only (no augmentation)
        tj.append(el.repeat(reps)); k.append(torch.full((len(el) * reps,), K, device="cuda"))
    tj, k = torch.cat(tj), torch.cat(k)
    tok, valid, w = build(D, tj, torch.zeros_like(tj), k)
    return dict(tok=tok, valid=valid, w=w, k=k, x0=torch.randn(w.shape, device="cuda", generator=g),
                tau=torch.rand(len(w), device="cuda", generator=g))


@torch.no_grad()
def fixed_loss(model, S, chunk=4096):
    model.eval()
    out = []
    for i in range(0, len(S["w"]), chunk):
        sl = slice(i, i + chunk)
        x1 = (S["w"][sl] - model.w_mu) / model.w_sd
        out.append(model.loss(S["tok"][sl], S["valid"][sl], x1, S["x0"][sl], S["tau"][sl]).mean(1))
    model.train()
    l = torch.cat(out)
    per_k = {K: l[S["k"] == K].mean().item() for K in K_SET}
    return float(np.mean([per_k[K] for K in K_MAIN])), per_k


def train(cfg, run_dir, tr, va):
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    model = Posterior(cfg, 9).cuda()
    if cfg.get("init"):
        ck = torch.load(os.path.join(paths.REPO, cfg["init"], "best.pt"), weights_only=False)
        model.load_state_dict(ck["model"])                 # incl. its state standardization
        print(f"fine-tuning from {cfg['init']} (step {ck['step']})", flush=True)
    else:
        tj, st, k = draw(tr, 20000, K_SET, cfg["windows"], gen)
        w = build(tr, tj, st, k)[2]
        model.w_mu.copy_(w.mean(0)); model.w_sd.copy_(w.std(0))
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M | train trajectories {len(tr)} | "
          f"val {len(va)}", flush=True)
    S_va, S_tr = fixed_set(va, 100), fixed_set(tr, 101, reps=1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    T, W = cfg["steps"], cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, T - W))))
    log = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["step", "train_loss", "fixed_train", "val_main"] + [f"val_K{K}" for K in K_SET])
    best, since, t0, run = float("inf"), 0, time.time(), 0.0
    for step in range(1, T + 1):
        n_pl = int(round(cfg.get("mix_planes", 0.0) * cfg["batch"]))
        tj, st, k = draw(tr, cfg["batch"] - n_pl, K_SET, cfg["windows"], gen)
        tok, valid, w = build(tr, tj, st, k)
        if n_pl:   # original random-plane conditions of train patients (same token layout, KMAX)
            pat = torch.randint(len(tr.views), (n_pl,), device="cuda", generator=gen)
            kp = torch.tensor(K_SET, device="cuda")[torch.randint(len(K_SET), (n_pl,), device="cuda", generator=gen)]
            tp, vp, wp = build_planes(tr.views, pat, draw_planes(tr.views, pat, gen), kp)
            tok, valid, w = torch.cat([tok, tp]), torch.cat([valid, vp]), torch.cat([w, wp[:, :9]])
        x1 = (w - model.w_mu) / model.w_sd
        x0 = torch.randn(x1.shape, device="cuda", generator=gen)
        tau = torch.rand(len(x1), device="cuda", generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(tok, valid, x1, x0, tau).float().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % cfg["eval_every"] == 0 or step == T:
            vl, vk = fixed_loss(model, S_va)
            tl, _ = fixed_loss(model, S_tr)
            wr.writerow([step, run, tl, vl] + [vk[K] for K in K_SET]); log.flush()
            print(f"step {step:6d} | train {run:.4f} fixed-train {tl:.4f} | val {vl:.4f} | "
                  + " ".join(f"K{K} {vk[K]:.3f}" for K in K_SET) + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
            if vl < best:
                best, since = vl, 0
                torch.save({"model": model.state_dict(), "config": cfg, "step": step, "val_loss": vl},
                           os.path.join(run_dir, "best.pt"))
            else:
                since += 1
                if since >= cfg["patience"]:
                    print(f"early stop at {step}", flush=True)
                    break
    log.close()


@torch.no_grad()
def evaluate(model, D, M=64, steps=25, seed=7):
    """Placement belief on prefix conditions of every eligible trajectory, per K."""
    model.eval()
    g = torch.Generator(device="cuda").manual_seed(seed)
    res = {}
    for K in K_SET:
        tj = D.eligible(K)
        tok, valid, w = build(D, tj, torch.zeros_like(tj), torch.full_like(tj, K))
        x = torch.cat([model.sample(tok[i:i + 512], valid[i:i + 512], M, steps, g) for i in range(0, len(tj), 512)])
        t_s, R_s = x[..., :3], gram_schmidt(x[..., 3:6], x[..., 6:9])
        R_true, t_true = placement(D.pose[tj, 0])
        t_hat, R_hat = t_s.mean(1), proj_mean(R_s)
        terr, rerr = (t_hat - t_true).norm(dim=-1) * 100, rot_err_deg(R_hat, R_true)
        t_d = (t_s - t_hat[:, None]).norm(dim=-1)
        rot_s = rot_err_deg(R_s, R_hat[:, None].expand_as(R_s))
        res[f"K{K}"] = {"n": len(tj), "t_err_mm": terr.mean().item(), "t_err_mm_median": terr.median().item(),
                        "t_spread_mm": (t_d * 100).mean().item(),
                        "t_cover90": ((t_d < (t_hat - t_true).norm(dim=-1)[:, None]).float().mean(1) <= 0.9).float().mean().item(),
                        "rot_err_deg": rerr.mean().item(), "rot_err_deg_median": rerr.median().item(),
                        "rot_err_gt30_frac": (rerr > 30).float().mean().item(), "rot_spread_deg": rot_s.mean().item(),
                        "rot_cover90": ((rot_s < rerr[:, None]).float().mean(1) <= 0.9).float().mean().item()}
    return res


def fmt(name, res):
    return "\n".join(f"{name:12s} K={k[1:]:>2s} (n {r['n']:4d}): t {r['t_err_mm']:5.1f} mm (med {r['t_err_mm_median']:5.1f}, "
                     f"spread {r['t_spread_mm']:5.1f}, cov90 {r['t_cover90']:.2f}) | rot {r['rot_err_deg']:5.1f} deg "
                     f"(med {r['rot_err_deg_median']:5.1f}, >30 {r['rot_err_gt30_frac']:.2f}, spread {r['rot_spread_deg']:5.1f}, "
                     f"cov90 {r['rot_cover90']:.2f})" for k, r in res.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--eval_only")
    ap.add_argument("--baseline", default=json.load(open(os.path.join(paths.CONFIGS, "hybrid_frozen.json")))["cfm_run"])
    args = ap.parse_args()
    if args.eval_only:
        run_dir = args.eval_only
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
    else:
        cfg = json.load(open(args.config))
        run_dir = os.path.join(paths.RUNS, f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir)
        json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    zl_run = os.path.join(paths.REPO, cfg["zl_run"])
    tr = TrajSet("train", zl_run, cfg["emb"], aug=cfg.get("aug", False))
    va = TrajSet("val", zl_run, cfg["emb"], tr.stats)
    if not args.eval_only:
        train(cfg, run_dir, tr, va)
    model, ck = load_flow(run_dir)
    print(f"best step {ck['step']} val loss {ck['val_loss']:.4f}", flush=True)
    res = {"model": evaluate(model, va), "best_step": ck["step"]}
    print(fmt("retrained", res["model"]), flush=True)
    if args.baseline:
        base, _ = load_flow(args.baseline)
        res["baseline_frozen_cfm"] = evaluate(base, va)
        res["baseline_run"] = args.baseline
        print(fmt("frozen CFM", res["baseline_frozen_cfm"]), flush=True)
    json.dump(res, open(os.path.join(run_dir, "eval_val.json"), "w"), indent=1)
    print(f"done -> {run_dir}")
