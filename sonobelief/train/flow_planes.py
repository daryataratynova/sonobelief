"""Step 2: amortised posterior p(w | O_1:K) over w = [p_L, z_L, r_GB] (Set Transformer + CFM).

Observations: K liver-aimed views of one patient; the first one defines P_ref. Each view is a token
    e_i = [E_img(M_i) (standardized, 128), Delta P_i = pose in the P_ref frame (apex / 100 mm, beam, lateral)]
C_K = S(e_1..e_K): learned null token + K tokens -> self-attention blocks -> attention pooling (PMA)
      + an embedding of K.  K = 0 sees only the null token, i.e. it learns the population prior.
State (44-d, standardized with train statistics):
    t (liver centroid in P_ref, /100 mm), R_L (liver RAS -> P_ref) as its first two columns (6D),
    z_L (frozen E_L, standardized as in train.observation_model), r_GB = c_GB - c_L (CT RAS, /100 mm).
CFM, linear path: x_tau = (1 - tau) x0 + tau x1, x0 ~ N(0, I), target v = x1 - x0;
velocity v_theta(x_tau, tau, C_K) = residual MLP with adaLN-zero conditioning on (tau, C_K).
K is drawn from K_SET per example; the observed views are drawn without replacement from the liver-aimed
windows (0/1), the reference among them. For K = 0 a reference view is still drawn: it defines the frame
of the ground truth but is not shown.

Model selection: val CFM loss on a fixed set (patients x K x draws, fixed noise and tau).
Final evaluation (val, best checkpoint; M samples per condition, midpoint ODE):
  per K: placement errors of the point estimate (sample mean; rotation = projected mean), GB centroid
  in the probe frame c_GB = t + R_L r_GB (point error, expected sample error, spread, coverage of the
  50 / 90% balls around the point), r_GB and z_L errors, held-out view Dice of D_obs (train.observation_model) per sample,
  and the r_GB error with another patient's views (shuffled control).
Output: runs/<name>_<timestamp>/{config.json, log.csv, best.pt, eval.json}
cfg["state"] = "placement": the flow models only p(p_L | O) (w = [t, R_L 6D]); used by belief.py.
Usage: python -m sonobelief.train.flow_planes --config configs/flow_planes.json
"""
import argparse
import csv
import json
import math
import os
import time

import torch

from sonobelief import paths
from sonobelief.datasets.views import Views, build, draw
from sonobelief.flow import K_SET, Posterior

GROUPS = {"t": slice(0, 3), "R": slice(3, 9), "z": slice(9, 41), "r": slice(41, 44)}   # parts of w
STATE_DIM = {"full": 44, "placement": 9}   # cfg["state"]: placement = [t, R_L 6D] only (hybrid, Step 2b)


def fixed_set(V, reps, seed):
    """Fixed (patients x K x reps) examples with fixed noise / tau, for comparable losses."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pat = torch.arange(len(V), device="cuda").repeat(len(K_SET) * reps)
    k = torch.tensor(K_SET, device="cuda").repeat_interleave(len(V)).repeat(reps)
    tok, valid, w = build(V, pat, draw(V, pat, g), k)
    return dict(tok=tok, valid=valid, w=w, k=k, x0=torch.randn(w.shape, device="cuda", generator=g),
                tau=torch.rand(len(w), device="cuda", generator=g))


@torch.no_grad()
def fixed_loss(model, S, chunk=4096):
    model.eval()
    out = []
    for i in range(0, len(S["w"]), chunk):
        sl = slice(i, i + chunk)
        x1 = (S["w"][sl, :len(model.w_mu)] - model.w_mu) / model.w_sd
        out.append(model.loss(S["tok"][sl], S["valid"][sl], x1, S["x0"][sl, :x1.shape[1]], S["tau"][sl]))
    model.train()
    l = torch.cat(out)
    groups = {g: l[:, sl].mean().item() for g, sl in GROUPS.items() if l[:, sl].numel()}
    l = l.mean(1)
    return l.mean().item(), {k: l[S["k"] == k].mean().item() for k in K_SET}, groups


def train(cfg, run_dir, tr, va):
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    pat = torch.randint(len(tr), (20000,), device="cuda", generator=gen)
    D = STATE_DIM[cfg.get("state", "full")]
    groups = [g for g, sl in GROUPS.items() if sl.stop <= D]
    w_all = build(tr, pat, draw(tr, pat, gen), torch.zeros_like(pat))[2][:, :D]
    model = Posterior(cfg, w_all.shape[1]).cuda()
    model.w_mu.copy_(w_all.mean(0)); model.w_sd.copy_(w_all.std(0))
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    S_va, S_tr = fixed_set(va, 8, 100), fixed_set(tr, 1, 101)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    T, W = cfg["steps"], cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, T - W))))
    k_set = torch.tensor(K_SET, device="cuda")
    log = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["step", "train_loss", "fixed_train", "val"] + [f"val_K{k}" for k in K_SET]
                + [f"train_{g}" for g in groups] + [f"val_{g}" for g in groups])
    best, since, t0, run = float("inf"), 0, time.time(), 0.0
    for step in range(1, T + 1):
        pat = torch.randint(len(tr), (cfg["batch"],), device="cuda", generator=gen)
        k = k_set[torch.randint(len(K_SET), (cfg["batch"],), device="cuda", generator=gen)]
        tok, valid, w = build(tr, pat, draw(tr, pat, gen), k)
        x1 = (w[:, :D] - model.w_mu) / model.w_sd
        if cfg["target_noise"] and D > 9:   # smooth the 302 shape targets (z_L, r_GB); placement stays exact
            x1 = x1.clone()
            x1[:, 9:] += cfg["target_noise"] * torch.randn(x1[:, 9:].shape, device="cuda", generator=gen)
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
            vl, vk, vg = fixed_loss(model, S_va)
            tl, _, tg = fixed_loss(model, S_tr)
            wr.writerow([step, run, tl, vl] + [vk[k] for k in K_SET] + list(tg.values()) + list(vg.values())); log.flush()
            print(f"step {step:6d} | train {run:.4f} fixed-train {tl:.4f} | val {vl:.4f} | "
                  + " ".join(f"K{k} {vk[k]:.3f}" for k in K_SET) + " | train/val "
                  + " ".join(f"{g} {tg[g]:.2f}/{vg[g]:.2f}" for g in groups)
                  + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(paths.CONFIGS, "flow_planes.json"))
    ap.add_argument("--steps", type=int)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.steps:
        cfg["steps"] = args.steps
    run_dir = os.path.join(paths.RUNS, f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    tr = Views("train", os.path.join(paths.REPO, cfg["zl_run"]))
    va = Views("val", os.path.join(paths.REPO, cfg["zl_run"]), tr.stats)
    train(cfg, run_dir, tr, va)
    ck = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)
    print(f"best step {ck['step']} val loss {ck['val_loss']:.4f}", flush=True)
    print(f"done -> {run_dir} (fine-tune on trajectories with python -m sonobelief.train.flow_trajectories)")
