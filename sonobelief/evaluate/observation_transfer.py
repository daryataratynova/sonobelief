"""Does the frozen observation model transfer to the trajectory views? (train/val only; no retraining)

Frozen: E_img / D_img (weights/image_encoder), D_obs (train.observation_model run of configs/hybrid_frozen.json), E_L (z_L).
For every trajectory frame (work/traj_emb/<split>.npz, canonical roll) and, as the reference, the original
liver-aimed random planes (windows 0/1) of the same patients:
  AE recon Dice     liver Dice of D_img(E_img(x)) vs x                  (does E_img still encode the views?)
  D_obs R^2         1 - MSE(D_obs(z_L, P^L), e) / var_ref               (true z_L, true pose; var_ref = per-dim
                    variance of the reference-plane embeddings of the split, the same denominator for all rows)
  D_obs Dice        liver Dice of D_img(D_obs(z_L, P^L)) vs the observed liver (frames with liver)
Occluded variant (shadow as occlusion): target e_occl; Dice restricted to the valid FOV (FOV & ~shadow), i.e.
  D_obs predicts the geometric view and the shadow mask removes what cannot be seen.
Breakdowns: shadow share, mirrored (flipped) frames, pre-acquisition vs censored frames.
Output: work/dobs_on_traj.json
Usage: python -m sonobelief.evaluate.observation_transfer
"""
import json
import os

import numpy as np
import torch

from sonobelief import paths
from sonobelief.datasets.planes import PlaneSet, unpack
from sonobelief.datasets.trajectories import traj_frames
from sonobelief.datasets.views import N_PLANES, Views
from sonobelief.image_ae import PlaneAE
from sonobelief.nets import DObs
from sonobelief.paths import ANATOMY, EMB_RUN, FROZEN


def dice(pred, m):
    inter = (pred & m).flatten(1).sum(1).float()
    return 2 * inter / (pred.flatten(1).sum(1) + m.flatten(1).sum(1)).float().clamp(min=1)


@torch.no_grad()
def decode_liver(dec, e_raw, chunk=2048):
    out = []
    for i in range(0, len(e_raw), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(dec(e_raw[i:i + chunk])[:, 0].float() > 0)
    return torch.cat(out)


@torch.no_grad()
def encode(enc, x, chunk=2048):
    out = []
    for i in range(0, len(x), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(enc(x[i:i + chunk]).float())
    return torch.cat(out)


def main():
    out = {}
    zl_run = FROZEN["zl_run"]
    tr = Views("train", os.path.join(paths.REPO, zl_run))
    ae = PlaneAE(2, 128).cuda().eval()
    ae.load_state_dict(torch.load(os.path.join(EMB_RUN, "best.pt"), weights_only=False)["model"])
    dobs = DObs(tr.zl.shape[1]).cuda().eval()
    dobs.load_state_dict(torch.load(os.path.join(paths.REPO, FROZEN["obs_run"], "zL.pt"))["model"])
    mu, sd = tr.stats["e"]
    for split, n_studies in (("val", None), ("train", 60)):
        V = tr if split == "train" else Views(split, os.path.join(paths.REPO, zl_run), tr.stats)
        studies = V.studies[:n_studies] if n_studies else V.studies
        idx = {s: i for i, s in enumerate(V.studies)}
        planes = PlaneSet(split)
        # ---- reference: original liver-aimed planes of the same patients
        pat = torch.tensor([idx[s] for s in studies], device="cuda")
        q = torch.arange(N_PLANES, device="cuda")
        P, Q = pat[:, None].expand(-1, N_PLANES), q[None].expand(len(pat), -1)
        aimed = V.window[P, Q] < 2
        P, Q = P[aimed], Q[aimed]
        e_ref = V.e[P, Q]
        var_ref = e_ref.var(0).mean().item()
        x_ref = unpack(planes.packed[P * N_PLANES + Q])
        m_ref = x_ref[:, 0] > 0.5
        with torch.no_grad():
            pr = dobs(V.zl[P], V.pose[P, Q][:, None])[:, 0]
        has = m_ref.flatten(1).any(1)
        rec = decode_liver(ae.dec, encode(ae.enc, x_ref))
        ref = {"n": int(len(P)), "ae_recon_dice": dice(rec, m_ref)[has].mean().item(),
               "dobs_r2": 1 - ((pr - e_ref) ** 2).mean().item() / var_ref,
               "dobs_dice": dice(decode_liver(ae.dec, pr * sd + mu), m_ref)[has].mean().item()}
        del x_ref, rec
        # ---- trajectory frames
        rows, obs = traj_frames(split, studies)
        pi = torch.tensor([idx[s] for s in rows["study"]], device="cuda")
        lc = torch.from_numpy(np.stack([np.load(os.path.join(ANATOMY, f"{s}.npz"))["liver_c"] for s in V.studies])).cuda()
        t = lambda k: torch.from_numpy(rows[k]).float().cuda()
        pose = torch.cat([(t("apex") - lc[pi]) / 100, t("beam"), t("lateral")], -1)
        with torch.no_grad():
            pr = torch.cat([dobs(V.zl[pi[i:i + 4096]], pose[i:i + 4096][:, None])[:, 0] for i in range(0, len(pi), 4096)])
        e_g = (t("e_geom") - mu) / sd
        e_o = (t("e_occl") - mu) / sd
        liver, fov, sh = obs[:, 0], obs[:, 1], obs[:, 2]
        valid = fov & ~sh
        pred_liver = decode_liver(ae.dec, pr * sd + mu)
        rec_g = decode_liver(ae.dec, t("e_geom"))
        rec_o = decode_liver(ae.dec, t("e_occl"))
        has_g, has_o = liver.flatten(1).any(1), (liver & valid).flatten(1).any(1)
        per = {"r2_geom": 1 - ((pr - e_g) ** 2).mean(1) / var_ref, "r2_occl": 1 - ((pr - e_o) ** 2).mean(1) / var_ref,
               "dice_geom": dice(pred_liver, liver), "dice_occl_valid": dice(pred_liver & valid, liver & valid),
               "ae_geom": dice(rec_g, liver), "ae_occl": dice(rec_o, liver & valid)}
        shf = torch.from_numpy(rows["shadow_frac"]).cuda()
        pre = torch.from_numpy((rows["t_gb"] < 0) | (rows["step"] < rows["t_gb"])).cuda()
        flipped = torch.from_numpy(rows["flipped"]).cuda()

        def summ(sel):
            g, o = sel & has_g, sel & has_o
            return {"n": int(sel.sum()), "ae_recon_dice_geom": per["ae_geom"][g].mean().item(),
                    "ae_recon_dice_occl": per["ae_occl"][o].mean().item(),
                    "dobs_r2_geom": per["r2_geom"][sel].mean().item(), "dobs_dice_geom": per["dice_geom"][g].mean().item(),
                    "dobs_r2_occl": per["r2_occl"][sel].mean().item(), "dobs_dice_occl_valid": per["dice_occl_valid"][o].mean().item()}

        allf = torch.ones_like(pre)
        res = {"reference_planes": ref, "trajectory_all": summ(allf), "pre_acquisition": summ(pre), "censored": summ(~pre),
               "flipped": summ(flipped), "not_flipped": summ(~flipped)}
        for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1.01)):
            res[f"shadow_{lo}-{min(hi, 1)}"] = summ((shf >= lo) & (shf < hi))
        out[split] = {"studies": len(studies), **res}
        print(f"\n[{split}] {len(studies)} studies", flush=True)
        print(f"  {'':22s} {'n':>6s} {'AE geom':>8s} {'AE occl':>8s} {'R2 geom':>8s} {'Dice geom':>9s} {'R2 occl':>8s} {'Dice occl':>9s}")
        print(f"  {'reference planes':22s} {ref['n']:6d} {ref['ae_recon_dice']:8.3f} {'':8s} {ref['dobs_r2']:8.3f} {ref['dobs_dice']:9.3f}")
        for k, r in res.items():
            if k == "reference_planes":
                continue
            print(f"  {k:22s} {r['n']:6d} {r['ae_recon_dice_geom']:8.3f} {r['ae_recon_dice_occl']:8.3f} {r['dobs_r2_geom']:8.3f} "
                  f"{r['dobs_dice_geom']:9.3f} {r['dobs_r2_occl']:8.3f} {r['dobs_dice_occl_valid']:9.3f}", flush=True)
    json.dump(out, open(os.path.join(paths.WORK, "dobs_on_traj.json"), "w"), indent=1)
