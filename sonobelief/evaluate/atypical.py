"""Does the anatomical belief move more for atypical patients? (val only; reads evaluate.anatomy per-condition output)

Original hypothesis: observations update a patient-specific latent belief, so that for a patient whose anatomy is
unusual the belief about the unseen GB departs from the population prior towards the truth.

Quantities per condition (val trajectory x K), in the liver frame (r_GB = c_GB - c_L, mm):
  r_bar       population prior mean = mean r_GB of the TRAIN patients
  d           atypicality = ||r_true - r_bar|| = the error of the population prior
  post_err    ||E[r_GB | O_1:K] - r_true||   (E over the belief samples: z_L Laplace draws -> r_GB | z_L)
  gain        d - post_err                    (> 0: the belief is closer to the truth than the prior)
  shift       ||E[r_GB | O] - r_bar||          (how far the belief moved)
  recovered   (E[r_GB | O] - r_bar) . (r_true - r_bar) / d^2   (share of the patient's deviation recovered; 1 = all)
Strata: atypicality tertiles with cut points from the TRAIN distribution of d.
Variants (evaluate.anatomy): oracle_occl (true placement, occlusion-aware likelihood), oracle_true_z (true z_L: the
ceiling of r_GB | z_L), cfm_occl (retrained placement flow + occlusion-aware z_L), and for the GB centroid in the
P_ref frame cfm_occl vs cfm_prior_z (the same placement samples with population shape).
Aggregation: per patient (mean over its trajectories), patient bootstrap 95% CI; Spearman rho across patients.
Output: <run>/atypical_val_summary.json
Usage: python -m sonobelief.evaluate.atypical --run weights/flow
       external: ... --csv atypical_external_per_condition.csv --out atypical_external_summary.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from sonobelief import paths


def boot(x, rng, n=5000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv", default="atypical_val_per_condition.csv")
    ap.add_argument("--out", default="atypical_val_summary.json")
    args = ap.parse_args()
    import torch  # noqa: F401  (Views loads tensors on the GPU)

    from sonobelief.datasets.views import Views
    hyb = json.load(open(os.path.join(paths.CONFIGS, "hybrid_frozen.json")))
    tr = Views("train", os.path.join(paths.REPO, hyb["zl_run"]))
    r_tr = tr.r_gb.cpu().numpy()
    r_bar = r_tr.mean(0)
    d_tr = np.linalg.norm(r_tr - r_bar, axis=1)
    cuts = np.percentile(d_tr, [100 / 3, 200 / 3])
    df = pd.read_csv(os.path.join(args.run, args.csv))
    R = lambda p: df[[f"{p}_{a}" for a in "xyz"]].values
    r_hat, r_true = R("r_hat"), R("r_true")
    dev = r_true - r_bar
    df["d"] = np.linalg.norm(dev, axis=1)
    df["post_err"] = np.linalg.norm(r_hat - r_true, axis=1)
    df["gain"] = df.d - df.post_err
    df["shift"] = np.linalg.norm(r_hat - r_bar, axis=1)
    df["recovered"] = ((r_hat - r_bar) * dev).sum(1) / df.d ** 2
    df["stratum"] = np.digitize(df.d, cuts)                     # 0 typical, 1 middle, 2 atypical (train tertiles)
    rng = np.random.default_rng(0)
    out = {"r_bar_mm": r_bar.tolist(), "tertile_cuts_mm": cuts.tolist(), "rows": {}}
    names = {0: "typical", 1: "middle", 2: "atypical"}
    cols = ["d", "post_err", "gain", "shift", "recovered"]
    print(f"population prior r_bar (train) = {np.round(r_bar, 1).tolist()} mm; atypicality tertile cuts (train) "
          f"{cuts[0]:.1f} / {cuts[1]:.1f} mm")
    for var in ("oracle_occl", "oracle_true_z", "cfm_occl"):
        print(f"\n== {var}: liver-frame r_GB (mm); per stratum mean over patients [95% CI]")
        for K in sorted(df.K.unique()):
            sub = df[(df.variant == var) & (df.K == K)]
            if sub.empty:
                continue
            pp = sub.groupby("study")[cols + ["stratum"]].mean()
            pp["stratum"] = pp.stratum.round().astype(int)
            res = {"rho_d_shift": spearman(pp.d, pp["shift"]), "rho_d_gain": spearman(pp.d, pp.gain)}
            line = []
            for st in (0, 1, 2):
                q = pp[pp.stratum == st]
                res[names[st]] = {"n_patients": int(len(q)), **{c: boot(q[c], rng) for c in cols}}
                g = res[names[st]]
                line.append(f"{names[st]} (n {len(q)}): prior {g['d'][0]:4.1f} -> post {g['post_err'][0]:4.1f} | gain "
                            f"{g['gain'][0]:+4.1f} [{g['gain'][1]:+.1f}, {g['gain'][2]:+.1f}] | shift {g['shift'][0]:4.1f} | "
                            f"recovered {g['recovered'][0]:.2f}")
            a, t = pp[pp.stratum == 2].gain.values, pp[pp.stratum == 0].gain.values
            bs = [rng.choice(a, len(a)).mean() - rng.choice(t, len(t)).mean() for _ in range(5000)]
            res["gain_atypical_minus_typical"] = [float(a.mean() - t.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            out["rows"][f"{var}:K{K}"] = res
            print(f"K={K}  " + "\n      ".join(line))
            print(f"      gain atypical - typical {res['gain_atypical_minus_typical'][0]:+.1f} "
                  f"[{res['gain_atypical_minus_typical'][1]:+.1f}, {res['gain_atypical_minus_typical'][2]:+.1f}] | "
                  f"Spearman(d, shift) {res['rho_d_shift']:.2f}, (d, gain) {res['rho_d_gain']:.2f}")
    # GB centroid in the probe (P_ref) frame with realistic placement: shape belief vs population shape
    print("\n== GB centroid in the P_ref frame, realistic placement: population shape (cfm_prior_z) vs belief (cfm_occl)")
    for K in sorted(df.K.unique()):
        a = df[(df.variant == "cfm_occl") & (df.K == K)].set_index(["study", "traj"]).gb_err_mm
        b = df[(df.variant == "cfm_prior_z") & (df.K == K)].set_index(["study", "traj"])
        if a.empty or b.empty:
            continue
        x = pd.DataFrame({"belief": a, "pop": b.gb_err_mm.loc[a.index],
                          "stratum": df[(df.variant == "cfm_occl") & (df.K == K)].set_index(["study", "traj"]).stratum})
        pp = x.groupby(level=0).mean()
        pp["stratum"] = pp.stratum.round().astype(int)
        res = {}
        line = []
        for st in (0, 1, 2):
            q = pp[pp.stratum == st]
            res[names[st]] = {"pop_shape_err": boot(q["pop"], rng), "belief_err": boot(q.belief, rng),
                              "gain": boot(q["pop"] - q.belief, rng)}
            g = res[names[st]]
            line.append(f"{names[st]}: {g['pop_shape_err'][0]:4.1f} -> {g['belief_err'][0]:4.1f} | gain {g['gain'][0]:+4.1f} "
                        f"[{g['gain'][1]:+.1f}, {g['gain'][2]:+.1f}]")
        out["rows"][f"pref_frame:K{K}"] = res
        print(f"K={K}  " + " | ".join(line))
    json.dump(out, open(os.path.join(args.run, args.out), "w"), indent=1)
    print(f"done -> {os.path.join(args.run, args.out)}")


if __name__ == "__main__":
    main()
