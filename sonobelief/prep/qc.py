"""Check every study in dataset.csv and build a visual QC gallery.

Automatic checks (re-done from the masks, independent of the coverage CSV):
  liver_ok   liver non-empty, not touching any face of the volume, and not touching the
             air outside the body in >= FOV_TOUCH_MAX voxels (catches cuts by the round
             reconstruction field of view or by in-plane crops, which are filled with air)
  gb_ok      gallbladder >= GB_MIN_ML and not touching any face of the volume
  abd_ok     T12..L5 all present, and T12 top / L5 bottom not touching the scan edge

For each study a coronal preview is written to work/qc/<study>.png: CT slice through the
gallbladder, liver (red) and gallbladder (green) outlines projected over the volume,
and dashed lines at the top of T12 and the bottom of L5 (the abdominal range).

Outputs: work/qc/qc_results.csv, work/qc/*.png, work/qc/index.html (open in a browser; you can
mark studies as rejected there and export the list as CSV).
"""
import argparse
import html
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from sonobelief import paths

matplotlib.use("Agg")

DATA_OUT = paths.WORK
INTER = os.path.join(DATA_OUT, "intermediate")
QC = paths.QC
GB_MIN_ML = 1.0  # smaller gallbladder masks are label noise (1-2 voxels, specks at the liver hilum)
FOV_TOUCH_MAX = 15  # liver-boundary voxels allowed to touch outside-body air (stray label specks)
ABD_VERT = ["vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]


def load(path):
    img = nib.as_closest_canonical(nib.load(path))  # RAS: axis 1 = A-P, axis 2 = I-S
    return np.asanyarray(img.dataobj), img.header.get_zooms()[:3]


def z_range(m):
    zs = np.where(m.any(axis=(0, 1)))[0]
    return (int(zs[0]), int(zs[-1])) if len(zs) else None


def largest_cc(m):
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    return lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)


def touches_face(m):
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or m[:, :, 0].any() or m[:, :, -1].any())


def body_mask(ct):
    b = ct > -500
    b = np.stack([ndimage.binary_fill_holes(b[:, :, z]) for z in range(b.shape[2])], axis=2)
    lab, n = ndimage.label(b)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else b


def fov_touch(organ, ct):
    """Voxels just outside the organ (in-plane) that are air outside the body."""
    st = np.zeros((3, 3, 1), bool)
    st[:, :, 0] = True
    ring = ndimage.binary_dilation(organ, st) & ~organ
    return int((ring & ~body_mask(ct)).sum())


def check(row):
    sid, seg = row["study"], row["seg_dir"]
    ct, sp = load(row["ct"])
    nz = ct.shape[2]
    masks = {o: load(os.path.join(seg, f"{o}.nii.gz"))[0] > 0 for o in ["liver", "gallbladder"] + ABD_VERT}
    for v in ABD_VERT:  # vertebra labels can carry stray voxels far from the bone
        masks[v] = largest_cc(masks[v])
    zr = {o: z_range(m) for o, m in masks.items()}
    lv, gb = masks["liver"], masks["gallbladder"]
    vox_ml = float(np.prod(sp)) / 1000

    res = {"study": sid, "liver_ml": lv.sum() * vox_ml, "gb_ml": gb.sum() * vox_ml}
    res["liver_face"], res["gb_face"] = touches_face(lv), touches_face(gb)
    res["liver_fov_touch"] = fov_touch(lv, ct) if lv.any() else 0
    res["liver_ok"] = bool(lv.any() and not res["liver_face"] and res["liver_fov_touch"] < FOV_TOUCH_MAX)
    res["gb_ok"] = bool(res["gb_ml"] >= GB_MIN_ML and not res["gb_face"])
    vert_present = all(zr[v] is not None for v in ABD_VERT)
    res["abd_ok"] = vert_present and zr["vertebrae_T12"][1] < nz - 1 and zr["vertebrae_L5"][0] > 0
    res["abd_top_z"] = zr["vertebrae_T12"][1] if zr["vertebrae_T12"] else None
    res["abd_bottom_z"] = zr["vertebrae_L5"][0] if zr["vertebrae_L5"] else None
    res["abd_len_mm"] = (res["abd_top_z"] - res["abd_bottom_z"]) * sp[2] if vert_present else None
    res["n_slices"] = nz
    res["max_vert_mm"] = max((zr[v][1] - zr[v][0]) * sp[2] for v in ABD_VERT) if vert_present else None

    res["valid"] = res["liver_ok"] and res["gb_ok"] and res["abd_ok"]

    # coronal preview through the gallbladder centroid
    y = int(np.round(np.where(gb.any(axis=(0, 2)))[0].mean())) if gb.any() else ct.shape[1] // 2
    sl = np.clip(ct[::-1, y, :].T.astype(np.float32), -200, 300)  # (z, x), patient right on image left
    fig, ax = plt.subplots(figsize=(3.2, 3.2 * min(2.5, nz * sp[2] / (ct.shape[0] * sp[0]))), dpi=90)
    ax.imshow(sl, cmap="gray", origin="lower", aspect=sp[2] / sp[0])
    for o, c in [("liver", "red"), ("gallbladder", "lime")]:
        proj = masks[o][::-1].any(axis=1).T
        if proj.any():
            ax.contour(proj, levels=[0.5], colors=c, linewidths=0.8)
    for z, lbl in [(res["abd_top_z"], "T12"), (res["abd_bottom_z"], "L5")]:
        if z is not None:
            ax.axhline(z, color="cyan", ls="--", lw=0.8)
            ax.text(2, z, lbl, color="cyan", fontsize=7, va="bottom")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(QC, f"{sid}.png"))
    plt.close(fig)
    return res


def write_html(df, ds):
    cards = []
    for sid, r in df.iterrows():
        flags = " ".join(
            f'<span class="{"ok" if r[k] else "bad"}">{k[:-3]}</span>' for k in ["liver_ok", "gb_ok", "abd_ok"]
        )
        cards.append(
            f'<div class="card {"" if r.valid else "invalid"}" data-id="{sid}">'
            f'<img loading="lazy" src="{sid}.png"><div class="cap"><b>{sid}</b> {flags}<br>'
            f'<small>{html.escape(str(ds.at[sid, "study_type"]))} · abd {r.abd_len_mm:.0f} mm · GB {r.gb_ml:.1f} ml</small><br>'
            f'<label><input type="checkbox" class="rej"> reject</label></div></div>'
        )
    page = f"""<!doctype html><meta charset="utf-8"><title>SonoBelief QC</title>
<style>
body{{font:13px sans-serif;margin:16px;background:#111;color:#ddd}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}
.card{{background:#222;padding:4px;width:230px;border:2px solid #222}}
.card img{{width:100%;background:#000}}
.card.invalid{{border-color:#c33}} .card.rejected{{opacity:.35;border-color:#e90}}
.ok{{color:#4c4}} .bad{{color:#f55;font-weight:bold}}
.bar{{position:sticky;top:0;background:#111;padding:8px 0;z-index:1}}
</style>
<div class="bar"><b>{len(df)} studies</b> · auto-valid {int(df.valid.sum())} ·
rejected <span id="nrej">0</span> ·
<label><input type="checkbox" id="onlybad"> show only auto-invalid / rejected</label> ·
<button id="exp">export rejected CSV</button>
<span style="color:#888"> red = liver, green = gallbladder, cyan = T12 top / L5 bottom</span></div>
<div class="grid">{''.join(cards)}</div>
<script>
const KEY="sonobelief_qc_rejected";
let rej=new Set(); try{{rej=new Set(JSON.parse(localStorage.getItem(KEY)||"[]"))}}catch(e){{}}
const save=()=>{{try{{localStorage.setItem(KEY,JSON.stringify([...rej]))}}catch(e){{}};document.getElementById("nrej").textContent=rej.size}};
document.querySelectorAll(".card").forEach(c=>{{const cb=c.querySelector(".rej"),id=c.dataset.id;
 cb.checked=rej.has(id);c.classList.toggle("rejected",cb.checked);
 cb.onchange=()=>{{cb.checked?rej.add(id):rej.delete(id);c.classList.toggle("rejected",cb.checked);save()}}}});
document.getElementById("onlybad").onchange=e=>document.querySelectorAll(".card").forEach(c=>
 c.style.display=!e.target.checked||c.classList.contains("invalid")||c.classList.contains("rejected")?"":"none");
document.getElementById("exp").onclick=()=>{{const a=document.createElement("a");
 a.href=URL.createObjectURL(new Blob(["study\\n"+[...rej].sort().join("\\n")+"\\n"],{{type:"text/csv"}}));
 a.download="qc_rejected.csv";a.click()}};
save();
</script>"""
    with open(os.path.join(QC, "index.html"), "w") as f:
        f.write(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejected", help="qc_rejected.csv exported from the gallery; these studies are excluded")
    args = ap.parse_args()
    os.makedirs(QC, exist_ok=True)
    ds = pd.read_csv(os.path.join(INTER, "dataset.csv"), index_col="study")
    rows = ds.reset_index().to_dict("records")
    with ProcessPoolExecutor(20) as ex:
        df = pd.DataFrame(ex.map(check, rows, chunksize=4)).set_index("study").sort_index()
    df.to_csv(os.path.join(QC, "qc_results.csv"))
    write_html(df, ds)

    print(f"checked {len(df)} studies: valid {df.valid.sum()}, invalid {(~df.valid).sum()}")
    for k in ["liver_ok", "gb_ok", "abd_ok"]:
        if (~df[k]).any():
            print(f"  failed {k}: {', '.join(df.index[~df[k]])}")
    print(f"abdominal range (T12 top -> L5 bottom): median {df.abd_len_mm.median():.0f} mm, "
          f"min {df.abd_len_mm.min():.0f}, max {df.abd_len_mm.max():.0f}")
    rejected = set(pd.read_csv(args.rejected).study) if args.rejected else set()
    final = ds.loc[df.index[df.valid & ~df.index.isin(rejected)]].join(df[["abd_top_z", "abd_bottom_z", "abd_len_mm"]])
    final.to_csv(os.path.join(DATA_OUT, "dataset_valid.csv"))
    print(f"dataset_valid.csv: {len(final)} studies ({len(rejected)} manually rejected)")
    print(f"gallery: {os.path.join(QC, 'index.html')}")


if __name__ == "__main__":
    main()
