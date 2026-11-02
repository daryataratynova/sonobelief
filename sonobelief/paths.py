"""Where everything lives. Override with environment variables; nothing else in the package hard-codes a path.

    SONOBELIEF_WORK     generated data: tables, simulated views, target banks, trajectories (default: <repo>/work)
    SONOBELIEF_RUNS     training and evaluation outputs (default: <repo>/runs)
    TOTALSEG_DIR        TotalSegmentator v2.0.1 (folder with meta.csv and one subfolder per study)
    ENHANCE_DIR         ENHANCE.PET (folder with imaging-data/ and totalseg/)
    AMOS_DIR            AMOS22 (folder with imagesTr/, labelsTr/, ...)

Frozen model components are listed in configs/hybrid_frozen.json and configs/target_belief_frozen.json as paths
relative to the repository root (weights/<component> by default).
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(REPO, "configs")
WEIGHTS = os.path.join(REPO, "weights")
WORK = os.environ.get("SONOBELIEF_WORK", os.path.join(REPO, "work"))
RUNS = os.environ.get("SONOBELIEF_RUNS", os.path.join(REPO, "runs"))
QC = os.path.join(WORK, "qc")

TOTALSEG = os.environ.get("TOTALSEG_DIR", "")
ENHANCE = os.environ.get("ENHANCE_DIR", "")
AMOS = os.environ.get("AMOS_DIR", "")

SEG_BODY = os.path.join(WORK, "seg_body")      # trunk masks (TotalSegmentator "body" task)
PLANES = os.path.join(WORK, "planes")          # random liver views for E_img / D_obs training
ANATOMY = os.path.join(WORK, "anatomy")        # 3D [body, liver, GB] crops around the liver centroid

FROZEN = json.load(open(os.path.join(CONFIGS, "hybrid_frozen.json")))
EMB_RUN = os.path.join(REPO, FROZEN["E_img"])  # frozen image encoder E_img (+ its embeddings_<split>.npz)
