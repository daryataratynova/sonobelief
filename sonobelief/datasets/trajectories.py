"""Source trajectories with frozen embeddings (work/traj_emb/<split>.npz) and their conditions: the first K
pre-target-acquisition observations of a trajectory, P_ref = the first view.
"""
import os

import numpy as np
import torch

from sonobelief import paths
from sonobelief.datasets.views import Views
from sonobelief.flow import KMAX
from sonobelief.geometry import canonical, placement, relative
from sonobelief.paths import ANATOMY

TRAJ_EMB = os.path.join(paths.WORK, "traj_emb")
K_MAIN = (1, 2, 4, 8)


class TrajSet:
    """Per split, padded per trajectory: e (T, L, 128) standardized, pose (T, L, 9) in the liver frame,
    n_pre (T,) pre-acquisition views, study index (T,)."""

    def __init__(self, split, zl_run, emb, stats=None, aug=False):
        """aug: also the train augmentation pool (traj_emb/train_aug.npz, extra seeds of the frozen generator)."""
        self.views = Views(split, zl_run, stats)          # z_L, r_GB, embedding standardization (train)
        self.stats = self.views.stats
        idx = {s: i for i, s in enumerate(self.views.studies)}
        lc = {s: np.load(os.path.join(ANATOMY, f"{s}.npz"))["liver_c"] for s in self.views.studies}
        names = [split] + (["train_aug"] if aug else [])
        assert not aug or split == "train", "augmentation is train-only"
        cols = ("study", "traj", "length", "t_gb", "apex", "beam", "lateral", emb)
        files = []
        for n in names:                                   # materialize once (NpzFile[key] decompresses per access)
            with np.load(os.path.join(TRAJ_EMB, f"{n}.npz")) as E:
                files.append({c: E[c] for c in cols})
        L = int(max(E["length"].max() for E in files))
        es, poses, n_pres, studies, row0s, srcs = [], [], [], [], [], []
        for f_i, E in enumerate(files):
            key = np.char.add(np.char.add(E["study"].astype(str), "_"), E["traj"].astype(str))
            _, first = np.unique(key, return_index=True)
            first = np.sort(first)
            T = len(first)
            e = np.zeros((T, L, 128), np.float16)
            pose = np.zeros((T, L, 9), np.float32)
            n_pre, study = np.zeros(T, np.int64), np.zeros(T, np.int64)
            for k, i in enumerate(first):
                n = int(E["length"][i]); t_gb = int(E["t_gb"][i])
                sl = slice(i, i + n)
                sid = E["study"][i]
                e[k, :n] = E[emb][sl]
                pose[k, :n] = np.concatenate([(E["apex"][sl] - lc[sid]) / 100, E["beam"][sl], E["lateral"][sl]], 1)
                n_pre[k] = t_gb if t_gb >= 0 else n
                study[k] = idx[sid]
            es.append(e); poses.append(pose); n_pres.append(n_pre); studies.append(study)
            row0s.append(first); srcs.append(np.full(T, f_i))
        mu, sd = self.stats["e"]
        self.e = (torch.from_numpy(np.concatenate(es)).cuda().float() - mu) / sd
        self.pose = torch.from_numpy(np.concatenate(poses)).cuda()
        self.n_pre = torch.from_numpy(np.concatenate(n_pres)).cuda()
        self.study = torch.from_numpy(np.concatenate(studies)).cuda()
        self.row0 = torch.from_numpy(np.concatenate(row0s)).cuda()   # row of step 0 in traj_emb/<file>.npz
        self.source = torch.from_numpy(np.concatenate(srcs)).cuda()  # 0 = official trajectories, 1 = augmentation
        self.split = split

    def __len__(self):
        return len(self.n_pre)

    def eligible(self, K):
        return (self.n_pre >= max(K, 1)).nonzero()[:, 0]


def build(D, tj, start, k):
    """Trajectories tj (B,), window start (B,), K (B,) -> tokens (B, KMAX, 137), valid (B, KMAX), raw w (B, 9)."""
    pos = start[:, None] + torch.arange(KMAX, device="cuda")[None]
    valid = torch.arange(KMAX, device="cuda")[None] < k[:, None]
    pos = torch.where(valid, pos, start[:, None]).clamp(max=D.pose.shape[1] - 1)
    pose_ref = D.pose[tj, start]
    rel = relative(D.pose[tj[:, None], pos], pose_ref)
    tok = torch.cat([D.e[tj[:, None], pos], rel], -1) * valid[..., None]
    R_L, t = placement(pose_ref)
    return tok, valid, torch.cat([t, R_L[..., 0], R_L[..., 1]], -1)


def draw(D, B, k_set, windows, gen):
    """Random conditions: K from k_set, a trajectory with >= K pre-acquisition views, a prefix or window start."""
    k = torch.tensor(k_set, device="cuda")[torch.randint(len(k_set), (B,), device="cuda", generator=gen)]
    tj = torch.empty(B, dtype=torch.long, device="cuda")
    for K in k_set:
        m = k == K
        el = D.eligible(K)
        tj[m] = el[torch.randint(len(el), (int(m.sum()),), device="cuda", generator=gen)]
    if windows:
        span = (D.n_pre[tj] - k.clamp(min=1) + 1).clamp(min=1)
        start = (torch.rand(B, device="cuda", generator=gen) * span).long()
    else:
        start = torch.zeros(B, dtype=torch.long, device="cuda")
    return tj, start, k


SRC = os.path.join(paths.WORK, "source_traj")


def traj_frames(split, studies):
    """Observed masks for the trajectory frames of traj_emb/<split>.npz (canonical roll), in the same row order."""
    E = np.load(os.path.join(paths.WORK, "traj_emb", f"{split}.npz"))
    keep = np.isin(E["study"], studies)
    masks = []
    for sid in [s for s in studies if s in set(E["study"][keep])]:
        d = np.load(os.path.join(SRC, f"{sid}.npz"))
        obs = torch.from_numpy(np.unpackbits(d["obs"], axis=-1).astype(bool)).cuda()
        for j, (n, off) in enumerate(zip(d["length"], d["offset"])):
            _, flip = canonical(torch.from_numpy(d["beam"][j, :n]).float().cuda(), torch.from_numpy(d["lateral"][j, :n]).float().cuda())
            o = obs[off:off + n]
            masks.append(torch.where(flip[:, None, None, None], o.flip(-1), o))
    rows = {k: E[k][keep] for k in E.files}
    order = np.concatenate([np.flatnonzero(rows["study"] == s) for s in dict.fromkeys(rows["study"])])
    rows = {k: v[order] for k, v in rows.items()}
    return rows, torch.cat(masks)
