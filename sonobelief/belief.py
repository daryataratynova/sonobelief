"""Liver-shape belief p(z_L | p_L, O) and the GB-offset conditional p(r_GB | z_L).

  z_posterior     MAP of z_L under D_obs (Adam from z = 0) and the Gauss-Newton Laplace precision
  Forward, lm_fit occlusion-aware observation model and its Levenberg-Marquardt MAP (shadowed pixels are not observed)
  laplace_sample  z ~ N(z_hat, H^-1)
  rgb_conditional ridge mean mu_r(z_L) and out-of-fold residual covariance of r_GB = c_GB - c_L
TF32 matmuls are enabled on import, as in the frozen evaluation.
"""
import torch
from torch.func import jacfwd, jvp, vmap

from sonobelief.regression import ridge_pred

MAP_STEPS, MAP_LR, LAM, FOLDS = 300, 0.05, 0.1, 10


def rgb_conditional(tr):
    """mu_r(z) and Sigma_r (mm^2) from 10-fold out-of-fold ridge residuals on train patients."""
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(len(tr), generator=g).cuda()
    oof = torch.zeros_like(tr.r_gb)
    for f in range(FOLDS):
        te = perm[f::FOLDS]
        trn = perm[~torch.isin(perm, te)]
        oof[te] = ridge_pred(tr.zl[trn], tr.r_gb[trn], tr.zl[te], LAM)
    res = tr.r_gb - oof
    return (lambda z: ridge_pred(tr.zl, tr.r_gb, z, LAM)), torch.cov(res.T), res.norm(dim=1).mean().item()


def z_posterior(dobs, pl, e, sigma2, chunk=4096, jchunk=512):
    """Observed views in the liver frame pl (B, K, 9), embeddings e (B, K, 128) -> MAP z_hat (B, 32), the Cholesky
    factor L of the Gauss-Newton precision H = L L^T (B, 32, 32) and the Laplace log-evidence (B,) (up to a const)."""
    B, K = pl.shape[:2]
    zd = dobs.zin.in_features
    eye = torch.eye(zd, device="cuda")
    if K == 0:
        return torch.zeros(B, zd, device="cuda"), eye.expand(B, -1, -1), torch.zeros(B, device="cuda")
    energy = lambda z, p, t: ((dobs(z, p) - t) ** 2).sum((1, 2)) / (2 * sigma2) + 0.5 * (z ** 2).sum(1)
    zs, Es = [], []
    for i in range(0, B, chunk):
        p, t = pl[i:i + chunk], e[i:i + chunk]
        z = torch.zeros(len(p), zd, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([z], lr=MAP_LR)
        for _ in range(MAP_STEPS):
            opt.zero_grad()
            energy(z, p, t).sum().backward()
            opt.step()
        zs.append(z.detach())
        with torch.no_grad():
            Es.append(energy(zs[-1], p, t))
    z_hat, E = torch.cat(zs), torch.cat(Es)
    f = lambda z1, p1: dobs(z1[None], p1[None])[0].flatten()
    Ls = []
    with torch.no_grad():
        for i in range(0, B, jchunk):
            J = vmap(jacfwd(f))(z_hat[i:i + jchunk], pl[i:i + jchunk])          # (b, K*128, 32)
            Ls.append(torch.linalg.cholesky(J.transpose(1, 2) @ J / sigma2 + eye))
    L = torch.cat(Ls)
    return z_hat, L, -E - torch.log(L.diagonal(dim1=1, dim2=2)).sum(1)


def laplace_sample(z_hat, L, g):
    """z = z_hat + L^-T eps  ->  cov (L L^T)^-1 = H^-1."""
    eps = torch.randn(z_hat.shape, device="cuda", generator=g)
    return z_hat + torch.linalg.solve_triangular(L.transpose(1, 2), eps[..., None], upper=True)[..., 0]


JCHUNK, FCHUNK = 16, 1024          # views per forward-mode Jacobian call (x 32 tangents) / per plain forward call
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Forward:
    """Occlusion-aware observation model: z (n, 32), pose (n, 9) liver frame, valid (n, H, W) -> e (n, 128) std."""

    def __init__(self, dobs, ae, stats):
        self.dobs, self.ae = dobs, ae
        self.mu, self.sd = stats["e"]

    def __call__(self, z, pl, v):
        e = self.dobs(z, pl[:, None])[:, 0] * self.sd + self.mu
        liver = torch.sigmoid(self.ae.dec(e)[:, 0]) * v
        return (self.ae.enc(torch.stack([liver, v], 1)) - self.mu) / self.sd

    @torch.no_grad()
    def value(self, z, pl, v_of):
        return torch.cat([self(z[i:i + FCHUNK], pl[i:i + FCHUNK], v_of(slice(i, i + FCHUNK)))
                          for i in range(0, len(z), FCHUNK)])

    @torch.no_grad()
    def jac(self, z, pl, v_of):
        """-> f (n, 128), J = df/dz (n, 128, 32), forward mode with one tangent per z dimension."""
        d = z.shape[1]
        fs, Js = [], []
        for i in range(0, len(z), JCHUNK):
            zc, pc, vc = z[i:i + JCHUNK], pl[i:i + JCHUNK], v_of(slice(i, i + JCHUNK))
            n = len(zc)
            rep = lambda a: a.repeat(d, *[1] * (a.ndim - 1))
            tang = torch.eye(d, device="cuda").repeat_interleave(n, 0)
            f, J = jvp(lambda zz: self(zz, rep(pc), rep(vc)), (rep(zc),), (tang,))
            fs.append(f[:n]); Js.append(J.view(d, n, -1).permute(1, 2, 0))
        return torch.cat(fs), torch.cat(Js)


def lm_fit(F, z0, pl, e, v_of, sigma2, iters):
    """Levenberg-Marquardt MAP of E(z) = sum_k ||f(z, P_k) - e_k||^2 / (2 sigma^2) + ||z||^2 / 2 for P problems with
    K views each: pl (P, K, 9), e (P, K, 128), v_of(view index tensor) -> valid masks.  Returns z_hat (P, 32), the
    Cholesky factor of H = J^T J / sigma^2 + I at z_hat, the final energies and the share of problems that improved."""
    P, K = pl.shape[:2]
    d = z0.shape[1]
    eye = torch.eye(d, device="cuda")
    flat = lambda a: a.reshape(P * K, *a.shape[2:])
    pl_f, e_f = flat(pl), flat(e)
    vidx = torch.arange(P * K, device="cuda")

    def assemble(z, sel):
        """Energy, gradient and Gauss-Newton precision for problems sel."""
        rows = (sel[:, None] * K + torch.arange(K, device="cuda")).flatten()
        f, J = F.jac(z[sel].repeat_interleave(K, 0), pl_f[rows], lambda s: v_of(vidx[rows][s]))
        r = (f - e_f[rows]).view(len(sel), K, -1)
        J = J.view(len(sel), K, -1, d)
        En = (r ** 2).sum((1, 2)) / (2 * sigma2) + 0.5 * (z[sel] ** 2).sum(1)
        g = torch.einsum("pkoi,pko->pi", J, r) / sigma2 + z[sel]
        H = torch.einsum("pkoi,pkoj->pij", J, J) / sigma2 + eye
        return En, g, H

    def energy(z):
        f = F.value(z.repeat_interleave(K, 0), pl_f, lambda s: v_of(vidx[s]))
        return ((f - e_f) ** 2).view(P, K, -1).sum((1, 2)) / (2 * sigma2) + 0.5 * (z ** 2).sum(1)

    z = z0.clone()
    allp = torch.arange(P, device="cuda")
    E, g, H = assemble(z, allp)
    E0 = E.clone()
    lam = torch.full((P,), 1e-2, device="cuda")
    for _ in range(iters):
        delta = torch.linalg.solve(H + lam[:, None, None] * eye, -g[..., None])[..., 0]
        E_new = energy(z + delta)
        acc = E_new < E
        lam = torch.where(acc, lam / 3, lam * 3)
        if acc.any():
            z[acc] = z[acc] + delta[acc]
            sel = acc.nonzero()[:, 0]
            E[sel], g[sel], H[sel] = assemble(z, sel)
    return z, torch.linalg.cholesky(H), E, (E < E0).float().mean().item()
