"""
tensor_dl.py
============
Tensor Dictionary Learning under one API, with five interchangeable atom models:

    Vectorized (baseline) | CP | Tucker | Tensor-Train (TT) | Tensor-Ring (TR)

DESIGN (why it's a fair comparison)
-----------------------------------
Every model exposes the *same* interface:

    .fit(X)                 learn the structured dictionary from training tensors
    .dictionary  -> D_eff   (n_features, n_atoms) matrix, unit-norm columns
    .encode(X, s)           OMP sparse coding against D_eff  (SHARED across models)
    .decode(codes)          linear reconstruction            (SHARED across models)
    .n_params               number of free parameters stored in the dictionary

Because encode/decode are identical for all models, the ONLY things that differ are
(a) how the atoms are parameterised/constrained and (b) how many parameters that costs.
That isolates the effect of the tensor-decomposition structure on accuracy / efficiency /
robustness, which is the whole point of the benchmark.

Author: built for a CP/Tucker/TT-DL benchmark.
"""

from __future__ import annotations
import time
import numpy as np
from numpy.linalg import svd, norm
from sklearn.linear_model import orthogonal_mp
import tensorly as tl
from tensorly.decomposition import parafac, tensor_train, tensor_ring, tucker

tl.set_backend("numpy")
EPS = 1e-12


# --------------------------------------------------------------------------------------
# Shared utilities
# --------------------------------------------------------------------------------------
def _normalize_cols(D):
    """Unit-L2 normalise columns; drop near-zero columns to a tiny random atom."""
    n = norm(D, axis=0)
    dead = n < EPS
    if np.any(dead):
        D[:, dead] = np.random.randn(D.shape[0], int(dead.sum())) * 1e-3
        n = norm(D, axis=0)
    return D / (n + EPS)


# --------------------------------------------------------------------------------------
# Atom projections -- the ONLY thing that differs between free/CP/TT/TR atoms
# Each takes a 3rd-order tensor and returns (projected_tensor, n_params_of_atom)
# --------------------------------------------------------------------------------------
def _proj_free(T, rank=None, **kw):
    return T, T.size, {}


def _proj_cp(T, rank=1, cp_init="svd", **kw):
    try:
        cp = parafac(T, rank=rank, n_iter_max=25, init=cp_init,
                     normalize_factors=False, tol=1e-6)
        Tp = tl.cp_to_tensor(cp)
        nparam = sum(f.size for f in cp.factors)          # weights folded into factors
        # CP degeneracy diagnostic: sum of component magnitudes / norm of their sum.
        # ~1 means well-separated; >>1 means diverging components that cancel (degenerate).
        comp = sum(np.prod([norm(f[:, r]) for f in cp.factors]) for r in range(rank))
        deg = float(comp / (norm(Tp) + EPS))
    except Exception:
        Tp, nparam, _ = _proj_free(T)
        deg = 1.0
    return Tp, nparam, {"deg": deg}


def _proj_tt(T, rank=2, **kw):
    shp = T.shape
    ranks = [1] + [rank] * (len(shp) - 1) + [1]
    try:
        cores = tensor_train(T, rank=ranks)
        Tp = tl.tt_to_tensor(cores)
        nparam = sum(c.size for c in cores)
    except Exception:
        Tp, nparam, _ = _proj_free(T)
    return Tp, nparam, {}


def _proj_tr(T, rank=2, **kw):
    shp = T.shape
    ranks = [rank] * (len(shp) + 1)                       # tensor-ring ranks, wrap-around
    try:
        cores = tensor_ring(T, rank=ranks)
        Tp = tl.tr_to_tensor(cores)
        nparam = sum(c.size for c in cores)
    except Exception:
        Tp, nparam, _ = _proj_tt(T, rank=rank)
    return Tp, nparam, {}


def _proj_btd(T, rank=2, n_blocks=1, **kw):
    """Block-Term Decomposition: sum of `n_blocks` multilinear rank-(L,L,L) Tucker blocks.
    Greedy residual fitting. L=1 reduces to CP; one full-rank block reduces to Tucker."""
    L = rank
    ranks = [min(L, d) for d in T.shape]
    resid = T.astype(float).copy()
    recon = np.zeros_like(resid)
    nparam = 0
    try:
        for _ in range(n_blocks):
            core, factors = tucker(resid, rank=ranks, n_iter_max=50, init="svd")
            block = tl.tucker_to_tensor((core, factors))
            recon = recon + block
            resid = resid - block
            nparam += core.size + sum(f.size for f in factors)
    except Exception:
        return _proj_free(T)
    return recon, nparam, {}


_PROJECTORS = {"free": _proj_free, "cp": _proj_cp, "tt": _proj_tt,
               "tr": _proj_tr, "btd": _proj_btd}


# --------------------------------------------------------------------------------------
# Base class: shared encode / decode / reconstruct / metrics
# --------------------------------------------------------------------------------------
class BaseTensorDL:
    name = "base"

    def __init__(self, atom_shape, n_atoms=128, n_iter=12, seed=0):
        self.atom_shape = tuple(atom_shape)
        self.n_features = int(np.prod(atom_shape))
        self.n_atoms = n_atoms
        self.n_iter = n_iter
        self.seed = seed
        self.D_ = None          # (n_features, n_atoms) unit-norm
        self._param_count = None
        self.fit_time_ = None

    # ---- shared sparse coding / reconstruction ----
    @property
    def dictionary(self):
        return self.D_

    @property
    def n_params(self):
        return self._param_count

    def encode(self, X, n_nonzero):
        """X: (M, *atom_shape) -> codes (M, n_atoms) via OMP on D_eff."""
        Y = X.reshape(X.shape[0], -1).T          # (n_features, M)
        coef = orthogonal_mp(self.D_, Y, n_nonzero_coefs=int(n_nonzero))
        return coef.T                            # (M, n_atoms)

    def decode(self, codes):
        flat = codes @ self.D_.T                  # (M, n_features)
        return flat.reshape((codes.shape[0],) + self.atom_shape)

    def reconstruct(self, X, n_nonzero):
        return self.decode(self.encode(X, n_nonzero))

    # ---- to be implemented ----
    def fit(self, X):
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Tensor K-SVD engine (covers Vectorized / CP / TT / TR via the atom projector)
# Standard K-SVD: alternate OMP coding with rank-1 SVD atom+code update, BUT the
# updated atom is projected onto the chosen tensor manifold (rank-1 CP, low TT-rank, ...).
# 'free' projection == ordinary K-SVD == the vectorized baseline.
# --------------------------------------------------------------------------------------
class KSVDTensorDL(BaseTensorDL):
    def __init__(self, atom_shape, structure="free", rank=2,
                 n_atoms=128, n_iter=12, sparsity=4, seed=0,
                 init="data", proj_kw=None):
        super().__init__(atom_shape, n_atoms, n_iter, seed)
        self.structure = structure
        self.rank = rank
        self.sparsity = sparsity
        self.init = init                       # 'data' (warm) or 'random' (cold)
        self.proj_kw = proj_kw or {}
        self.project = _PROJECTORS[structure]
        self.degeneracy_ = None
        self.name = {"free": "Vectorized", "cp": "CP", "tt": "TT",
                     "tr": "TR", "btd": "BTD"}[structure]

    def _project(self, T):
        return self.project(T, self.rank, **self.proj_kw)

    def _init_dict(self, Y):
        rng = np.random.default_rng(self.seed)
        M = Y.shape[1]
        if self.init == "random":
            D = rng.standard_normal((self.n_features, self.n_atoms))
        else:                                                # warm: random data samples
            idx = rng.choice(M, size=min(self.n_atoms, M), replace=False)
            D = Y[:, idx].copy()
            if D.shape[1] < self.n_atoms:                    # pad if few samples
                extra = rng.standard_normal((self.n_features, self.n_atoms - D.shape[1]))
                D = np.hstack([D, extra])
        atom_params = np.empty(self.n_atoms)
        for k in range(self.n_atoms):                        # start on-manifold
            Tp, p, _ = self._project(D[:, k].reshape(self.atom_shape))
            D[:, k] = Tp.ravel()
            atom_params[k] = p
        self._atom_params = atom_params
        return _normalize_cols(D)

    def fit(self, X):
        np.random.seed(self.seed)
        t0 = time.time()
        Y = X.reshape(X.shape[0], -1).T          # (n_features, M)
        D = self._init_dict(Y)
        last_degs = []

        for it in range(self.n_iter):
            codes = orthogonal_mp(D, Y, n_nonzero_coefs=self.sparsity).T   # (M, n_atoms)
            degs = []
            for k in range(self.n_atoms):
                omega = np.nonzero(codes[:, k])[0]
                if omega.size == 0:
                    # dead atom: re-seed on worst-reconstructed sample
                    resid = Y - D @ codes.T
                    worst = np.argmax(norm(resid, axis=0))
                    Tp, p, _ = self._project(Y[:, worst].reshape(self.atom_shape))
                    D[:, k] = Tp.ravel() / (norm(Tp.ravel()) + EPS)
                    self._atom_params[k] = p
                    continue
                E = Y[:, omega] - D @ codes[omega].T + np.outer(D[:, k], codes[omega, k])
                U, S, Vt = svd(E, full_matrices=False)
                atom = U[:, 0]
                Tp, p, info = self._project(atom.reshape(self.atom_shape))
                atom = Tp.ravel()
                nrm = norm(atom)
                if nrm < EPS:
                    continue
                atom /= nrm
                D[:, k] = atom
                self._atom_params[k] = p
                codes[omega, k] = E.T @ atom      # refresh active codes onto new atom
                if "deg" in info:
                    degs.append(info["deg"])
            if it == self.n_iter - 1:
                last_degs = degs

        self.D_ = _normalize_cols(D)
        self._param_count = int(self._atom_params.sum())
        self.degeneracy_ = float(np.mean(last_degs)) if last_degs else None
        self.fit_time_ = time.time() - t0
        return self


# --------------------------------------------------------------------------------------
# Tucker / separable (Kronecker) dictionary -- IN THE SHARED HARNESS.
# Same outer loop as the projection methods: OMP sparse-code against D_eff, then update.
# The Tucker-specific "snap toward Kronecker shape" update is a separable ALS on the mode
# factors U1..UN (Tucker's constraint is shared across all atoms, so it can't be snapped
# one atom at a time -- the whole dictionary is re-fit to the Kronecker manifold instead).
# D_eff atoms = every tensor product u1_i o u2_j o ...   Parameters = sum of |U_n|.
# --------------------------------------------------------------------------------------
class TuckerDL(BaseTensorDL):
    name = "Tucker"

    def __init__(self, atom_shape, mode_ranks=None, n_iter=12,
                 sparsity=4, seed=0, mode_sparsity=None, ridge=1e-3):
        super().__init__(atom_shape, n_atoms=None, n_iter=n_iter, seed=seed)
        if mode_ranks is None:
            mode_ranks = [d + 1 for d in atom_shape]      # mildly overcomplete per mode
        self.mode_ranks = list(mode_ranks)
        # accept legacy `mode_sparsity` but use one shared DL sparsity like every other model
        self.sparsity = mode_sparsity if mode_sparsity is not None else sparsity
        self.ridge = ridge
        self.U_ = None
        self.degeneracy_ = None

    def _build_Deff(self, U):
        N = len(self.atom_shape)
        big, sub = "ijklm"[:N], "abcde"[:N]
        eq = ",".join(f"{a}{b}" for a, b in zip(big, sub)) + "->" + big + sub
        return np.einsum(eq, *U).reshape(self.n_features, -1)   # (n_features, prod(k))

    def _init_modes(self, X):
        rng = np.random.default_rng(self.seed)
        N = len(self.atom_shape)
        U = []
        for mode in range(N):
            fibers = tl.unfold(np.moveaxis(X, 0, -1), mode)     # (n_mode, rest)
            # leading left singular vectors (HOSVD), padded with random to reach k_mode
            try:
                Uu, _, _ = svd(fibers, full_matrices=False)
            except Exception:
                Uu = rng.standard_normal((self.atom_shape[mode], self.atom_shape[mode]))
            k = self.mode_ranks[mode]
            cols = [Uu[:, :min(k, Uu.shape[1])]]
            if k > Uu.shape[1]:
                cols.append(rng.standard_normal((self.atom_shape[mode], k - Uu.shape[1])) * 0.1)
            U.append(np.hstack(cols))
        return U

    def fit(self, X):
        t0 = time.time()
        np.random.seed(self.seed)
        N = len(self.atom_shape)
        ks = self.mode_ranks
        U = self._init_modes(X)
        Xarr = X.astype(float)                       # (M, n1, ..., nN)
        M = Xarr.shape[0]

        for _ in range(self.n_iter):
            D = _normalize_cols(self._build_Deff(U))
            # SAME sparse coding as every other model
            codes = orthogonal_mp(D, X.reshape(M, -1).T,
                                  n_nonzero_coefs=self.sparsity).T          # (M, prod(k))
            Ccore = codes.reshape((M,) + tuple(ks))   # per-sample sparse core tensor
            # separable ALS: update each mode factor to best fit X given the cores
            for mode in range(N):
                other = [p for p in range(N) if p != mode]
                # G_m = Ccore_m  x_{p!=mode} U_p   (leave sample axis 0 untouched)
                G = tl.tenalg.multi_mode_dot(Ccore, [U[p] for p in other],
                                             modes=[p + 1 for p in other])
                # unfold X and G with the `mode` axis as rows, everything else as columns
                Xn = np.moveaxis(Xarr, mode + 1, 0).reshape(self.atom_shape[mode], -1)
                Gn = np.moveaxis(G, mode + 1, 0).reshape(ks[mode], -1)
                GG = Gn @ Gn.T + self.ridge * np.eye(ks[mode])
                U[mode] = (Xn @ Gn.T) @ np.linalg.inv(GG)     # (n_mode, k_mode)

        self.U_ = U
        self.D_ = _normalize_cols(self._build_Deff(U))
        self.n_atoms = self.D_.shape[1]
        self._param_count = int(sum(u.size for u in U))
        self.fit_time_ = time.time() - t0
        return self


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
def make_model(kind, atom_shape, **kw):
    kind = kind.lower()
    if kind in ("vectorized", "free", "baseline"):
        return KSVDTensorDL(atom_shape, structure="free", **kw)
    if kind == "cp":
        return KSVDTensorDL(atom_shape, structure="cp", **kw)
    if kind == "tt":
        return KSVDTensorDL(atom_shape, structure="tt", **kw)
    if kind == "tr":
        return KSVDTensorDL(atom_shape, structure="tr", **kw)
    if kind == "btd":
        return KSVDTensorDL(atom_shape, structure="btd", **kw)
    if kind == "tucker":
        mr = kw.pop("mode_ranks", None)
        return TuckerDL(atom_shape, mode_ranks=mr,
                        n_iter=kw.get("n_iter", 12),
                        sparsity=kw.get("sparsity", 4),
                        mode_sparsity=kw.get("mode_sparsity", None),
                        seed=kw.get("seed", 0))
    raise ValueError(kind)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def psnr(clean, recon):
    mse = np.mean((clean - recon) ** 2)
    if mse < EPS:
        return 99.0
    peak = clean.max() - clean.min()
    return 20 * np.log10(peak / np.sqrt(mse) + EPS)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))
