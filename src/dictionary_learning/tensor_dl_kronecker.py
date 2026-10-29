"""
tensor_dl.py -- structured tensor dictionary learning.

Every model learns an over-complete dictionary of K atoms, each atom a 3rd-order
tensor of shape `atom_shape`. Models differ ONLY in the low-rank structure each
atom is constrained to live on:

    vectorized : none (dense atom)                    -> prod(shape) params/atom
    cp         : CP rank-R                            -> R * sum(shape)
    tt         : TT with internal rank r              -> r*s0 + r*s1*r + r*s2
    tr         : Tensor-Ring, all ranks r             -> r^2 * sum(shape)
    btd        : b blocks of rank-(L,L,L) Tucker      -> b*(L^N + L*sum(shape))
    tucker     : ONE separable/Kronecker dictionary   -> sum_n s_n * k_n  (all atoms)

Training = alternating (i) batch-OMP sparse coding, (ii) MOD dictionary update,
(iii) projection of every atom back onto its structural manifold, (iv) re-normalise.
The Tucker/Kronecker model instead does closed-form ALS on its mode factors.
"""
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram

EPS = 1e-10


# ----------------------------------------------------------------------------- metrics
def psnr(clean, recon, data_range=None):
    """PSNR in dB. data_range defaults to max|clean| (patches are unit-norm)."""
    clean = np.asarray(clean, float)
    recon = np.asarray(recon, float)
    mse = np.mean((clean - recon) ** 2)
    if data_range is None:
        data_range = float(np.max(clean) - np.min(clean))
    return 10.0 * np.log10((data_range ** 2) / (mse + 1e-20))


# ------------------------------------------------------------------- structure projections
def _unit(A):
    """Row-normalise a stack of atoms (K, *shape)."""
    n = np.sqrt((A ** 2).sum(axis=tuple(range(1, A.ndim)), keepdims=True))
    return A / (n + EPS)


def project_cp(A, rank, n_iter=8):
    """Batched CP-ALS projection of atoms A (K, *shape) onto CP rank-`rank`."""
    K = A.shape[0]
    shape = A.shape[1:]
    N = len(shape)
    rng = np.random.default_rng(0)
    F = [rng.standard_normal((K, s, rank)) for s in shape]
    # svd init on mode-0 helps conditioning
    for n in range(N):
        Un = np.moveaxis(A, n + 1, 1).reshape(K, shape[n], -1)
        u, _, _ = np.linalg.svd(Un, full_matrices=False)
        r = min(rank, u.shape[2])
        F[n][:, :, :r] = u[:, :, :r]
    for _ in range(n_iter):
        for n in range(N):
            # khatri-rao of the other factors
            others = [m for m in range(N) if m != n]
            KR = F[others[0]]
            for m in others[1:]:
                KR = (KR[:, :, None, :] * F[m][:, None, :, :]).reshape(K, -1, rank)
            An = np.moveaxis(A, n + 1, 1).reshape(K, shape[n], -1)  # (K, s_n, prod)
            G = np.einsum('kpr,kps->krs', KR, KR)                   # (K, R, R)
            B = np.einsum('ksp,kpr->ksr', An, KR)                   # (K, s_n, R)
            G = G + 1e-8 * np.eye(rank)
            F[n] = np.linalg.solve(G, B.transpose(0, 2, 1)).transpose(0, 2, 1)
    out = F[0]
    for n in range(1, N):
        out = (out[..., None, :] * F[n][:, None, :, :]) if out.ndim == 3 else out
    # explicit reconstruction (N=3 general path)
    rec = np.zeros_like(A)
    for r in range(rank):
        term = F[0][:, :, r]
        for n in range(1, N):
            term = term[..., None] * F[n][:, :, r].reshape((K,) + (1,) * n + (shape[n],))
        rec += term
    return rec


def project_tt(A, rank):
    """Batched TT-SVD projection (exact, non-iterative) for 3rd-order atoms."""
    K, s0, s1, s2 = A.shape
    C = A.reshape(K, s0, s1 * s2)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    r1 = min(rank, U.shape[2])
    G0 = U[:, :, :r1]                                     # (K, s0, r1)
    C1 = (S[:, :r1, None] * Vt[:, :r1, :]).reshape(K, r1 * s1, s2)
    U2, S2, Vt2 = np.linalg.svd(C1, full_matrices=False)
    r2 = min(rank, U2.shape[2])
    G1 = U2[:, :, :r2].reshape(K, r1, s1, r2)
    G2 = (S2[:, :r2, None] * Vt2[:, :r2, :])              # (K, r2, s2)
    return np.einsum('kar,krbs,ksc->kabc', G0, G1, G2)


def _tr_reconstruct(cores):
    K = cores[0].shape[0]
    Z = cores[0]                                          # (K, r, s0, r)
    for G in cores[1:]:
        Z = np.einsum('ka...b,kbjc->ka...jc', Z, G)
    return np.einsum('ka...a->k...', Z)


def project_tr(A, rank, n_iter=6):
    """Batched Tensor-Ring ALS projection, all TR-ranks = `rank`."""
    K = A.shape[0]
    shape = A.shape[1:]
    N = len(shape)
    rng = np.random.default_rng(0)
    cores = [rng.standard_normal((K, rank, s, rank)) / np.sqrt(rank) for s in shape]
    for _ in range(n_iter):
        for n in range(N):
            order = [(n + i) % N for i in range(1, N)]
            Z = cores[order[0]]
            for o in order[1:]:
                Z = np.einsum('ka...b,kbjc->ka...jc', Z, cores[o])
            Z = Z.reshape(K, rank, -1, rank)                       # (K, r_{n+1}, P, r_n)
            Zt = Z.transpose(0, 2, 1, 3).reshape(K, -1, rank * rank)
            perm = [0, n + 1] + [o + 1 for o in order]
            Xn = A.transpose(perm).reshape(K, shape[n], -1)        # (K, s_n, P)
            G = np.einsum('kpq,kpr->kqr', Zt, Zt) + 1e-8 * np.eye(rank * rank)
            B = np.einsum('ksp,kpq->ksq', Xn, Zt)
            Gt = np.linalg.solve(G, B.transpose(0, 2, 1)).transpose(0, 2, 1)
            cores[n] = Gt.reshape(K, shape[n], rank, rank).transpose(0, 3, 1, 2)
    return _tr_reconstruct(cores)


def _mdot(T, Mat, mode, adjoint=True):
    """Batched mode product. T:(K,s0..), Mat:(K,s,r).
    adjoint=True  -> contract s  (result has r on that mode)
    adjoint=False -> contract r  (result has s on that mode)"""
    T2 = np.moveaxis(T, mode + 1, -1)
    if adjoint:
        out = np.einsum('k...s,ksr->k...r', T2, Mat)
    else:
        out = np.einsum('k...r,ksr->k...s', T2, Mat)
    return np.moveaxis(out, -1, mode + 1)


def _hooi(A, ranks, n_iter=4):
    """Batched Tucker/HOOI approximation of atoms A (K, *shape)."""
    K = A.shape[0]
    shape = A.shape[1:]
    N = len(shape)
    U = []
    for n in range(N):
        An = np.moveaxis(A, n + 1, 1).reshape(K, shape[n], -1)
        u, _, _ = np.linalg.svd(An, full_matrices=False)
        r = min(ranks[n], u.shape[2])
        Un = u[:, :, :r]
        if r < ranks[n]:
            Un = np.concatenate([Un, np.zeros((K, shape[n], ranks[n] - r))], 2)
        U.append(Un)
    for _ in range(n_iter):
        for n in range(N):
            Y = A
            for m in range(N):
                if m != n:
                    Y = _mdot(Y, U[m], m, adjoint=True)
            Yn = np.moveaxis(Y, n + 1, 1).reshape(K, shape[n], -1)
            u, _, _ = np.linalg.svd(Yn, full_matrices=False)
            r = min(ranks[n], u.shape[2])
            Un = u[:, :, :r]
            if r < ranks[n]:
                Un = np.concatenate([Un, np.zeros((K, shape[n], ranks[n] - r))], 2)
            U[n] = Un
    core = A
    for n in range(N):
        core = _mdot(core, U[n], n, adjoint=True)
    rec = core
    for n in range(N):
        rec = _mdot(rec, U[n], n, adjoint=False)
    return rec


def project_btd(A, L, n_blocks, n_iter=3):
    """Block-term decomposition: sum of `n_blocks` rank-(L,L,L) Tucker terms."""
    N = A.ndim - 1
    ranks = [L] * N
    blocks = [np.zeros_like(A) for _ in range(n_blocks)]
    for _ in range(n_iter):
        for b in range(n_blocks):
            R = A - sum(blocks[j] for j in range(n_blocks) if j != b)
            blocks[b] = _hooi(R, ranks, n_iter=3)
        if n_blocks == 1:
            break
    return sum(blocks)


# ----------------------------------------------------------------------------- costs
def atom_cost(kind, shape, rank=None, n_blocks=1):
    Sn, N = int(sum(shape)), len(shape)
    if kind == "vectorized":
        return int(np.prod(shape))
    if kind == "cp":
        return int(rank * Sn)
    if kind == "tt":
        c = shape[0] * rank + shape[-1] * rank
        for nm in shape[1:-1]:
            c += rank * nm * rank
        return int(c)
    if kind == "tr":
        return int(rank ** 2 * Sn)
    if kind == "btd":
        return int(n_blocks * (rank ** N + rank * Sn))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- models
class StructuredDL:
    """Over-complete dictionary of K structurally-constrained tensor atoms."""

    def __init__(self, kind, atom_shape, n_atoms=64, rank=2, sparsity=4,
                 n_iter=8, seed=0, proj_kw=None):
        self.kind = kind
        self.atom_shape = tuple(atom_shape)
        self.K = int(n_atoms)
        self.rank = rank
        self.sparsity = sparsity
        self.n_iter = n_iter
        self.seed = seed
        self.proj_kw = dict(proj_kw or {})
        self.D = None

    # -- structure -----------------------------------------------------------
    def _project(self, A):
        if self.kind == "vectorized":
            return A
        if self.kind == "cp":
            return project_cp(A, self.rank)
        if self.kind == "tt":
            return project_tt(A, self.rank)
        if self.kind == "tr":
            return project_tr(A, self.rank)
        if self.kind == "btd":
            return project_btd(A, self.rank, self.proj_kw.get("n_blocks", 1))
        raise ValueError(self.kind)

    @property
    def n_atoms(self):
        return self.K

    @property
    def n_params(self):
        return self.K * atom_cost(self.kind, self.atom_shape, self.rank,
                                  self.proj_kw.get("n_blocks", 1))

    # -- fitting -------------------------------------------------------------
    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        M = X.shape[0]
        P = int(np.prod(self.atom_shape))
        Xf = X.reshape(M, P)
        idx = rng.choice(M, size=min(self.K, M), replace=self.K > M)
        A = X[idx]
        if self.K > M:
            A = np.concatenate([A, rng.standard_normal((self.K - M,) + self.atom_shape)])
        A = _unit(self._project(A))
        self.D = A.reshape(self.K, P)
        for _ in range(self.n_iter):
            C = self._omp(Xf, self.sparsity)                 # (K, M)
            G = C @ C.T + 1e-6 * np.eye(self.K)
            Dn = np.linalg.solve(G, C @ Xf)                  # MOD update
            A = _unit(self._project(Dn.reshape((self.K,) + self.atom_shape)))
            # revive dead atoms
            used = (np.abs(C) > 1e-9).sum(axis=1)
            dead = np.where(used == 0)[0]
            if len(dead):
                repl = X[rng.choice(M, size=len(dead))]
                A[dead] = _unit(self._project(repl))
            self.D = A.reshape(self.K, P)
        return self

    # -- coding --------------------------------------------------------------
    def _omp(self, Xf, s):
        s = int(min(s, self.K))
        Gram = self.D @ self.D.T
        Gram = Gram + 1e-8 * np.eye(self.K)
        Dty = self.D @ Xf.T
        return orthogonal_mp_gram(Gram, Dty, n_nonzero_coefs=s, copy_Gram=True,
                                  copy_Xy=True)

    def reconstruct(self, X, sparsity=None):
        s = self.sparsity if sparsity is None else sparsity
        Xf = X.reshape(X.shape[0], -1)
        C = self._omp(Xf, s)
        return (C.T @ self.D).reshape(X.shape)


class KroneckerDL:
    """Separable ('Tucker') dictionary: atoms = u1_a (x) u2_b (x) u3_c.

    K = prod(mode_ranks) atoms are described by only sum_n s_n * k_n parameters,
    which is exactly why this family behaves so differently under a parameter budget.
    """

    def __init__(self, atom_shape, mode_ranks, sparsity=4, n_iter=8, seed=0):
        self.atom_shape = tuple(atom_shape)
        self.ks = [int(k) for k in mode_ranks]
        self.sparsity = sparsity
        self.n_iter = n_iter
        self.seed = seed
        self.U = None
        self.D = None

    @property
    def n_atoms(self):
        return int(np.prod(self.ks))

    @property
    def n_params(self):
        return int(sum(s * k for s, k in zip(self.atom_shape, self.ks)))

    def _build(self):
        D = self.U[0]
        for n in range(1, len(self.ks)):
            D = (D[..., None, None] * self.U[n][None, None, :, :]) \
                if False else D
        # explicit Kronecker build (N=3)
        U0, U1, U2 = self.U
        D = np.einsum('ia,jb,kc->abcijk', U0, U1, U2)
        self.D = D.reshape(self.n_atoms, -1)
        self.D /= (np.linalg.norm(self.D, axis=1, keepdims=True) + EPS)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        N = X.ndim - 1
        # HOSVD init on the training set
        self.U = []
        for n in range(N):
            Xn = np.moveaxis(X, n + 1, 1).reshape(-1, self.atom_shape[n]).T
            u, _, _ = np.linalg.svd(Xn @ Xn.T)
            k = self.ks[n]
            Un = u[:, :k]
            if Un.shape[1] < k:                       # pad with random directions
                Un = np.concatenate(
                    [Un, rng.standard_normal((self.atom_shape[n], k - Un.shape[1]))], 1)
            self.U.append(Un / (np.linalg.norm(Un, axis=0, keepdims=True) + EPS))
        self._build()
        M = X.shape[0]
        Xf = X.reshape(M, -1)
        for _ in range(self.n_iter):
            C = self._omp(Xf, self.sparsity).T.reshape((M,) + tuple(self.ks))
            for n in range(N):
                T = C
                for m in range(N):
                    if m == n:
                        continue
                    T = np.moveaxis(np.tensordot(T, self.U[m], axes=([m + 1], [1])),
                                    -1, m + 1)
                Bn = np.moveaxis(T, n + 1, 1).reshape(M, self.ks[n], -1)
                Xn = np.moveaxis(X, n + 1, 1).reshape(M, self.atom_shape[n], -1)
                S = np.einsum('msp,mkp->sk', Xn, Bn)
                Gk = np.einsum('mkp,mlp->kl', Bn, Bn) + 1e-6 * np.eye(self.ks[n])
                Un = np.linalg.solve(Gk, S.T).T
                self.U[n] = Un / (np.linalg.norm(Un, axis=0, keepdims=True) + EPS)
            self._build()
        return self

    def _omp(self, Xf, s):
        s = int(min(s, self.n_atoms))
        Gram = self.D @ self.D.T + 1e-8 * np.eye(self.n_atoms)
        return orthogonal_mp_gram(Gram, self.D @ Xf.T, n_nonzero_coefs=s)

    def reconstruct(self, X, sparsity=None):
        s = self.sparsity if sparsity is None else sparsity
        C = self._omp(X.reshape(X.shape[0], -1), s)
        return (C.T @ self.D).reshape(X.shape)


def make_model(kind, atom_shape, **kw):
    if kind == "tucker":
        return KroneckerDL(atom_shape, kw.pop("mode_ranks"), **kw)
    return StructuredDL(kind, atom_shape, **kw)
