"""
legacy_tucker.py -- a reconstruction of the ORIGINAL TuckerDL, kept only so the
notebook can *demonstrate* the objective-mismatch failure instead of asserting it.

This is NOT the shipped model. The shipped TuckerDL (tensor_dl.TuckerDL) sits inside
the shared harness: OMP against the Kronecker dictionary, then separable ALS.

The old one did something different, and that difference is the whole lesson:
it learned each mode dictionary U_n by running a *separate* 1-D MOD-style dictionary
learning pass on the mode-n fibers of the training patches, with its own sparsity
setting, never touching the shared OMP-on-whole-patches objective. It then built
D = U_3 (x) U_2 (x) U_1 once and stopped.

So the dictionary was fit to "reconstruct raw unfolded fibers" and later graded on
"sparse-code whole patches" -- two different objectives.
"""
from __future__ import annotations
import time
import numpy as np
from numpy.linalg import norm, svd, pinv
from sklearn.linear_model import orthogonal_mp
import tensorly as tl

from tensor_dl import BaseTensorDL, _normalize_cols, EPS


class LegacyTuckerDL(BaseTensorDL):
    """Original Tucker: per-mode MOD on unfolded fibers, run once, outside the harness."""
    name = "Tucker (legacy)"

    def __init__(self, atom_shape, mode_ranks=None, n_iter=12,
                 mode_sparsity=2, seed=0, max_fibers=4000):
        super().__init__(atom_shape, n_atoms=None, n_iter=n_iter, seed=seed)
        if mode_ranks is None:
            mode_ranks = [d + 1 for d in atom_shape]
        self.mode_ranks = list(mode_ranks)
        self.mode_sparsity = mode_sparsity     # its OWN sparsity, unrelated to encode-time s
        self.max_fibers = max_fibers           # it subsampled fibers "for speed"
        self.U_ = None
        self.degeneracy_ = None

    def _build_Deff(self, U):
        N = len(self.atom_shape)
        big, sub = "ijklm"[:N], "abcde"[:N]
        eq = ",".join(f"{a}{b}" for a, b in zip(big, sub)) + "->" + big + sub
        return np.einsum(eq, *U).reshape(self.n_features, -1)

    def _mod_1d(self, fibers, k, rng):
        """MOD dictionary learning on a set of 1-D fibers. fibers: (n_mode, n_fibers)."""
        n_mode, n_fib = fibers.shape
        # subsample fibers for speed
        if n_fib > self.max_fibers:
            fibers = fibers[:, rng.choice(n_fib, self.max_fibers, replace=False)]
        # init from data columns
        idx = rng.choice(fibers.shape[1], size=min(k, fibers.shape[1]), replace=False)
        U = fibers[:, idx].copy()
        if U.shape[1] < k:
            U = np.hstack([U, rng.standard_normal((n_mode, k - U.shape[1])) * 0.1])
        U = _normalize_cols(U)
        s = min(self.mode_sparsity, k)
        for _ in range(self.n_iter):
            C = orthogonal_mp(U, fibers, n_nonzero_coefs=s)      # (k, n_fibers)
            U = fibers @ pinv(C)                                  # MOD least-squares update
            U = _normalize_cols(U)
        return U

    def fit(self, X):
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)
        N = len(self.atom_shape)
        U = []
        for mode in range(N):
            # mode-n fibers of the whole training stack, sample axis moved to the back
            fibers = tl.unfold(np.moveaxis(X, 0, -1), mode)       # (n_mode, rest)
            U.append(self._mod_1d(fibers, self.mode_ranks[mode], rng))
        # build the Kronecker dictionary ONCE, then stop. No feedback from whole-patch coding.
        self.U_ = U
        self.D_ = _normalize_cols(self._build_Deff(U))
        self.n_atoms = self.D_.shape[1]
        self._param_count = int(sum(u.size for u in U))
        self.fit_time_ = time.time() - t0
        return self
