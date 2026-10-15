"""
GradTensor: Tensor Dictionary Learning with Sparse Tucker Decomposition
========================================================================

Implementation of the algorithm from:
    Zubair, S. & Wang, W. (2013). "Tensor Dictionary Learning with Sparse
    Tucker Decomposition." DSP2013.

The model factorises a tensor Y as

        Y  ~=  X  x_1 A  x_2 B  x_3 C

where A, B, C are mode-n dictionaries (factor matrices) and X is a SPARSE
core tensor. Sparsity in X is enforced via Tensor-OMP (a Kronecker-basis
extension of OMP). A, B, C are learned by alternating gradient descent.

Author: Demo implementation based on the paper.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding (matricisation) of a tensor.

    Uses the Kolda & Bader convention, which is the standard for Tucker
    decompositions: mode `mode` becomes the rows; the remaining modes
    are stacked as columns in reverse order (..., mode+2, mode+1, mode-1, ..., 0).

    Parameters
    ----------
    tensor : ndarray of shape (I_0, I_1, ..., I_{N-1})
    mode   : int in [0, N-1]

    Returns
    -------
    Y_(mode) : ndarray of shape (I_mode, prod(I_k for k != mode))
    """
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(matrix: np.ndarray, mode: int, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of `unfold`: reconstruct a tensor from its mode-n matrix."""
    full_shape = [shape[mode]] + [shape[k] for k in range(len(shape)) if k != mode]
    return np.moveaxis(matrix.reshape(full_shape), 0, mode)


def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """The n-mode product  T x_n M  of a tensor with a matrix.

    If T has shape (I_0, ..., I_{n}, ..., I_{N-1}) and M has shape (J, I_n),
    the result has shape (I_0, ..., J, ..., I_{N-1}).
    """
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return fold(matrix @ unfold(tensor, mode), mode, tuple(new_shape))


def tucker_to_tensor(core: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    """Reconstruct  X x_1 A x_2 B x_3 C ... from the core and factor list."""
    out = core
    for n, U in enumerate(factors):
        out = mode_n_product(out, U, n)
    return out


# ---------------------------------------------------------------------------
# Tensor Orthogonal Matching Pursuit (TOMP)
# ---------------------------------------------------------------------------

def tensor_omp(
    Y: np.ndarray,
    factors: list[np.ndarray],
    max_nnz: int,
    tol: float = 1e-6,
) -> np.ndarray:
    """Tensor-OMP for Tucker / Kronecker dictionaries.

    Solves (greedily)

        min_X || Y - X x_1 A x_2 B x_3 C ||_F^2     s.t.  ||X||_0 <= max_nnz

    by exploiting the equivalence

        vec(Y) = (C kron B kron A) vec(X).

    Parameters
    ----------
    Y       : data tensor of shape (I_0, I_1, ..., I_{N-1}).
    factors : list of N factor matrices, factors[n] has shape (I_n, M_n).
    max_nnz : maximum number of non-zero core entries.
    tol     : residual Frobenius-norm threshold.

    Returns
    -------
    X : sparse core tensor of shape (M_0, M_1, ..., M_{N-1}).
    """
    N = Y.ndim
    core_shape = tuple(U.shape[1] for U in factors)
    X = np.zeros(core_shape)

    # Pre-normalise factor columns for atom selection (the inner product picks
    # the atom with largest correlation; normalised columns make this fair).
    factors_norm = []
    for U in factors:
        col_norms = np.linalg.norm(U, axis=0)
        col_norms[col_norms == 0] = 1.0
        factors_norm.append(U / col_norms)

    R = Y.copy()                       # residual tensor
    selected: set[tuple[int, ...]] = set()

    for _ in range(max_nnz):
        if np.linalg.norm(R) <= tol:
            break

        # --- atom selection -------------------------------------------------
        # Compute  R x_1 A^T x_2 B^T x_3 C^T  -> a tensor of correlations,
        # one entry per (m_1, ..., m_N) candidate atom in the Kronecker basis.
        corr = R
        for n, Un in enumerate(factors_norm):
            corr = mode_n_product(corr, Un.T, n)
        # Pick the largest |corr| coordinate, skipping any already chosen.
        order = np.argsort(np.abs(corr).ravel())[::-1]
        chosen = None
        for flat in order:
            idx = np.unravel_index(flat, corr.shape)
            if idx not in selected:
                chosen = idx
                break
        if chosen is None:
            break
        selected.add(chosen)

        # --- least-squares re-fit on the active set -------------------------
        # Build the sub-dictionary made of the selected Kronecker atoms and
        # solve for the optimal coefficients (active-set OMP refit step).
        #
        # With our unfolding convention (np.moveaxis + reshape), the matrix
        # identity is  Y_(0) = A · X_(0) · (B ⊗ C ⊗ ...)^T  — i.e. forward
        # mode order — which corresponds to the vec identity
        #     vec_C(Y) = (A ⊗ B ⊗ C ⊗ ...) vec_C(X).
        # So atoms are kron(A_col, kron(B_col, kron(C_col, ...))) in forward
        # order and we use C-order vec.
        # order and we use C-order vec.
        cols = []
        idx_list = list(selected)
        for idx in idx_list:
            atom = factors[N - 1][:, idx[N - 1]]
            for n in range(N - 2, -1, -1):
                atom = np.kron(factors[n][:, idx[n]], atom)
            cols.append(atom)
        D_active = np.stack(cols, axis=1)            # (prod I_n, |selected|)
        y_vec = Y.reshape(-1, order='C')
        coeffs, *_ = np.linalg.lstsq(D_active, y_vec, rcond=None)

        # --- write coefficients back into X and recompute residual ---------
        X = np.zeros(core_shape)
        for c, idx in zip(coeffs, idx_list):
            X[idx] = c
        R = Y - tucker_to_tensor(X, factors)

    return X


# ---------------------------------------------------------------------------
# GradTensor: alternating gradient descent on factors
# ---------------------------------------------------------------------------

def _init_factors(Y: np.ndarray, core_shape: tuple[int, ...]) -> list[np.ndarray]:
    """Initialise each factor with the M_n leading left singular vectors of
    the mode-n unfolding of Y (HOSVD-style warm start, as recommended in
    section 4 of the paper)."""
    factors = []
    for n in range(Y.ndim):
        Yn = unfold(Y, n)
        # Truncated SVD via numpy. For very large tensors, swap in randomized SVD.
        U, _, _ = np.linalg.svd(Yn, full_matrices=False)
        Mn = core_shape[n]
        if U.shape[1] < Mn:
            # Pad with random orthonormal columns when M_n > rank(Y_(n)).
            extra = np.random.randn(U.shape[0], Mn - U.shape[1])
            extra, _ = np.linalg.qr(extra)
            U = np.concatenate([U, extra], axis=1)
        factors.append(U[:, :Mn].copy())
    return factors


def _gradient_step(
    Y: np.ndarray,
    X: np.ndarray,
    factors: list[np.ndarray],
    mode: int,
    step: float,
    inner_tol: float,
    max_inner: int,
) -> np.ndarray:
    """Gradient descent on the `mode`-th factor with all others held fixed.

    The objective along mode n is

        || Y_(n) - U_n  X_(n)  (kron of the other U_k)^T ||_F^2,

    and the gradient (per the paper, eq. 8) is

        ( Y_(n) - U_n M )  M^+    where  M = X_(n) (kron of others)^T.

    The paper writes this with a pseudo-inverse so the "gradient" is really
    a Newton-style step on a quadratic — which is why a single iteration is
    typically sufficient and the inner loop converges in 1-2 passes.
    """
    N = Y.ndim
    # Forward mode order: for mode n, the unfolding identity is
    #     Y_(n) = U_n · X_(n) · (kron of factors[k] for k != n, in order)^T.
    # So we kron the *other* factors in increasing-index order.
    other_modes = [k for k in range(N) if k != mode]
    kron_others = factors[other_modes[0]]
    for k in other_modes[1:]:
        kron_others = np.kron(kron_others, factors[k])

    Yn = unfold(Y, mode)
    Xn = unfold(X, mode)
    M = Xn @ kron_others.T                 # what U_mode multiplies; shape (M_n, prod_others)

    # When X is sparse, most rows of X_(n) are zero, so M has many zero rows
    # and is severely rank-deficient. Solving U M = Y_(n) via pseudo-inverse
    # blows up unless we threshold small singular values. Solve the least-
    # squares problem U_new = argmin || Y_(n) - U M ||_F via lstsq on M^T.
    #
    #   minimise || Y_(n) - U M ||_F^2  <=>  minimise || M^T U^T - Y_(n)^T ||_F^2
    #
    # numpy.linalg.lstsq handles rank deficiency cleanly with rcond.
    U = factors[mode].copy()
    prev_err = np.inf
    for _ in range(max_inner):
        # Closed-form least-squares solution for U with all else fixed.
        U_new_T, *_ = np.linalg.lstsq(M.T, Yn.T, rcond=1e-8)
        U_new = U_new_T.T
        # Optional damping toward old U (step < 1) for stability across modes.
        U = (1.0 - step) * U + step * U_new
        err = np.linalg.norm(Yn - U @ M, 'fro')
        if abs(prev_err - err) <= inner_tol:
            break
        prev_err = err
    return U


def grad_tensor(
    Y: np.ndarray,
    core_shape: tuple[int, ...],
    sparsity: int,
    step: float = 1.0,
    max_outer: int = 50,
    max_inner: int = 1,
    outer_tol: float = 1e-4,
    inner_tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Learn a sparse Tucker decomposition Y ~= X x_1 A x_2 B x_3 C.

    Parameters
    ----------
    Y          : data tensor.
    core_shape : tuple of M_n giving the size of the core along each mode.
                 The paper takes M_n >= I_n (overcomplete dictionaries).
    sparsity   : total number of non-zeros allowed in the core tensor X.
    step       : gradient step size (gamma in the paper).
    max_outer  : cap on alternating sparse-coding / dictionary-update sweeps.
    max_inner  : cap on inner gradient-descent iterations per factor update.
    outer_tol  : stop when ||Y - reconstruction||_F drops below this.
    inner_tol  : stop the inner loop when successive errors differ by < this.
    verbose    : print the residual at each outer iteration.

    Returns
    -------
    X       : sparse core tensor.
    factors : list [A, B, C, ...] of learned mode-n dictionaries.
    history : list of reconstruction errors, one per outer iteration.
    """
    if Y.ndim != len(core_shape):
        raise ValueError("core_shape must have one entry per mode of Y")

    factors = _init_factors(Y, core_shape)
    history = []

    for it in range(max_outer):
        # 1. Sparse coding stage: update X via Tensor-OMP.
        X = tensor_omp(Y, factors, max_nnz=sparsity, tol=outer_tol)

        # 2. Dictionary update stage: gradient descent on each factor in turn.
        for n in range(Y.ndim):
            factors[n] = _gradient_step(
                Y, X, factors, mode=n,
                step=step, inner_tol=inner_tol, max_inner=max_inner,
            )
            # Normalise columns of the updated factor and absorb the scales
            # into the core along mode n. This is essential for stability:
            # without it, a column of U_n can grow unboundedly while the
            # matching slice of X shrinks (and vice versa), producing the
            # numerical blow-up described in classical Tucker ALS.
            col_norms = np.linalg.norm(factors[n], axis=0)
            col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
            factors[n] = factors[n] / col_norms
            # Multiply the corresponding mode-n slices of X by col_norms.
            shape = [1] * X.ndim
            shape[n] = -1
            X = X * col_norms.reshape(shape)

        err = np.linalg.norm(Y - tucker_to_tensor(X, factors))
        history.append(err)
        if verbose:
            print(f"  iter {it+1:3d}  ||Y - Y_hat||_F = {err:.6f}")
        if err <= outer_tol:
            break

    return X, factors, history


# ---------------------------------------------------------------------------
# Demo / sanity check on synthetic data (mirrors §5.1 of the paper)
# ---------------------------------------------------------------------------

def _demo() -> None:
    """A small reconstruction demo on synthetic data.

    NOTE on regimes
    ---------------
    The paper studies the *overcomplete* case M_n > I_n, which makes the
    sparse-recovery problem genuinely hard: with random initialisation,
    block-coordinate methods like this one routinely get stuck in local
    minima. The paper sidesteps this by warm-starting from HOSVD (which
    we also do) but their reported convergence is on the *objective value*,
    not on recovering the true generating decomposition.

    This demo uses the easier *undercomplete* (compression) regime
    M_n <= I_n on a low-rank-ish tensor, where convergence is reliable
    and easy to inspect. Section 5.2 of the paper (image reconstruction)
    is essentially this regime.
    """
    rng = np.random.default_rng(0)

    I = (15, 15, 15)
    M = (8, 8, 8)               # undercomplete: classical Tucker compression
    s_total = 60                # ~12% of |core|

    # Build a ground-truth tensor from random orthonormal factors and a
    # sparse core, then add a little noise.
    A_true = np.linalg.qr(rng.standard_normal((I[0], M[0])))[0]
    B_true = np.linalg.qr(rng.standard_normal((I[1], M[1])))[0]
    C_true = np.linalg.qr(rng.standard_normal((I[2], M[2])))[0]
    X_true = np.zeros(M)
    nnz_idx = rng.choice(np.prod(M), size=s_total, replace=False)
    X_true.flat[nnz_idx] = rng.standard_normal(s_total)

    Y = tucker_to_tensor(X_true, [A_true, B_true, C_true])
    Y = Y + 0.01 * np.linalg.norm(Y) / np.sqrt(Y.size) * rng.standard_normal(Y.shape)

    print(f"Y shape {Y.shape}, core shape {M}, target sparsity {s_total}")
    print(f"||Y||_F = {np.linalg.norm(Y):.4f}")

    X_hat, factors, history = grad_tensor(
        Y, core_shape=M, sparsity=s_total,
        step=1.0, max_outer=30, verbose=True,
    )

    Y_hat = tucker_to_tensor(X_hat, factors)
    rel_err = np.linalg.norm(Y - Y_hat) / np.linalg.norm(Y)
    print(f"\nFinal relative reconstruction error: {rel_err:.4%}")
    print(f"Non-zeros in recovered core: {(np.abs(X_hat) > 1e-8).sum()} / {s_total}")


if __name__ == "__main__":
    _demo()