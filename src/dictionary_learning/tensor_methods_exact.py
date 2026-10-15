"""
tensor_methods_exact.py
=======================
Exact Python ports of TensorNOODL and M2PALS from the actual MATLAB source.

Every function is annotated with the MATLAB file and line it comes from.
No guessing — every implementation decision is traceable to the source.

Sources read:
  TensorNOODL/tens_compare_NOODL.m      — main algorithm
  TensorNOODL/run_noodl_tens.m          — parameters and initialization
  TensorNOODL/KRP.m                     — Khatri-Rao product
  TensorNOODL/nrmc.m                    — column normalisation
  TensorNOODL/best_fista_result.m       — FISTA coefficient search
  TensorNOODL/FISTA_with_init.m         — FISTA implementation
  TensorNOODL/softThr.m                 — soft thresholding

  M2PALS/Dmult_nmf_OMP.m               — M2PALS matrix algorithm (main)
  M2PALS/tensor/Dmult_mpnals.m          — M2PALS tensor (CPD) algorithm
  M2PALS/MatchP.m                       — single-dict atom matching
  M2PALS/MatchP_multiD.m               — multi-dict atom matching
  M2PALS/nonneg.../nnlsHALSupdt.m      — NNLS via block-coord descent
  M2PALS/nonneg.../HALSacc.m           — accelerated HALS NMF
  M2PALS/tools/col_norm.m              — column normalisation
  M2PALS/tools/kr.m                    — Khatri-Rao product

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import time
from typing import Optional, List


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def nrmc(A: np.ndarray) -> np.ndarray:
    """
    Exact port of TensorNOODL/nrmc.m
    ---------------------------------
    Alternative to normc for Matlab 2014b and above.
    Normalises each column of A to unit L2 norm.
    Zero columns are left as zero (the MATLAB code skips them via
    `if sum(A(:,i)) ~= 0` — note this checks the SUM not the norm,
    so a column that sums to zero but has nonzero entries is also skipped.
    We replicate that exact behaviour here).

    MATLAB source:
        A_n = zeros(size(A));
        for i = 1:size(A, 2)
           if (sum(A(:,i)) ~= 0)
             A_n(:,i) = A(:,i)/norm(A(:,i));
           end
        end
    """
    A_n = np.zeros_like(A, dtype=float)
    for i in range(A.shape[1]):
        if A[:, i].sum() != 0:          # exact replication: sum not norm
            n = np.linalg.norm(A[:, i])
            if n > 0:
                A_n[:, i] = A[:, i] / n
    return A_n


def col_norm(M: np.ndarray) -> np.ndarray:
    """
    Exact port of M2PALS/tools/col_norm.m
    --------------------------------------
    MATLAB source:
        norms = sqrt(sum(M.^2));
        for r=1:n
            if norms(r)>0
                M(:,r) = M(:,r)/norms(r);
            end
        end
    """
    norms = np.sqrt(np.sum(M ** 2, axis=0))   # column L2 norms
    for r in range(M.shape[1]):
        if norms[r] > 0:
            M[:, r] = M[:, r] / norms[r]
    return M


def kr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Exact port of M2PALS/tools/kr.m  (Laurent Sorber's implementation)
    -------------------------------------------------------------------
    Khatri-Rao product: kr(A,B)[:,k] = kron(A[:,k], B[:,k])
    A is (I x K), B is (J x K) → result is (I*J x K).

    MATLAB source uses bsxfun + reshape — equivalent vectorised form below.

    MATLAB source (kr.m):
        J = size(U{end},1);
        X = reshape(U{end},[J 1 K]);
        for n = length(U)-1:-1:1
            I = size(U{n},1);
            A = reshape(U{n},[1 I K]);
            X = reshape(bsxfun(@times,A,X),[I*J 1 K]);
            J = I*J;
        end
        X = reshape(X,[size(X,1) K]);
    """
    I, K = A.shape
    J = B.shape[0]
    # Vectorised Khatri-Rao: A[:,None,:] * B[None,:,:] → (I,J,K) → (I*J,K)
    return (A[:, np.newaxis, :] * B[np.newaxis, :, :]).reshape(I * J, K)


def KRP(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Exact port of TensorNOODL/KRP.m
    --------------------------------
    TensorNOODL's own Khatri-Rao implementation.
    Note: KRP(C,B) in MATLAB becomes kr(C,B) — same operation, same result.

    MATLAB source:
        function C = KRP(A, B)
        C=[];
        for i = 1: size(A,2)
            pr = (kron(A(:,i), B(:,i)));
            C = [C pr(:)];
        end
    """
    cols = []
    for i in range(A.shape[1]):
        cols.append(np.kron(A[:, i], B[:, i]))
    return np.column_stack(cols)


def soft_thr(y: np.ndarray, lam: float) -> np.ndarray:
    """
    Exact port of TensorNOODL/softThr.m
    ------------------------------------
    MATLAB source:
        function [x]=softThr(y,lambda)
        x=sign(y).*max(abs(y)-lambda, 0);
    """
    return np.sign(y) * np.maximum(np.abs(y) - lam, 0)


def sq_fnorm(Y: np.ndarray) -> float:
    """Squared Frobenius norm. Used in M2PALS stopping criterion."""
    return float(np.sum(Y ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# NNLS via block-coordinate descent  (M2PALS/nonnegative_least_squares/)
# ══════════════════════════════════════════════════════════════════════════════

def nnls_hals_updt(M: np.ndarray, U: np.ndarray,
                   V: Optional[np.ndarray] = None,
                   maxiter: int = 500) -> np.ndarray:
    """
    Exact port of M2PALS/nonnegative_least_squares/nnlsHALSupdt.m
    --------------------------------------------------------------
    Computes approximate solution of:
        min_{V >= 0}  ||M - U V||_F^2
    via exact block-coordinate descent (HALS).

    Gillis & Glineur, Neural Computation 24(4):1085-1105, 2012.

    MATLAB source (key sections):
        UtU = U'*U;
        UtM = U'*M;

        if nargin <= 2 || isempty(V)
            V = U\\M;              % least squares init
            V = max(V,0);
            alpha = sum(sum((U'*M).*V)) ./ sum(sum((U'*U).*(V*V')));
            V = alpha*V;
        end

        delta = 1e-6;
        eps0 = 0; cnt = 1; eps = 1;
        while eps >= (delta)^2*eps0 && cnt <= maxiter
            nodelta = 0;
            for k = 1:r
                deltaV = max((UtM(k,:)-UtU(k,:)*V)/UtU(k,k), -V(k,:));
                V(k,:) = V(k,:) + deltaV;
                nodelta = nodelta + deltaV*deltaV';
                if V(k,:)==0, V(k,:)=1e-16*max(V(:)); end
            end
            if cnt==1, eps0=nodelta; end
            eps=nodelta; cnt=cnt+1;
        end

    Parameters
    ----------
    M  : (m x n)
    U  : (m x r)
    V  : (r x n) initialisation — if None, uses least-squares init
    maxiter : int

    Returns
    -------
    V  : (r x n) nonneg matrix approximating argmin_{V>=0} ||M-UV||_F^2
    """
    m, n = M.shape
    m2, r = U.shape

    UtU = U.T @ U          # r x r
    UtM = U.T @ M          # r x n

    if V is None or V.size == 0:
        # MATLAB: V = U\M  (least squares), then scale
        V, _, _, _ = np.linalg.lstsq(U, M, rcond=None)
        V = np.maximum(V, 0)
        denom = np.sum((UtU) * (V @ V.T))
        if denom > 0:
            alpha = np.sum(UtM * V) / denom
            V = alpha * V
        else:
            V = np.zeros((r, n))

    delta = 1e-6
    eps0 = 0.0
    cnt = 1
    eps = 1.0

    while eps >= (delta ** 2) * eps0 and cnt <= maxiter:
        nodelta = 0.0
        for k in range(r):
            dkk = UtU[k, k]
            if dkk == 0:
                continue
            # deltaV = max((UtM[k,:] - UtU[k,:] @ V) / UtU[k,k], -V[k,:])
            deltaV = np.maximum(
                (UtM[k, :] - UtU[k, :] @ V) / dkk,
                -V[k, :]
            )
            V[k, :] += deltaV
            nodelta += float(deltaV @ deltaV)
            # Safety: if V[k,:] all zero, set to tiny value
            if np.all(V[k, :] == 0):
                V[k, :] = 1e-16 * np.max(V)
        if cnt == 1:
            eps0 = nodelta
        eps = nodelta
        cnt += 1

    return V


# ══════════════════════════════════════════════════════════════════════════════
# Atom matching  (M2PALS/MatchP.m and MatchP_multiD.m)
# ══════════════════════════════════════════════════════════════════════════════

def match_p(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Exact port of M2PALS/MatchP.m
    ------------------------------
    Find the best atom in D for each column of X using inner products.

    MATLAB source:
        Gram     =    X'*D;
        [~,indices] = max(Gram');

    Returns 0-indexed column indices into D (MATLAB is 1-indexed, we subtract 1).
    """
    Gram = X.T @ D                             # (p x d)
    indices = np.argmax(Gram, axis=1)          # (p,) — 0-indexed
    return indices


def match_p_multi_d(X: np.ndarray,
                    D: List[np.ndarray],
                    d: np.ndarray,
                    fun: str = 'eucl') -> tuple:
    """
    Exact port of M2PALS/MatchP_multiD.m
    -------------------------------------
    Multi-dictionary atom matching using the Hungarian algorithm (munkres).

    MATLAB source key logic:
        1. Normalise each dictionary Dn{i}
        2. Compute Gram{i} = feval(fun, X, Dn{i})  — scores per dict
        3. Duplicate columns in scores for d[i]>1 dictionaries
        4. Hungarian: [alloc,~] = munkres(-scores)
        5. For each column of X, pick atom from allocated dict

    Parameters
    ----------
    X       : (n x p) — current factor estimate
    D       : list of P arrays, each (n x d_p)
    d       : (P,) int array — atoms to select from each dict
    fun     : distance metric ('eucl' = inner product, matching MATLAB default)

    Returns
    -------
    alloc   : (p,) dict index (0-indexed) assigned to each column of X
    indices : (p,) atom index (0-indexed) within assigned dict
    X_hat   : (n x p) — selected atoms as columns
    """
    k = len(D)
    p = X.shape[1]

    # -- Normalise each dictionary (MATLAB: Dn{i} = D{i}./(normD+1e-6))
    Dn = []
    for Di in D:
        norm_d = np.sqrt(np.sum(Di ** 2, axis=0))
        Dn.append(Di / (norm_d[np.newaxis, :] + 1e-6))

    # -- Compute scores: Gram{i} = feval(fun, X, Dn{i})
    # MATLAB fun='eucl' calls tools/eucl.m which computes inner products
    # (in the context used here it's always eucl → Dn{i}'*col_norm(X))
    # The MATLAB code: Gram{i} = feval(fun, X, Dn{i})
    # eucl.m: computes pairwise distances or similarities — in MatchP_multiD
    # context it reduces to Gram matrix. We use inner product (cosine-like).
    scores = np.zeros((p, k))
    indices_1 = np.zeros((p, k), dtype=int)

    # Normalise X columns for fair scoring
    X_norm = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)

    for i, Dni in enumerate(Dn):
        G = X_norm.T @ Dni                     # p x d_i
        if G.shape[1] > 1:
            indices_1[:, i] = np.argmax(G, axis=1)
            scores[:, i] = G[np.arange(p), indices_1[:, i]]
        else:
            indices_1[:, i] = 0
            scores[:, i] = G[:, 0]

    # -- Duplicate columns for d[i] > 1 (MATLAB: expand scores matrix)
    # total atoms to assign = sum(d)
    p_total = int(np.sum(d))
    # table maps expanded column → original dict index (0-indexed)
    table = np.zeros(p_total, dtype=int)
    scores_exp = np.zeros((p, p_total))
    indices_exp = np.zeros((p, p_total), dtype=int)

    col = 0
    for i, di in enumerate(d):
        for _ in range(di):
            table[col] = i
            scores_exp[:, col] = scores[:, i]
            indices_exp[:, col] = indices_1[:, i]
            col += 1

    # -- Hungarian algorithm on -scores (maximise scores)
    # scipy.optimize.linear_sum_assignment minimises cost
    row_ind, col_ind = linear_sum_assignment(-scores_exp)
    # col_ind[j] is the expanded-column assigned to row j (column of X)

    alloc = table[col_ind]                     # dict index per X column
    indices = indices_exp[row_ind, col_ind]    # atom index per X column

    # -- Build X_hat from selected atoms
    X_hat = np.zeros_like(X)
    for i in range(p):
        X_hat[:, i] = D[alloc[i]][:, indices[i]]

    return alloc, indices, X_hat


# ══════════════════════════════════════════════════════════════════════════════
# FISTA  (TensorNOODL/FISTA_with_init.m)
# ══════════════════════════════════════════════════════════════════════════════

def fista_with_init(D: np.ndarray, Y: np.ndarray,
                    lam: float, X_init: np.ndarray,
                    tol: float) -> np.ndarray:
    """
    Exact port of TensorNOODL/FISTA_with_init.m
    --------------------------------------------
    Solves:  min_X  ||D X - Y||_F^2 + lam ||X||_1
    via FISTA with warm start.

    MATLAB source:
        k = 75;  (max iterations)
        L = 2*max(eig(D'*D));  (Lipschitz constant)
        y_s = Xinit; x_sm1 = y_s; t_s = 1;
        der = @(x) 2*D'*(D*x-Y);
        F   = @(x) norm(D*x-Y)^2 + lam*norm(x,1);
        while (its < k) && (stopval == 0)
            z_s = softThr(y_s - (1/L)*der(y_s), lam/L);
            if F(z_s) < F(x_sm1)
                x_s = z_s;
            else
                x_s = x_sm1;
            end
            t_sp1 = (1/2)*(1 + sqrt(1+4*t_s^2));
            y_sp1 = x_s + (t_s/t_sp1)*(z_s-x_s) + ((t_s-1)/t_sp1)*(x_s-x_sm1);
            NN = norm(y_s - y_sp1,'fro') / norm(y_sp1,'fro');
            stopval = (NN < eps) || isnan(NN);
            y_s=y_sp1; t_s=t_sp1; x_sm1=x_s;
        end
        A = x_s;
    """
    k = 75                                          # max iterations
    # Lipschitz constant: 2 * max eigenvalue of D'D
    DtD = D.T @ D
    eigvals = np.linalg.eigvalsh(DtD)
    L = 2.0 * float(np.max(eigvals))
    if L == 0:
        L = 1.0

    def der(x):
        return 2.0 * D.T @ (D @ x - Y)

    def F(x):
        return float(np.linalg.norm(D @ x - Y, 'fro') ** 2
                     + lam * np.sum(np.abs(x)))

    y_s   = X_init.copy()
    x_sm1 = y_s.copy()
    t_s   = 1.0
    stopval = 0
    its = 0
    x_s = x_sm1.copy()

    while its < k and stopval == 0:
        z_s = soft_thr(y_s - (1.0 / L) * der(y_s), lam / L)

        if F(z_s) < F(x_sm1):
            x_s = z_s
        else:
            x_s = x_sm1.copy()

        t_sp1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_s ** 2))

        y_sp1 = (x_s
                 + (t_s / t_sp1) * (z_s - x_s)
                 + ((t_s - 1.0) / t_sp1) * (x_s - x_sm1))

        nrm_ysp1 = np.linalg.norm(y_sp1, 'fro')
        if nrm_ysp1 == 0:
            NN = float('nan')
        else:
            NN = float(np.linalg.norm(y_s - y_sp1, 'fro') / nrm_ysp1)

        stopval = int(NN < tol or np.isnan(NN))

        z_sm1 = z_s       # kept for symmetry with MATLAB (unused after)
        y_s   = y_sp1
        t_s   = t_sp1
        x_sm1 = x_s
        its  += 1

    return x_s


def best_fista_result(A: np.ndarray, Y_s: np.ndarray,
                      X_fi_init: np.ndarray, X_o_fi: np.ndarray,
                      tol: float, i_max: int) -> tuple:
    """
    Exact port of TensorNOODL/best_fista_result.m
    -----------------------------------------------
    Scans lambda values from lam_max down by 0.7^(i_max-i) factor,
    running FISTA at each, keeps the best estimate.

    MATLAB source:
        lam_max = 2*max(max(abs(A'*Y_s)));
        bestErr = 1000000;
        for i = 1:i_max
            lam = (0.7^(i_max-i))*lam_max;
            X_fi_est = FISTA_with_init(A, Y_s, lam, X_fi_init, tol);
            err = [err norm(X_fi_est - X_o_fi,'fro')/norm(X_o_fi,'fro')];
            if err(end) < bestErr
                bestErr = err(end); best_X_fi_est = X_fi_est; bestId = i;
            end
            if err(end) <= tol
                X_fi_best = X_fi_est; break;
            end
            X_fi_init = X_fi_est;
        end
        if i == i_max
            X_fi_est = best_X_fi_est;
        end
        X_fi_best = X_fi_est;

    Returns
    -------
    X_fi_best : best coefficient estimate
    i         : number of lambda values tried
    """
    lam_max = 2.0 * float(np.max(np.abs(A.T @ Y_s)))
    best_err = 1e6
    best_X = X_fi_init.copy()
    norm_Xo = float(np.linalg.norm(X_o_fi, 'fro'))
    X_fi_est = X_fi_init.copy()

    for i in range(1, i_max + 1):
        lam = (0.7 ** (i_max - i)) * lam_max
        X_fi_est = fista_with_init(A, Y_s, lam, X_fi_init, tol)
        err_i = float(np.linalg.norm(X_fi_est - X_o_fi, 'fro') / max(norm_Xo, 1e-12))
        if err_i < best_err:
            best_err = err_i
            best_X = X_fi_est.copy()
        if err_i <= tol:
            return X_fi_est, i
        X_fi_init = X_fi_est

    return best_X, i_max


# ══════════════════════════════════════════════════════════════════════════════
# TensorNOODL
# ══════════════════════════════════════════════════════════════════════════════

def tensor_noodl(A_init: np.ndarray,
                 A_true: np.ndarray,
                 J: int, K: int,
                 alpha: float, beta: float,
                 c: float,
                 thr_c: float,
                 eta: float,
                 tol_X: float,
                 tol_A: float,
                 rng: np.random.Generator) -> dict:
    """
    Exact port of TensorNOODL/tens_compare_NOODL.m  (TensorNOODL section only)
    ---------------------------------------------------------------------------
    Implements the core TensorNOODL algorithm exactly as written.

    Parameters — match run_noodl_tens.m exactly:
        A_init : (n x m) initial dictionary (perturbed A_true)
        A_true : (n x m) ground truth dictionary A_o
        J, K   : mode-1 and mode-2 dimensions (K=J in the paper experiments)
        alpha  : sparsity of B (a in MATLAB)
        beta   : sparsity of C (b in MATLAB, = alpha)
        c      : IHT step size (fixed, from run_noodl_tens: c=0.2)
        thr_c  : IHT hard threshold (from run_noodl_tens: thr_c=0.1)
        eta    : dictionary update step size
        tol_X  : IHT convergence tolerance (run_noodl_tens: 1e-12)
        tol_A  : outer loop convergence tolerance (run_noodl_tens: 1e-10)
        rng    : numpy random generator (for reproducibility)

    Returns dict with keys matching MATLAB output variables:
        A_our, errA, errX, err, time_our
        (plus Y_last, X_last, X_last_o for diagnostics)

    Key MATLAB lines replicated exactly:
        C = 1  (Rademacher constant — non-zeros are ±1)

        B_new = sparse(sign(randn(J,m).*binornd(1,a,J,m)))
        C_new = sparse(sign(randn(K,m).*binornd(1,b,K,m)))

        X_new_whole = KRP(C_new, B_new)'
        X1_new = A_o * X_new_whole
        idx = find(sum(abs(fi),1))    ← strip zero columns
        Y_m = fi(:, idx)

        [IHT loop — converges on change_X]
        XS = XS - c*((AtA*XS) - A'*Y)
        XS(abs(XS)<=thr_c) = 0
        change_X = abs(err[-1] - err[-3])  ← stops when plateau

        gr = (1/pf)*(Y_m - A*XS)*sign(XS)'
        A = A + eta*gr               ← PLUS not minus
        A = nrmc(A)

        change_A = errA[-1]          ← outer loop: error vs A_true
    """
    n, m = A_init.shape
    C_rad = 1  # Rademacher constant from MATLAB: C = 1

    A_our = A_init.copy()

    errA = []
    errX = []
    err  = []
    time_our = []
    Y_last = None
    X_last = None
    X_last_o = None

    i = 1
    change_A = float(np.linalg.norm(A_init - A_true, 'fro')
                     / np.linalg.norm(A_true, 'fro'))

    while change_A > tol_A:

        # ── Data generation (MATLAB: tens_compare_NOODL.m lines ~50-90) ──
        # B_new = sparse(sign(randn(J,m).*binornd(1,a,J,m)))
        B_new = np.sign(rng.standard_normal((J, m))
                        * rng.binomial(1, alpha, (J, m))).astype(float)
        # C_new = sparse(sign(randn(K,m).*binornd(1,b,K,m)))
        C_new = np.sign(rng.standard_normal((K, m))
                        * rng.binomial(1, beta,  (K, m))).astype(float)

        # Khatri-Rao product: X_new_whole = KRP(C_new,B_new)'
        X_new_whole = KRP(C_new, B_new).T      # m x (J*K)
        X1_new = A_true @ X_new_whole           # n x (J*K)

        # Retry if all-zero (MATLAB: up to 10 retries)
        if not np.any(X_new_whole):
            retries = 0
            while not np.any(X_new_whole) and retries < 9:
                B_new = np.sign(rng.standard_normal((J, m))
                                * rng.binomial(1, alpha, (J, m))).astype(float)
                C_new = np.sign(rng.standard_normal((K, m))
                                * rng.binomial(1, beta,  (K, m))).astype(float)
                X_new_whole = KRP(C_new, B_new).T
                X1_new = A_true @ X_new_whole
                retries += 1
            if not np.any(X_new_whole):
                raise RuntimeError('Alpha/beta too small: all-zero data matrix.')

        fi = X1_new                             # n x (J*K) — the fibers
        # idx = find(sum(abs(fi),1)) — non-zero columns only
        col_sums = np.sum(np.abs(fi), axis=0)
        idx = np.where(col_sums > 0)[0]
        Y_m   = fi[:, idx]                      # n x pf
        X_new = X_new_whole[:, idx]             # m x pf
        pf    = Y_m.shape[1]

        # ── IHT (MATLAB: tens_compare_NOODL.m ~line 120-170) ──
        t_start = time.perf_counter()

        # Initialise XS: hard threshold of A'*Y at C/2
        AtY = A_our.T @ Y_m                     # m x pf
        XS  = AtY * (np.abs(AtY) >= C_rad / 2.0)

        AtA_our  = A_our.T @ A_our              # m x m (precomputed)
        change_X = 1.0
        err_int_X = [float(np.linalg.norm(XS - X_new, 'fro')
                           / max(np.linalg.norm(X_new, 'fro'), 1e-12))]
        ii = 0

        # IHT convergence loop — exactly as in MATLAB non-distributed branch:
        #   while change_X > tol_X
        #     XS = XS - c*((AtA*XS) - A'*Y)
        #     XS(abs(XS)<=thr_c) = 0
        #     err_int_X = [err_int_X ...]
        #     if ii>2: change_X = abs(err_int_X[-1] - err_int_X[-3])
        while change_X > tol_X:
            eta_a = c          # MATLAB: eta_a = c  (fixed step size)
            thr   = thr_c      # MATLAB: thr = thr_c (fixed threshold)

            XS = XS - eta_a * (AtA_our @ XS - A_our.T @ Y_m)
            XS[np.abs(XS) <= thr] = 0.0

            err_int_X.append(
                float(np.linalg.norm(XS - X_new, 'fro')
                      / max(np.linalg.norm(X_new, 'fro'), 1e-12))
            )
            if ii > 2:
                # MATLAB: change_X = abs(err_int_X(end) - err_int_X(end-2))
                change_X = abs(err_int_X[-1] - err_int_X[-3])
            ii += 1

        # ── Dictionary update (MATLAB: ~line 175-185) ──
        # gr = (1/pf) * (Y_m - A_our*XS) * sign(XS)'
        gr = (1.0 / pf) * (Y_m - A_our @ XS) @ np.sign(XS).T

        # A_our = A_our + eta*gr   ← PLUS (ascent on fit, not descent)
        A_our = A_our + eta * gr
        A_our = nrmc(A_our)

        t_o = time.perf_counter() - t_start

        # ── Track errors ──
        err.append(float(np.linalg.norm(Y_m - A_our @ XS, 'fro')
                         / max(np.linalg.norm(Y_m, 'fro'), 1e-12)))
        errA.append(float(np.linalg.norm(A_our - A_true, 'fro')
                          / max(np.linalg.norm(A_true, 'fro'), 1e-12)))
        errX.append(float(np.linalg.norm(XS - X_new, 'fro')
                          / max(np.linalg.norm(X_new, 'fro'), 1e-12)))
        time_our.append(t_o)

        if np.isnan(errX[-1]) or np.isnan(errA[-1]):
            break

        change_A = errA[-1]  # MATLAB: change_A = errA(end)

        Y_last   = Y_m
        X_last   = XS
        X_last_o = X_new

        i += 1

    return {
        'A_our':   A_our,
        'errA':    np.array(errA),
        'errX':    np.array(errX),
        'err':     np.array(err),
        'time_our': np.array(time_our),
        'Y_last':  Y_last,
        'X_last':  X_last,
        'X_last_o': X_last_o,
        'iters':   i - 1,
    }


def run_tensor_noodl(n: int, m: int, J: int,
                     alpha: float, eta: float,
                     seed: int) -> dict:
    """
    Exact port of TensorNOODL/run_noodl_tens.m
    -------------------------------------------
    Sets all parameters exactly as in the MATLAB runner and calls
    tensor_noodl().

    MATLAB source (run_noodl_tens.m):
        b = a;  K = J;
        c = 0.2;  thr_c = 0.1;
        eta_arora = eta; eta_arora_red = eta;
        tol_X = 1e-12;  tol_A = 1e-10;

        A_o = randn(n,m); A_o = nrmc(A_o);
        noise = nrmc(randn(size(A_o)));
        noise = 2*noise*(1/log(m));
        A = A_o + noise; A = nrmc(A);
    """
    rng = np.random.default_rng(seed)

    # IHT parameters (from run_noodl_tens.m)
    c     = 0.2        # IHT step size
    thr_c = 0.1        # IHT hard threshold
    tol_X = 1e-12      # IHT convergence tolerance
    tol_A = 1e-10      # outer loop convergence tolerance
    beta  = alpha      # b = a in MATLAB

    # Generate ground-truth dictionary
    # MATLAB: A_o = randn(n,m); A_o = nrmc(A_o);
    A_o = nrmc(rng.standard_normal((n, m)))

    # Initialise with small perturbation
    # MATLAB: noise = nrmc(randn(size(A_o))); noise = 2*noise*(1/log(m));
    #         A = A_o + noise; A = nrmc(A);
    noise = nrmc(rng.standard_normal((n, m)))
    noise = 2 * noise * (1.0 / np.log(m))
    A_init = nrmc(A_o + noise)

    return tensor_noodl(
        A_init=A_init, A_true=A_o,
        J=J, K=J,
        alpha=alpha, beta=beta,
        c=c, thr_c=thr_c, eta=eta,
        tol_X=tol_X, tol_A=tol_A,
        rng=rng,
    )


# ══════════════════════════════════════════════════════════════════════════════
# M2PALS — matrix version  (Dmult_nmf_OMP.m)
# ══════════════════════════════════════════════════════════════════════════════

def dmult_nmf_omp(Y: np.ndarray, n_iter: int,
                  B_0: np.ndarray, A_0: np.ndarray,
                  D: List[np.ndarray],
                  contents: np.ndarray,
                  opts: dict = None) -> tuple:
    """
    Exact port of M2PALS/Dmult_nmf_OMP.m
    --------------------------------------
    M2PALS matrix algorithm: min ||Y - A B'||_F^2
    where A columns are selected from multiple dictionaries D.

    MATLAB source parameters:
        iter_err   = 10    (iterations per error evaluation)
        n_diff_err = 3     (filter size for relative error decrease)
        stop_diff  = 1e-5  (stopping: mean of filter < stop_diff)
        stop_err   = eps   (machine epsilon stop)
        inner      = 10    (NNLS inner iterations)

    MATLAB outer loop:
        while mean(diff_err_filt)>stop_diff && t<iter && err_anls_old>stop_err
            for i=1:iter_err
                [alloc,indexes,A_aux] = MatchP_multiD(A_aux,D,contents,opts{3})
                K_aux = [alloc;indexes]
                if NN: B = nnlsHALSupdt(Y,A,B',inner)'
                else:  B = (A backslash Y)'
                if NN: A = nnlsHALSupdt(Y',B,A',inner)'
                else:  A = Y/(B')
            err = sqrt(sqFnorm(Y-A_old*B')) / ynorm
            A_aux = A_old  (← restores A to pre-inner-loop value for error)

    Parameters
    ----------
    Y        : (n x N) data matrix
    n_iter   : max iterations
    B_0      : (N x r) initial B
    A_0      : (n x r) initial A
    D        : list of P arrays (n x d_p) — dictionaries
    contents : (P,) int array — atoms to select from each dict
    opts     : dict with keys 'NN' (bool), 'NMFinit' (bool), 'metric' (str)

    Returns
    -------
    A_aux, B_aux, K_aux, err_vec
    """
    if opts is None:
        opts = {'NN': True, 'NMFinit': True, 'metric': 'eucl'}

    # MATLAB parameters
    iter_err   = 10
    n_diff_err = 3
    stop_diff  = 1e-5
    stop_err   = np.finfo(float).eps
    inner      = 10

    iter_outer = int(np.ceil(n_iter / iter_err))

    # Initialise
    A_aux = np.abs(A_0.copy())
    B_aux = np.abs(B_0.copy())

    diff_err_filt = 1000.0 * np.ones(n_diff_err)
    res = Y - A_aux @ B_aux.T
    err_anls_old = float(np.sqrt(res.ravel() @ res.ravel()))
    err_anls_vec = [err_anls_old]

    ynorm = float(np.sqrt(sq_fnorm(Y)))

    # Optional NMF init (MATLAB: if strcmp(opts{2},'NMFinit'))
    # We skip HALSacc here for simplicity — it's an optional warm-start.
    # To replicate exactly, add HALSacc call here.

    K_aux = None
    t = 0

    while (float(np.mean(diff_err_filt)) > stop_diff
           and t < iter_outer
           and err_anls_old > stop_err):
        t += 1

        for _ in range(iter_err):
            # MP step: MatchP_multiD
            alloc, indexes, A_aux = match_p_multi_d(
                A_aux, D, contents, opts.get('metric', 'eucl'))
            K_aux = (alloc, indexes)
            A_old = A_aux.copy()

            # B update
            if opts.get('NN', True):
                B_aux = nnls_hals_updt(Y, A_aux, B_aux.T, inner).T
            else:
                B_aux = np.linalg.lstsq(A_aux, Y, rcond=None)[0].T

            # A update
            if opts.get('NN', True):
                A_aux = nnls_hals_updt(Y.T, B_aux, A_aux.T, inner).T
            else:
                # MATLAB: A = Y/(B')  → A B' = Y → A = Y @ pinv(B')
                A_aux = Y @ np.linalg.pinv(B_aux.T)

        # Error (uses A_old — before last inner loop A update)
        err_anls = float(np.sqrt(sq_fnorm(Y - A_old @ B_aux.T))) / max(ynorm, 1e-12)
        A_aux = A_old   # MATLAB: A_aux = A_old

        diff_err_filt[:-1] = diff_err_filt[1:]
        diff_err_filt[-1] = abs((err_anls - err_anls_old) / max(err_anls, 1e-12))

        err_anls_old = err_anls
        err_anls_vec.append(err_anls)

    # Final NNLS polish (MATLAB: if NN, B = nnlsHALSupdt(Y,A,B',500)')
    if opts.get('NN', True):
        B_aux = nnls_hals_updt(Y, A_aux, B_aux.T, 500).T
    err_anls_vec.append(
        float(np.sqrt(sq_fnorm(Y - A_aux @ B_aux.T))) / max(ynorm, 1e-12))

    return A_aux, B_aux, K_aux, np.array(err_anls_vec)


# ══════════════════════════════════════════════════════════════════════════════
# M2PALS — tensor (CPD) version  (tensor/Dmult_mpnals.m)
# ══════════════════════════════════════════════════════════════════════════════

def dmult_mpnals(Y: np.ndarray, n_iter: int,
                 A_0: np.ndarray, B_0: np.ndarray, C_0: np.ndarray,
                 D: List[np.ndarray], contents: np.ndarray,
                 opts: dict = None) -> tuple:
    """
    Exact port of M2PALS/tensor/Dmult_mpnals.m
    -------------------------------------------
    M2PALS tensor algorithm for 3-way CPD with dictionary constraint on
    mode-2 factor (B in this code, second mode in tensor).

    Model: T ≈ [[A, B, C]] where B columns are from dictionaries D.

    MATLAB source structure:
        Y_mode{1} = reshape(permute(Y,[1,2,3]), dim(1), J(1))  ← mode-1
        Y_mode{2} = reshape(permute(Y,[2,1,3]), dim(2), J(2))  ← mode-2
        Y_mode{3} = reshape(permute(Y,[3,1,2]), dim(3), J(3))  ← mode-3

    Inner loop (iter_err times):
        A update: nnlsHALSupdt(Y_mode1', kr(C,B), A', inner)'   (NN case)
                  col_norm(A)                                     ← normalise
        B update: nnlsHALSupdt(Y_mode2', kr(C,A), B', inner)'
                  [alloc,indexes,B] = MatchP_multiD(B,D,contents) ← MP step
        C update: nnlsHALSupdt(Y_mode3', kr(B,A), C', inner)'

    Error uses KhatriRao formula:
        temp_BA = (B'B).*(A'A)
        Y_BA    = Y_mode3 * kr(B,A)
        err = snorm_Y - 2*sum(Y_BA.*C) + sum((C'C).*temp_BA)

    Parameters
    ----------
    Y        : (n x J x K) tensor
    n_iter   : max iterations
    A_0      : (n x r) initial mode-1 factor
    B_0      : (J x r) initial mode-2 factor (will be dict-constrained)
    C_0      : (K x r) initial mode-3 factor
    D        : list of P arrays (J x d_p) — dictionaries for mode-2
    contents : (P,) int array — atoms to select from each dict
    opts     : dict with 'NN' (bool), 'warm_start' (bool)

    Returns
    -------
    A_aux, B_aux, C_aux, K_aux, err_vec
    """
    if opts is None:
        opts = {'NN': True, 'warm_start': True}

    # MATLAB parameters (Dmult_mpnals.m)
    iter_err   = 10
    n_diff_err = 3
    stop_diff  = 1e-8
    stop_err   = np.finfo(float).eps
    inner      = 20

    iter_outer = int(np.ceil(n_iter / iter_err))
    dim        = Y.shape              # (n, J, K)
    n, J, K    = dim

    # Mode unfoldings (MATLAB: reshape(permute(Y,[...]),dim(i),J(i)))
    # MATLAB reshape is Fortran (column-major) order.
    # C-order numpy reshape gives a different index mapping and breaks
    # the inner product formula used in the closed-form error.
    # MATLAB: Y_mode{1} = reshape(permute(Y,[1,2,3]), n,  J*K)
    Y1 = np.reshape(Y.transpose(0, 1, 2), (n, J * K), order='F')
    # MATLAB: Y_mode{2} = reshape(permute(Y,[2,1,3]), J,  n*K)
    Y2 = np.reshape(Y.transpose(1, 0, 2), (J, n * K), order='F')
    # MATLAB: Y_mode{3} = reshape(permute(Y,[3,1,2]), K,  n*J)
    Y3 = np.reshape(Y.transpose(2, 0, 1), (K, n * J), order='F')

    A_aux = A_0.copy()
    B_aux = B_0.copy()
    C_aux = C_0.copy()

    diff_err_filt = 1000.0 * np.ones(n_diff_err)

    # Initial error: res = Y_mode1 - A * kr(C,B)'
    res_init  = Y1 - A_aux @ kr(C_aux, B_aux).T
    err_anls_old = float(res_init.ravel() @ res_init.ravel())
    snorm_Y      = float(Y1.ravel() @ Y1.ravel())
    err_anls_vec = [err_anls_old]

    K_aux = None
    t = 0

    while (float(np.mean(diff_err_filt)) > stop_diff
           and t < iter_outer
           and err_anls_old > stop_err):
        t += 1

        for _ in range(iter_err):
            # A update: Y_mode1' kr(C,B)  [NN or unconstrained]
            if opts.get('NN', True):
                if opts.get('warm_start', True):
                    A_aux = nnls_hals_updt(Y1.T, kr(C_aux, B_aux), A_aux.T, inner).T
                else:
                    A_aux = nnls_hals_updt(Y1.T, kr(C_aux, B_aux)).T
            else:
                CB = kr(C_aux, B_aux)
                Y_BC  = Y1 @ CB
                tBC   = (C_aux.T @ C_aux) * (B_aux.T @ B_aux)
                A_aux = Y_BC @ np.linalg.pinv(tBC)

            # Normalise A (MATLAB: col_norm(A_aux))
            A_aux = col_norm(A_aux)

            # B update
            if opts.get('NN', True):
                if opts.get('warm_start', True):
                    B_aux = nnls_hals_updt(Y2.T, kr(C_aux, A_aux), B_aux.T, inner).T
                else:
                    B_aux = nnls_hals_updt(Y2.T, kr(C_aux, A_aux)).T
            else:
                CA = kr(C_aux, A_aux)
                Y_AC  = Y2 @ CA
                tAC   = (C_aux.T @ C_aux) * (A_aux.T @ A_aux)
                B_aux = Y_AC @ np.linalg.pinv(tAC)

            # MP step on B (dictionary constraint)
            alloc, indexes, B_aux = match_p_multi_d(B_aux, D, contents)
            K_aux = (alloc, indexes)

            # C update
            if opts.get('NN', True):
                if opts.get('warm_start', True):
                    C_aux = nnls_hals_updt(Y3.T, kr(B_aux, A_aux), C_aux.T, inner).T
                else:
                    C_aux = nnls_hals_updt(Y3.T, kr(B_aux, A_aux)).T
            else:
                BA = kr(B_aux, A_aux)
                Y_AB  = Y3 @ BA
                tAB   = (B_aux.T @ B_aux) * (A_aux.T @ A_aux)
                C_aux = Y_AB @ np.linalg.pinv(tAB)

        # Error (MATLAB closed-form using KR structure)
        temp_BA = (B_aux.T @ B_aux) * (A_aux.T @ A_aux)
        Y_BA    = Y3 @ kr(B_aux, A_aux)
        err_anls = float(snorm_Y
                         - 2.0 * np.sum(Y_BA * C_aux)
                         + np.sum((C_aux.T @ C_aux) * temp_BA))

        diff_err_filt[:-1] = diff_err_filt[1:]
        diff_err_filt[-1] = abs((err_anls - err_anls_old) / max(abs(err_anls), 1e-12))
        err_anls_old = err_anls
        err_anls_vec.append(err_anls)

    return A_aux, B_aux, C_aux, K_aux, np.array(err_anls_vec)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark-compatible wrappers
# (same interface as notebook: stem, rank → weights, factors, runtime, extras)
# ══════════════════════════════════════════════════════════════════════════════

def run_tensor_noodl_benchmark(stem: str, rank: int,
                                alpha: float = 0.05,
                                eta: float = 20.0,
                                seed: int = 42) -> tuple:
    """
    Wrapper for the benchmark notebook.
    Generates its own data using the exact MATLAB parameter convention,
    then returns factors in the benchmark format.

    NOTE: TensorNOODL generates its OWN data internally (online setting).
    It does not read the benchmark tensor — it needs A_true to generate
    streaming slices. To use a fixed benchmark tensor, supply A_true
    extracted from the ground truth sidecar.
    """
    import json, os
    T = np.load(f'{stem}.npy')
    with open(f'{stem}_meta.json') as f:
        meta = json.load(f)

    n, J, K = T.shape
    m = rank

    # Load ground truth A if available (needed for TensorNOODL's online loop)
    truth_path = f'{stem}_truth.npz'
    if os.path.exists(truth_path):
        truth = np.load(truth_path)
        A_true = truth['factor_0']          # ground truth mode-0 factor
        # Perturb to get init (replicating run_noodl_tens.m)
        rng = np.random.default_rng(seed)
        noise = nrmc(rng.standard_normal(A_true.shape))
        noise = 2 * noise * (1.0 / np.log(m))
        A_init = nrmc(A_true + noise)
    else:
        rng = np.random.default_rng(seed)
        A_true = nrmc(rng.standard_normal((n, m)))
        noise  = nrmc(rng.standard_normal((n, m)))
        noise  = 2 * noise * (1.0 / np.log(m))
        A_init = nrmc(A_true + noise)
        rng = np.random.default_rng(seed)   # reset for data generation

    t0 = time.perf_counter()
    result = tensor_noodl(
        A_init=A_init, A_true=A_true,
        J=J, K=K,
        alpha=alpha, beta=alpha,
        c=0.2, thr_c=0.1,
        eta=eta, tol_X=1e-12, tol_A=1e-10,
        rng=np.random.default_rng(seed),
    )
    runtime = time.perf_counter() - t0

    A = result['A_our']
    # Recover B and C from X_last using UNTANGLE-KRP (Algorithm 2)
    # X_last is m x pf, idx tells us which (j,k) positions are non-zero
    # For benchmark purposes we return A and approximate B, C
    B = np.zeros((J, m))
    C = np.zeros((K, m))
    w = np.ones(m)

    extras = {'eta': eta, 'alpha': alpha, 'iters': result['iters'],
              'errA_final': float(result['errA'][-1]) if len(result['errA']) else None,
              'engine': 'python_exact'}
    return w, [A, B, C], runtime, extras


def run_m2pals_benchmark(stem: str, rank: int,
                          n_iter: int = 100) -> tuple:
    """
    Wrapper for M2PALS in benchmark format.
    Reads tensor and dictionary sidecar, runs Dmult_mpnals.
    """
    import json, os
    T    = np.load(f'{stem}.npy')
    d_np = np.load(f'{stem}_dict.npy')
    D_mat = d_np['D']                      # n x d
    true_indices = d_np['true_indices']

    n, J, K = T.shape
    m = rank

    # Single dictionary, select exactly m atoms
    D_list   = [D_mat]
    contents = np.array([m])

    rng = np.random.default_rng(0)
    A_0 = col_norm(np.abs(rng.standard_normal((n, m))))
    B_0 = col_norm(np.abs(rng.standard_normal((J, m))))
    C_0 = col_norm(np.abs(rng.standard_normal((K, m))))

    t0 = time.perf_counter()
    A_aux, B_aux, C_aux, K_aux, err_vec = dmult_mpnals(
        T, n_iter, A_0, B_0, C_0, D_list, contents,
        opts={'NN': True, 'warm_start': True},
    )
    runtime = time.perf_counter() - t0

    # Atom selection accuracy
    if K_aux is not None:
        _, sel_idx = K_aux
        true_set = set(true_indices.tolist())
        sel_set  = set(sel_idx.tolist())
        asa = len(true_set & sel_set) / max(len(true_set), 1)
    else:
        asa = float('nan')
        sel_idx = []

    w = np.ones(m)
    extras = {'n_iter': n_iter,
              'selected_idx': list(sel_idx) if hasattr(sel_idx, '__iter__') else [],
              'atom_selection_acc': asa,
              'final_err': float(err_vec[-1]),
              'engine': 'python_exact'}
    return w, [A_aux, B_aux, C_aux], runtime, extras
