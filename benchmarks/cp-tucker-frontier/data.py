"""
data.py -- synthetic 3rd-order tensor "patch" datasets with controllable structure.

We build a ground-truth set of atoms that deliberately MIXES structure types so that
no single decomposition trivially dominates:
  - separable rank-1 atoms          (favour CP / Tucker)
  - low TT-rank smooth atoms        (favour TT / TR)
  - dense random atoms              (favour the vectorized baseline)
Each sample is a sparse combination of true atoms plus Gaussian noise at a target SNR.
"""
import numpy as np
import tensorly as tl
from tensorly import tt_to_tensor


def _rand_separable(shape, rng):
    vecs = [rng.standard_normal(d) for d in shape]
    T = vecs[0]
    for v in vecs[1:]:
        T = np.multiply.outer(T, v)
    return T


def _rand_tt(shape, rng, rank=2):
    ranks = [1] + [rank] * (len(shape) - 1) + [1]
    cores = [rng.standard_normal((ranks[i], shape[i], ranks[i + 1]))
             for i in range(len(shape))]
    return tt_to_tensor(cores)


def make_dataset(atom_shape=(6, 6, 6), n_true=24, n_train=320, n_test=90,
                 s_true=3, snr_db=12.0, mix=(0.4, 0.4, 0.2), seed=0):
    """
    Returns dict with:
      train_clean, train_noisy, test_clean, test_noisy : (M, *atom_shape)
      true_atoms : (n_true, *atom_shape)
    mix = fraction of (separable, tt, dense) ground-truth atoms.
    """
    rng = np.random.default_rng(seed)
    n_sep = int(round(n_true * mix[0]))
    n_tt = int(round(n_true * mix[1]))
    n_dense = n_true - n_sep - n_tt

    atoms = []
    for _ in range(n_sep):
        atoms.append(_rand_separable(atom_shape, rng))
    for _ in range(n_tt):
        atoms.append(_rand_tt(atom_shape, rng, rank=2))
    for _ in range(n_dense):
        atoms.append(rng.standard_normal(atom_shape))
    atoms = np.stack(atoms)
    atoms /= np.sqrt((atoms ** 2).sum(axis=tuple(range(1, atoms.ndim)),
                                       keepdims=True)) + 1e-12

    def synth(M):
        clean = np.zeros((M,) + atom_shape)
        for m in range(M):
            sel = rng.choice(n_true, size=s_true, replace=False)
            coef = rng.standard_normal(s_true)
            clean[m] = np.tensordot(coef, atoms[sel], axes=(0, 0))
        sig_pow = np.mean(clean ** 2)
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        noisy = clean + rng.standard_normal(clean.shape) * np.sqrt(noise_pow)
        return clean, noisy

def make_image_dataset(image="astronaut", patch=(8, 8, 3), n_train=320, n_test=90,
                       snr_db=12.0, seed=0):
    """Real RGB-image patches as 3rd-order tensors (spatial x spatial x colour).

    Patches are mean-removed and unit-normalised (standard for dictionary learning),
    then corrupted with Gaussian noise to a target SNR. Structure here is *real* and
    only mildly separable, which is a harder, more realistic test than synthetic atoms.
    """
    from skimage import data as skdata
    rng = np.random.default_rng(seed)
    img = getattr(skdata, image)().astype(float) / 255.0      # (H, W, 3)
    H, W, C = img.shape
    ph, pw, pc = patch
    M = n_train + n_test

    P = np.empty((M,) + patch)
    for m in range(M):
        i = rng.integers(0, H - ph + 1)
        j = rng.integers(0, W - pw + 1)
        blk = img[i:i + ph, j:j + pw, :pc]
        blk = blk - blk.mean()
        nrm = np.sqrt((blk ** 2).sum())
        P[m] = blk / (nrm + 1e-12)

    clean = P
    sig_pow = np.mean(clean ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noisy = clean + rng.standard_normal(clean.shape) * np.sqrt(noise_pow)

def make_hsi_dataset(patch=(6, 6, 16), n_endmembers=5, img=(72, 72),
                     n_train=300, n_val=100, n_test=100, snr_db=12.0, seed=0):
    """Synthetic hyperspectral-style cube: spatial x spatial x MANY bands.

    Physically-motivated low-rank structure: the cube is a few smooth spectral
    'endmember' signatures mixed by smooth spatial abundance maps, so it is genuinely
    low multilinear-rank and low TT-rank across the fat spectral mode -- the regime where
    Tucker/TT/TR are supposed to win and a vectorised dictionary is starved (huge atoms).
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    ph, pw, bands = patch
    H, W = img
    wl = np.linspace(0, 1, bands)

    # smooth endmember spectra (Gaussian bumps over wavelength)
    spectra = np.stack([np.exp(-0.5 * ((wl - rng.uniform(0.1, 0.9)) /
                                        rng.uniform(0.05, 0.18)) ** 2)
                        for _ in range(n_endmembers)])                  # (E, bands)
    # smooth spatial abundance maps, softmaxed across endmembers
    ab = np.stack([gaussian_filter(rng.standard_normal((H, W)), sigma=4)
                   for _ in range(n_endmembers)])                       # (E, H, W)
    ab = np.exp(ab - ab.max(0)); ab /= ab.sum(0, keepdims=True)
    cube = np.einsum("ehw,eb->hwb", ab, spectra)                       # (H, W, bands)

    M = n_train + n_val + n_test
    P = np.empty((M,) + patch)
    for m in range(M):
        i, j = rng.integers(0, H - ph + 1), rng.integers(0, W - pw + 1)
        blk = cube[i:i + ph, j:j + pw, :]
        blk = blk - blk.mean()
        P[m] = blk / (np.sqrt((blk ** 2).sum()) + 1e-12)

    sig = np.mean(P ** 2)
    noisy = P + rng.standard_normal(P.shape) * np.sqrt(sig / (10 ** (snr_db / 10)))
    a, b = n_train, n_train + n_val
    return dict(train_clean=P[:a], train_noisy=noisy[:a],
                val_clean=P[a:b], val_noisy=noisy[a:b],
                test_clean=P[b:], test_noisy=noisy[b:],
                atom_shape=patch, snr_db=snr_db, image="synthetic-HSI")
