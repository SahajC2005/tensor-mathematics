"""
run_extended.py -- pushing the benchmark further.

Three new studies, all on REAL RGB image patches (skimage 'astronaut'):
  E. Real-image efficiency frontier, now including Block-Term Decomposition (BTD)
  F. CP instability: svd-init vs random-init x rank, measuring PSNR spread AND
     a CP-degeneracy diagnostic (diverging, partially-cancelling components)
  G. BTD spectrum: BTD traces the CP <-> Tucker continuum as you vary block rank L
     and number of blocks; overlaid against CP and Tucker reference points.
"""
import os, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os
_SRC_DL = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "..", "..", "src", "dictionary_learning")
_sys.path.insert(0, _os.path.normpath(_SRC_DL))
from data import make_image_dataset
from tensor_dl import make_model, psnr

OUT = os.environ.get("TDL_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
COLORS = {"Vectorized": "#444444", "CP": "#d1495b", "Tucker": "#2e8b57",
          "TT": "#3b6fb6", "TR": "#e08e0b", "BTD": "#7b4fb6"}

SHAPE = (8, 8, 3)
NA = 100
ENCODE_S = 6


def fit(kind, ds, **kw):
    m = make_model(kind, SHAPE, **kw)
    m.fit(ds["train_noisy"])
    return m


# ======================================================================================
# E. Real-image frontier (with BTD)
# ======================================================================================
def experiment_real_frontier(ds, seeds):
    print("\n[E] real-image efficiency frontier (with BTD) ...")
    ORDER = ["Vectorized", "CP", "Tucker", "TT", "TR", "BTD"]
    specs = lambda sd: {
        "Vectorized": ("vectorized", dict(n_atoms=NA, sparsity=4, n_iter=10, seed=sd)),
        "CP":         ("cp", dict(n_atoms=NA, sparsity=4, n_iter=10, rank=1, seed=sd)),
        "Tucker":     ("tucker", dict(mode_ranks=[9, 9, 3], n_iter=6, mode_sparsity=2, seed=sd)),
        "TT":         ("tt", dict(n_atoms=NA, sparsity=4, n_iter=10, rank=2, seed=sd)),
        "TR":         ("tr", dict(n_atoms=NA, sparsity=4, n_iter=10, rank=2, seed=sd)),
        "BTD":        ("btd", dict(n_atoms=NA, sparsity=4, n_iter=10, rank=2,
                                   proj_kw=dict(n_blocks=1), seed=sd)),
    }
    rec = {n: {"params": [], "psnr": [], "atoms": [], "ft": []} for n in ORDER}
    for sd in seeds:
        sp = specs(sd)
        for n in ORDER:
            kind, kw = sp[n]
            m = fit(kind, ds, **kw)
            rec[n]["params"].append(m.n_params)
            rec[n]["atoms"].append(m.n_atoms)
            rec[n]["ft"].append(m.fit_time_)
            rec[n]["psnr"].append(psnr(ds["test_clean"], m.reconstruct(ds["test_noisy"], ENCODE_S)))
        print("   seed %d done" % sd)
    # plot
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for n in ORDER:
        p, mu, sd = np.mean(rec[n]["params"]), np.mean(rec[n]["psnr"]), np.std(rec[n]["psnr"])
        ax.errorbar(p, mu, yerr=sd, fmt="o", ms=12, color=COLORS[n], capsize=4)
        ax.annotate(" " + n, (p, mu), color=COLORS[n], fontweight="bold", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("dictionary parameters (model size, log scale)")
    ax.set_ylabel("test denoising PSNR vs clean (dB)")
    ax.set_title("Real RGB patches (astronaut, 8x8x3): accuracy vs model size")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_real_frontier.png"); plt.close(fig)

    # table
    lines = ["| Model | Atoms | Dict params | Compression | Fit (s) | Test PSNR (dB) |",
             "|---|---:|---:|---:|---:|---:|"]
    vp = np.mean(rec["Vectorized"]["params"])
    csv = ["model,atoms,dict_params,compression_vs_vec,fit_s,test_psnr_db"]
    for n in ORDER:
        a = int(np.mean(rec[n]["atoms"])); p = np.mean(rec[n]["params"])
        ft = np.mean(rec[n]["ft"]); mu = np.mean(rec[n]["psnr"])
        lines.append(f"| {n} | {a} | {int(p)} | {vp/p:4.1f}x | {ft:4.2f} | {mu:5.2f} |")
        csv.append(f"{n},{a},{int(p)},{vp/p:.2f},{ft:.3f},{mu:.3f}")
    md = ("# Extended results — real RGB image patches\n\n"
          f"Image: {ds['image']}, patch {ds['atom_shape']}, SNR={ds['snr_db']} dB, "
          f"encode sparsity={ENCODE_S}.\n\n" + "\n".join(lines) + "\n")
    open(f"{OUT}/results_real.md", "w").write(md)
    open(f"{OUT}/results_real.csv", "w").write("\n".join(csv) + "\n")
    print(md)
    return rec


# ======================================================================================
# F. CP instability: init strategy x rank -> PSNR spread + degeneracy
# ======================================================================================
def experiment_cp_instability(ds, seeds, ranks=(1, 2, 3)):
    print("\n[F] CP instability (svd vs random init) ...")
    res = {ci: {"rank": [], "psnr_mean": [], "psnr_std": [],
                "deg_mean": [], "deg_std": [], "ft_mean": []} for ci in ("svd", "random")}
    for ci in ("svd", "random"):
        for r in ranks:
            ps, dg, ft = [], [], []
            for sd in seeds:
                m = fit("cp", ds, n_atoms=80, sparsity=4, n_iter=6, rank=r,
                        init=("data" if ci == "svd" else "random"),
                        proj_kw=dict(cp_init=ci), seed=sd)
                ps.append(psnr(ds["test_clean"], m.reconstruct(ds["test_noisy"], ENCODE_S)))
                dg.append(m.degeneracy_); ft.append(m.fit_time_)
            res[ci]["rank"].append(r)
            res[ci]["psnr_mean"].append(np.mean(ps)); res[ci]["psnr_std"].append(np.std(ps))
            res[ci]["deg_mean"].append(np.mean(dg)); res[ci]["deg_std"].append(np.std(dg))
            res[ci]["ft_mean"].append(np.mean(ft))
            print(f"   {ci:6s} rank {r}: PSNR {np.mean(ps):5.2f}±{np.std(ps):.2f}  "
                  f"deg {np.mean(dg):4.2f}  fit {np.mean(ft):4.2f}s")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for ci, ls, mk in (("svd", "-", "o"), ("random", "--", "s")):
        a1.errorbar(res[ci]["rank"], res[ci]["psnr_mean"], yerr=res[ci]["psnr_std"],
                    ls=ls, marker=mk, color=COLORS["CP"], capsize=3,
                    label=f"CP ({ci} init)")
        a2.errorbar(res[ci]["rank"], res[ci]["deg_mean"], yerr=res[ci]["deg_std"],
                    ls=ls, marker=mk, color="#444", capsize=3, label=f"{ci} init")
    a1.set_xlabel("CP atom rank"); a1.set_ylabel("test PSNR (dB)")
    a1.set_title("CP accuracy & seed-spread vs rank"); a1.legend(frameon=False)
    a1.set_xticks(list((1, 2, 3)))
    a2.axhline(1.0, color="green", lw=1, ls=":", label="no degeneracy")
    a2.set_xlabel("CP atom rank"); a2.set_ylabel("degeneracy ratio  (>1 = cancelling factors)")
    a2.set_title("CP degeneracy diagnostic"); a2.legend(frameon=False)
    a2.set_xticks(list((1, 2, 3)))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_cp_instability.png"); plt.close(fig)
    return res


# ======================================================================================
# G. BTD spectrum: trace the CP <-> Tucker continuum
# ======================================================================================
def experiment_btd_spectrum(ds, seeds):
    print("\n[G] BTD spectrum (CP <-> Tucker continuum) ...")
    # (label, L, n_blocks)
    grid = [("L1xb1", 1, 1), ("L1xb2", 1, 2), ("L1xb3", 1, 3),
            ("L2xb1", 2, 1), ("L2xb2", 2, 2), ("L3xb1", 3, 1)]
    pts = {"label": [], "params": [], "psnr": [], "psnr_std": []}
    for label, L, nb in grid:
        ps, pr = [], []
        for sd in seeds:
            m = fit("btd", ds, n_atoms=NA, sparsity=4, n_iter=8, rank=L,
                    proj_kw=dict(n_blocks=nb), seed=sd)
            ps.append(psnr(ds["test_clean"], m.reconstruct(ds["test_noisy"], ENCODE_S)))
            pr.append(m.n_params)
        pts["label"].append(label); pts["params"].append(np.mean(pr))
        pts["psnr"].append(np.mean(ps)); pts["psnr_std"].append(np.std(ps))
        print(f"   BTD {label}: params={int(np.mean(pr)):5d} PSNR={np.mean(ps):.2f}")

    # reference CP (rank 1..3) and Tucker
    cp_pts = {"params": [], "psnr": []}
    for r in (1, 2, 3):
        ps, pr = [], []
        for sd in seeds:
            m = fit("cp", ds, n_atoms=NA, sparsity=4, n_iter=8, rank=r, seed=sd)
            ps.append(psnr(ds["test_clean"], m.reconstruct(ds["test_noisy"], ENCODE_S)))
            pr.append(m.n_params)
        cp_pts["params"].append(np.mean(pr)); cp_pts["psnr"].append(np.mean(ps))
    tk = fit("tucker", ds, mode_ranks=[9, 9, 3], n_iter=6, mode_sparsity=2, seed=seeds[0])
    tk_pt = (tk.n_params, psnr(ds["test_clean"], tk.reconstruct(ds["test_noisy"], ENCODE_S)))

    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.errorbar(pts["params"], pts["psnr"], yerr=pts["psnr_std"], fmt="o-",
                color=COLORS["BTD"], capsize=3, ms=9, label="BTD (vary L, #blocks)")
    for lab, x, y in zip(pts["label"], pts["params"], pts["psnr"]):
        ax.annotate(lab, (x, y), fontsize=8, color=COLORS["BTD"],
                    xytext=(0, 7), textcoords="offset points", ha="center")
    ax.plot(cp_pts["params"], cp_pts["psnr"], "s--", color=COLORS["CP"],
            ms=9, label="CP (rank 1-3)")
    ax.plot([tk_pt[0]], [tk_pt[1]], "D", color=COLORS["Tucker"], ms=12, label="Tucker")
    ax.set_xscale("log")
    ax.set_xlabel("dictionary parameters (log scale)")
    ax.set_ylabel("test denoising PSNR vs clean (dB)")
    ax.set_title("BTD traces the CP↔Tucker continuum\n(L=block multilinear rank)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_btd_spectrum.png"); plt.close(fig)
    return pts


def main():
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    print("Loading real image patches (astronaut) ...")
    ds = make_image_dataset("astronaut", patch=SHAPE, n_train=320, n_test=90,
                            snr_db=12.0, seed=7)
    if stage in ("E", "all"):
        experiment_real_frontier(ds, seeds=[0, 1, 2, 3])
    if stage in ("F", "all"):
        experiment_cp_instability(ds, seeds=[0, 1, 2, 3, 4])
    if stage in ("G", "all"):
        experiment_btd_spectrum(ds, seeds=[0, 1, 2])
    print(f"\nstage {stage} DONE in {time.time()-t0:.1f}s. Figures in {OUT}")


if __name__ == "__main__":
    main()
