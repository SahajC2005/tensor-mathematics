"""
run_benchmark.py -- the experiment.

Four studies, all on a shared mixed-structure synthetic dataset:
  A. Efficiency frontier   : test denoising PSNR vs dictionary size (params)
  B. Sparsity behaviour    : test PSNR vs coding sparsity (atoms used per sample)
  C. Rank sensitivity      : how each method responds to its rank knob
  D. Initialisation sens.  : spread of PSNR across random seeds (robustness)

Everything is averaged over seeds; figures + CSV/Markdown tables are written to ./outputs.
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
from data import make_dataset
from tensor_dl import make_model, psnr

OUT = os.environ.get("TDL_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT, exist_ok=True)

# ----- global style -----
plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"Vectorized": "#444444", "CP": "#d1495b",
          "Tucker": "#2e8b57", "TT": "#3b6fb6", "TR": "#e08e0b"}
ORDER = ["Vectorized", "CP", "Tucker", "TT", "TR"]

ATOM_SHAPE = (6, 6, 6)
NA = 120                     # n_atoms for the KSVD-family models
ENCODE_S = 6                 # default coding sparsity for frontier / init studies


def base_configs(seed, n_iter=10):
    """Default config per model at a comparable operating point."""
    return {
        "Vectorized": ("vectorized", dict(n_atoms=NA, sparsity=4, n_iter=n_iter, seed=seed)),
        "CP":         ("cp",         dict(n_atoms=NA, sparsity=4, n_iter=n_iter, rank=1, seed=seed)),
        "Tucker":     ("tucker",     dict(mode_ranks=[7, 7, 7], n_iter=6, mode_sparsity=2, seed=seed)),
        "TT":         ("tt",         dict(n_atoms=NA, sparsity=4, n_iter=n_iter, rank=2, seed=seed)),
        "TR":         ("tr",         dict(n_atoms=NA, sparsity=4, n_iter=n_iter, rank=2, seed=seed)),
    }


def fit_one(name, kind, kw, ds):
    m = make_model(kind, ATOM_SHAPE, **kw)
    m.fit(ds["train_noisy"])
    return m


# ======================================================================================
# Experiments
# ======================================================================================
def experiment_frontier_and_sparsity(ds, seeds, sparsity_grid):
    """Fit each model once per seed, then evaluate at every coding sparsity."""
    print("\n[A/B] efficiency frontier + sparsity sweep ...")
    # records[name] = dict(params=[], fit_time=[], atoms=[], psnr_by_s={s:[...]})
    rec = {n: dict(params=[], fit_time=[], atoms=[],
                   psnr=[], psnr_by_s={s: [] for s in sparsity_grid}) for n in ORDER}
    for seed in seeds:
        cfgs = base_configs(seed)
        for name in ORDER:
            kind, kw = cfgs[name]
            m = fit_one(name, kind, kw, ds)
            rec[name]["params"].append(m.n_params)
            rec[name]["fit_time"].append(m.fit_time_)
            rec[name]["atoms"].append(m.n_atoms)
            # denoising PSNR at default sparsity
            recon = m.reconstruct(ds["test_noisy"], ENCODE_S)
            rec[name]["psnr"].append(psnr(ds["test_clean"], recon))
            # sparsity sweep (reuse fitted dict)
            for s in sparsity_grid:
                r = m.reconstruct(ds["test_noisy"], s)
                rec[name]["psnr_by_s"][s].append(psnr(ds["test_clean"], r))
            print(f"   seed {seed} {name:11s} params={m.n_params:7d} "
                  f"atoms={m.n_atoms:4d} PSNR={rec[name]['psnr'][-1]:5.2f}")
    return rec


def experiment_rank(ds, seeds):
    """Sweep each method's rank knob; vectorized swept on n_atoms as its 'capacity' knob."""
    print("\n[C] rank sensitivity ...")
    grids = {
        "CP":     [("rank", r) for r in (1, 2, 3)],
        "TT":     [("rank", r) for r in (1, 2, 3, 4)],
        "TR":     [("rank", r) for r in (1, 2, 3)],
        "Tucker": [("mode_ranks", [r, r, r]) for r in (5, 6, 7, 8)],
    }
    out = {}
    for name, grid in grids.items():
        out[name] = {"knob": [], "psnr_mean": [], "psnr_std": [], "params": []}
        for key, val in grid:
            ps, pr = [], []
            for seed in seeds:
                kind = {"CP": "cp", "TT": "tt", "TR": "tr", "Tucker": "tucker"}[name]
                if name == "Tucker":
                    kw = dict(mode_ranks=val, n_iter=6, mode_sparsity=2, seed=seed)
                    knob = val[0]
                else:
                    kw = dict(n_atoms=NA, sparsity=4, n_iter=8, rank=val, seed=seed)
                    knob = val
                m = fit_one(name, kind, kw, ds)
                ps.append(psnr(ds["test_clean"], m.reconstruct(ds["test_noisy"], ENCODE_S)))
                pr.append(m.n_params)
            out[name]["knob"].append(knob)
            out[name]["psnr_mean"].append(np.mean(ps))
            out[name]["psnr_std"].append(np.std(ps))
            out[name]["params"].append(int(np.mean(pr)))
            print(f"   {name:7s} knob={knob} PSNR={np.mean(ps):5.2f}±{np.std(ps):.2f} "
                  f"params={int(np.mean(pr))}")
    return out


def experiment_init(ds, seeds):
    """Spread of PSNR across seeds at the base config (robustness to init)."""
    print("\n[D] initialisation sensitivity ...")
    out = {n: [] for n in ORDER}
    for seed in seeds:
        cfgs = base_configs(seed)
        for name in ORDER:
            kind, kw = cfgs[name]
            m = fit_one(name, kind, kw, ds)
            out[name].append(psnr(ds["test_clean"], m.reconstruct(ds["test_noisy"], ENCODE_S)))
    for name in ORDER:
        a = np.array(out[name])
        print(f"   {name:11s} PSNR {a.mean():5.2f} ± {a.std():.2f}  "
              f"(min {a.min():.2f}, max {a.max():.2f})")
    return out


# ======================================================================================
# Plots
# ======================================================================================
def plot_frontier(rec):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ORDER:
        p = np.mean(rec[name]["params"])
        m = np.mean(rec[name]["psnr"])
        s = np.std(rec[name]["psnr"])
        ax.errorbar(p, m, yerr=s, fmt="o", ms=11, color=COLORS[name],
                    capsize=4, elinewidth=1.5)
        ax.annotate(f" {name}", (p, m), color=COLORS[name],
                    fontsize=10, fontweight="bold", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("dictionary parameters (model size, log scale)")
    ax.set_ylabel("test denoising PSNR vs clean (dB)")
    ax.set_title("Efficiency frontier: accuracy vs model size\n"
                 "(up-and-left is better)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_frontier.png"); plt.close(fig)


def plot_sparsity(rec, sparsity_grid):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ORDER:
        means = [np.mean(rec[name]["psnr_by_s"][s]) for s in sparsity_grid]
        stds = [np.std(rec[name]["psnr_by_s"][s]) for s in sparsity_grid]
        ax.errorbar(sparsity_grid, means, yerr=stds, marker="o",
                    color=COLORS[name], label=name, capsize=3)
    ax.set_xlabel("coding sparsity  (non-zero atoms per sample)")
    ax.set_ylabel("test denoising PSNR vs clean (dB)")
    ax.set_title("Reconstruction accuracy vs coding sparsity")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_sparsity.png"); plt.close(fig)


def plot_rank(rankres):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, d in rankres.items():
        a1.errorbar(d["knob"], d["psnr_mean"], yerr=d["psnr_std"], marker="o",
                    color=COLORS[name], label=name, capsize=3)
        a2.plot(d["params"], d["psnr_mean"], marker="o", color=COLORS[name], label=name)
    a1.set_xlabel("rank knob  (CP/TT/TR rank, or Tucker per-mode size)")
    a1.set_ylabel("test PSNR (dB)"); a1.set_title("PSNR vs rank knob")
    a1.legend(frameon=False)
    a2.set_xscale("log"); a2.set_xlabel("dictionary parameters (log)")
    a2.set_ylabel("test PSNR (dB)"); a2.set_title("PSNR vs model size (rank sweep)")
    a2.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_rank.png"); plt.close(fig)


def plot_init(initres):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [initres[n] for n in ORDER]
    bp = ax.boxplot(data, labels=ORDER, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black"))
    for patch, n in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[n]); patch.set_alpha(0.65)
    ax.set_ylabel("test denoising PSNR vs clean (dB)")
    ax.set_title("Initialisation sensitivity across random seeds\n"
                 "(taller box = less robust)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_init.png"); plt.close(fig)


# ======================================================================================
# Summary tables
# ======================================================================================
def write_summary(rec, initres, ds):
    lines = []
    lines.append("| Model | Atoms | Dict params | Compression vs Vec | "
                 "Fit time (s) | Test PSNR (dB) | PSNR std (init) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    vec_params = np.mean(rec["Vectorized"]["params"])
    csv = ["model,atoms,dict_params,compression_vs_vec,fit_time_s,test_psnr_db,psnr_std_init"]
    for name in ORDER:
        atoms = int(np.mean(rec[name]["atoms"]))
        params = np.mean(rec[name]["params"])
        comp = vec_params / params
        ft = np.mean(rec[name]["fit_time"])
        p = np.mean(rec[name]["psnr"])
        sd = np.std(initres[name])
        lines.append(f"| {name} | {atoms} | {int(params)} | {comp:5.1f}x | "
                     f"{ft:5.2f} | {p:5.2f} | {sd:4.2f} |")
        csv.append(f"{name},{atoms},{int(params)},{comp:.2f},{ft:.3f},{p:.3f},{sd:.3f}")
    header = (f"# Tensor Dictionary Learning Benchmark — Results\n\n"
              f"Dataset: mixed-structure 3rd-order patches {ds['atom_shape']}, "
              f"SNR={ds['snr_db']} dB, true sparsity={ds['s_true']}, "
              f"encode sparsity={ENCODE_S}.\n\n")
    md = header + "\n".join(lines) + "\n"
    open(f"{OUT}/results_summary.md", "w").write(md)
    open(f"{OUT}/results_summary.csv", "w").write("\n".join(csv) + "\n")
    print("\n" + md)


# ======================================================================================
def main():
    t0 = time.time()
    seeds = [0, 1, 2, 3, 4]
    sparsity_grid = [2, 4, 6, 8, 10]
    print("Generating dataset ...")
    ds = make_dataset(atom_shape=ATOM_SHAPE, n_true=24, n_train=320, n_test=90,
                      s_true=3, snr_db=12.0, mix=(0.4, 0.4, 0.2), seed=123)

    rec = experiment_frontier_and_sparsity(ds, seeds, sparsity_grid)
    rankres = experiment_rank(ds, seeds[:3])          # 3 seeds to bound runtime
    initres = experiment_init(ds, [10, 11, 12, 13, 14, 15, 16, 17])

    plot_frontier(rec)
    plot_sparsity(rec, sparsity_grid)
    plot_rank(rankres)
    plot_init(initres)
    write_summary(rec, initres, ds)

    json.dump({"rank": {k: {kk: list(map(float, vv)) if isinstance(vv, list) else vv
                            for kk, vv in d.items()} for k, d in rankres.items()}},
              open(f"{OUT}/rank_results.json", "w"), indent=2)
    print(f"\nDONE in {time.time()-t0:.1f}s. Figures + tables in {OUT}")


if __name__ == "__main__":
    main()
