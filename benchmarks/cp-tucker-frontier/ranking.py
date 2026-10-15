"""
ranking.py -- an efficiency ranking/classification system for tensor decompositions
              used as dictionary-learning atom structures.

WHY A SYSTEM AND NOT A SINGLE NUMBER
------------------------------------
"Efficiency" of a decomposition for DL is inherently multi-objective:
    Accuracy        reconstruction quality (test PSNR, dB)
    Parameter eff.  how little it stores (dictionary parameters)
    Speed           training cost (fit time)
    Robustness      stability across random seeds (PSNR std)
No scalar captures all four, and any weighting is a value judgement. So the system
has THREE layers, from most objective to most opinionated:

  1. Pareto layer (weight-free, objective)
     A method is DOMINATED if another is >= on every axis and > on one. Dominated
     methods cannot be "most efficient" under ANY non-negative weighting, so they are
     classified out of contention. Non-dominated methods form the frontier.

  2. Interpretable scalar
     E = PSNR / log10(params)  ->  "accuracy retained per decade of parameters".
     A single, unit-ful efficiency number that doesn't rely on normalisation.

  3. Profile-weighted composite + letter tier (opinionated but transparent)
     Min-max normalise each axis to [0,1] (higher=better), take a weighted sum under
     a named profile (Balanced / Quality / Compression / Realtime), grade S-D.

IMPORTANT HONESTY NOTE
----------------------
This ranks THESE implementations (projection-based K-SVD for CP/TT/TR/BTD, separable
Kronecker for Tucker) on THIS benchmark (one image, one noise level, one operating
point). The *methodology* is the reusable part; the numbers should be re-derived on
your own data before being quoted as properties of the decompositions themselves.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import make_image_dataset
from tensor_dl import make_model, psnr

OUT = "/home/claude/outputs"
SHAPE, NA, ENCODE_S, SEEDS = (8, 8, 3), 100, 6, [0, 1, 2, 3, 4]
COLORS = {"Vectorized": "#444444", "CP": "#d1495b", "Tucker": "#2e8b57",
          "TT": "#3b6fb6", "TR": "#e08e0b", "BTD": "#7b4fb6"}

PROFILES = {   # weights over (accuracy, param_eff, speed, robustness); each sums to 1
    "Balanced":    (0.40, 0.30, 0.15, 0.15),
    "Quality":     (0.60, 0.20, 0.10, 0.10),   # offline / server, accuracy matters most
    "Compression": (0.25, 0.55, 0.10, 0.10),   # edge / embedded, model size matters most
    "Realtime":    (0.30, 0.20, 0.40, 0.10),   # latency-sensitive training
}
TIER_CUTOFFS = [("S", 0.80), ("A", 0.65), ("B", 0.45), ("C", 0.25), ("D", 0.0)]


def specs(sd):
    return {
        "Vectorized": ("vectorized", dict(n_atoms=NA, sparsity=4, n_iter=10, seed=sd)),
        "CP":         ("cp", dict(n_atoms=NA, sparsity=4, n_iter=10, rank=1, seed=sd)),
        "Tucker":     ("tucker", dict(mode_ranks=[9, 9, 3], n_iter=6, mode_sparsity=2, seed=sd)),
        "TT":         ("tt", dict(n_atoms=NA, sparsity=4, n_iter=10, rank=2, seed=sd)),
        "TR":         ("tr", dict(n_atoms=NA, sparsity=4, n_iter=10, rank=2, seed=sd)),
        "BTD":        ("btd", dict(n_atoms=NA, sparsity=4, n_iter=10, rank=2,
                                   proj_kw=dict(n_blocks=1), seed=sd)),
    }


def measure(ds):
    """Fit every method over SEEDS; collect accuracy/params/time/robustness."""
    names = list(specs(0).keys())
    raw = {}
    for n in names:
        ps, prm, ft = [], [], []
        for sd in SEEDS:
            kind, kw = specs(sd)[n]
            m = make_model(kind, SHAPE, **kw).fit(ds["train_noisy"])
            ps.append(psnr(ds["test_clean"], m.reconstruct(ds["test_noisy"], ENCODE_S)))
            prm.append(m.n_params); ft.append(m.fit_time_)
        raw[n] = dict(psnr=float(np.mean(ps)), psnr_std=float(np.std(ps)),
                      params=int(np.mean(prm)), fit_time=float(np.mean(ft)))
        print(f"   {n:11s} PSNR {raw[n]['psnr']:5.2f}±{raw[n]['psnr_std']:.2f}  "
              f"params {raw[n]['params']:6d}  fit {raw[n]['fit_time']:4.2f}s")
    return raw


def _minmax(vals):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def build_system(raw):
    names = list(raw.keys())
    # higher-is-better raw axis values
    acc = [raw[n]["psnr"] for n in names]
    peff = [-np.log10(raw[n]["params"]) for n in names]      # fewer params -> higher
    spd = [-np.log10(raw[n]["fit_time"]) for n in names]     # faster -> higher
    rob = [-raw[n]["psnr_std"] for n in names]               # more stable -> higher
    A, P, S, R = map(_minmax, (acc, peff, spd, rob))
    axes = {n: dict(accuracy=A[i], param_eff=P[i], speed=S[i], robustness=R[i])
            for i, n in enumerate(names)}

    # interpretable scalar E = PSNR per decade of parameters
    E = {n: raw[n]["psnr"] / np.log10(raw[n]["params"]) for n in names}

    # Pareto layer = rate-distortion frontier on the two CORE axes (accuracy, param_eff).
    # (A 4-axis Pareto is non-discriminating here: nearly every method wins some axis.)
    def dominated(a, b):  # does b dominate a on accuracy & parameter-efficiency?
        ge = (axes[b]["accuracy"] >= axes[a]["accuracy"] - 1e-9 and
              axes[b]["param_eff"] >= axes[a]["param_eff"] - 1e-9)
        gt = (axes[b]["accuracy"] > axes[a]["accuracy"] + 1e-9 or
              axes[b]["param_eff"] > axes[a]["param_eff"] + 1e-9)
        return ge and gt
    pareto = {n: not any(dominated(n, o) for o in names if o != n) for n in names}

    # profile composites + tiers
    composites = {n: {} for n in names}
    for prof, (wa, wp, ws, wr) in PROFILES.items():
        for n in names:
            composites[n][prof] = (wa * axes[n]["accuracy"] + wp * axes[n]["param_eff"]
                                   + ws * axes[n]["speed"] + wr * axes[n]["robustness"])

    def tier(score):
        for t, c in TIER_CUTOFFS:
            if score >= c:
                return t
        return "D"

    rows = []
    for n in names:
        rows.append(dict(name=n, **raw[n], axes=axes[n], E=round(E[n], 2),
                         pareto=pareto[n], composites=composites[n],
                         tier_balanced=tier(composites[n]["Balanced"])))
    rows.sort(key=lambda r: r["composites"]["Balanced"], reverse=True)
    return rows


def print_tables(rows):
    print("\n=== EFFICIENCY RANKING (Balanced profile) ===")
    print(f"{'rank':>4} {'method':11} {'tier':>4} {'PSNR':>6} {'params':>7} "
          f"{'E(dB/dec)':>9} {'Pareto':>7} {'composite':>9}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>4} {r['name']:11} {r['tier_balanced']:>4} {r['psnr']:6.2f} "
              f"{r['params']:7d} {r['E']:9.2f} {'FRONT' if r['pareto'] else 'dom':>7} "
              f"{r['composites']['Balanced']:9.3f}")

    print("\n=== WINNER BY USE-CASE PROFILE ===")
    for prof in PROFILES:
        best = max(rows, key=lambda r: r["composites"][prof])
        order = sorted(rows, key=lambda r: r["composites"][prof], reverse=True)
        chain = " > ".join(r["name"] for r in order)
        print(f"   {prof:11}: {best['name']:11}   ({chain})")


def make_figure(rows):
    names = [r["name"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (1) Pareto scatter: accuracy vs params, frontier highlighted
    for r in rows:
        mk = "*" if r["pareto"] else "o"
        sz = 360 if r["pareto"] else 150
        ax1.scatter(r["params"], r["psnr"], marker=mk, s=sz,
                    color=COLORS[r["name"]], edgecolor="black",
                    linewidth=1.2 if r["pareto"] else 0.5, zorder=3)
        ax1.annotate(" " + r["name"], (r["params"], r["psnr"]),
                     color=COLORS[r["name"]], fontweight="bold", fontsize=9, va="center")
    front = sorted([r for r in rows if r["pareto"]], key=lambda r: r["params"])
    ax1.plot([r["params"] for r in front], [r["psnr"] for r in front],
             "k--", lw=1, alpha=0.6, zorder=1, label="Pareto frontier")
    ax1.set_xscale("log")
    ax1.set_xlabel("dictionary parameters (log)"); ax1.set_ylabel("test PSNR (dB)")
    ax1.set_title("Objective layer: Pareto frontier\n(stars = non-dominated)")
    ax1.legend(frameon=False, loc="lower right"); ax1.grid(alpha=0.25)

    # (2) profile composite heatmap
    profs = list(PROFILES.keys())
    M = np.array([[r["composites"][p] for p in profs] for r in rows])
    im = ax2.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax2.set_xticks(range(len(profs))); ax2.set_xticklabels(profs, rotation=20, ha="right")
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(profs)):
            ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                     color="white" if M[i, j] < 0.6 else "black", fontsize=9)
    ax2.set_title("Opinionated layer: composite score by use-case profile")
    fig.colorbar(im, ax=ax2, fraction=0.046, label="composite (0-1)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig8_ranking.png"); plt.close(fig)


def main():
    print("Measuring all methods on real RGB patches (5 seeds) ...")
    ds = make_image_dataset("astronaut", patch=SHAPE, n_train=320, n_test=90,
                            snr_db=12.0, seed=7)
    raw = measure(ds)
    rows = build_system(raw)
    print_tables(rows)
    make_figure(rows)
    json.dump({"profiles": PROFILES, "rows": rows},
              open(f"{OUT}/ranking.json", "w"), indent=2)
    print(f"\nSaved fig8_ranking.png and ranking.json to {OUT}")


if __name__ == "__main__":
    main()
