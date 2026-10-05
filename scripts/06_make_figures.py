"""Draw the figures in docs/images from the JSON files in results/."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R, IMG = ROOT / "results", ROOT / "docs" / "images"
IMG.mkdir(parents=True, exist_ok=True)
INK, Q, C, GREY, RED = "#1B2430", "#5B3FD1", "#0E8C7A", "#9AA3AD", "#C8463D"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. small-instance approximation ratios
rows = json.loads((R / "small_benchmark.json").read_text())
fig, ax = plt.subplots(figsize=(8, 4))
ns = sorted({r["n_qubits"] for r in rows})
for k, (key, label, col, dx) in enumerate([("qaoa_approx_ratio", "QAOA p=1 (best of 1024 shots)", Q, -0.6),
                                           ("random_approx_ratio", "Uniform random (1024 shots)", GREY, 0.6)]):
    for n in ns:
        vals = [r[key] for r in rows if r["n_qubits"] == n]
        found = [v for v in vals if v is not None]
        ax.scatter([n + dx] * len(found), found, color=col, s=40, label=label if n == ns[0] else None, zorder=3)
        miss = len(vals) - len(found)
        if miss:
            ax.scatter([n + dx] * miss, [0.905] * miss, marker="x", color=col, s=40,
                       label="no feasible sample" if (key.startswith("random") and n == 12) else None)
ax.axhline(0.95, color=RED, ls="--", lw=1, label="95% target")
ax.set_xticks(ns); ax.set_xlabel("Problem size (qubits)"); ax.set_ylabel("Approximation ratio (MILP / found)")
ax.set_ylim(0.89, 1.01); ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("Small instances: best feasible award found vs exact MILP", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(IMG / "approx_ratio_small.png", dpi=150); plt.close(fig)

# 2. depth scaling
d = json.loads((R / "depth_scaling.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
groups = defaultdict(list)
for r in d:
    groups[(r["n_qubits"], r["seed"])].append(r)
for (n, s), rs in groups.items():
    rs = sorted(rs, key=lambda r: r["p"])
    col = Q if n == 12 else C
    axes[0].plot([r["p"] for r in rs], [r["p_optimal"] / r["p_optimal_uniform"] for r in rs], marker="o", color=col,
                 alpha=0.8, label=f"N={n}" if s == 0 else None)
    axes[1].plot([r["p"] for r in rs], [r["p_feasible"] / r["p_feasible_uniform"] for r in rs], marker="o", color=col,
                 alpha=0.8, label=f"N={n}" if s == 0 else None)
for ax, t in zip(axes, ["P(optimal) vs uniform sampling", "P(feasible) vs uniform sampling"]):
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("QAOA depth p"); ax.set_yscale("log"); ax.set_title(t, loc="left", fontsize=10)
    ax.axhline(1, color=GREY, lw=1, ls=":"); ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Amplification (x)")
fig.suptitle("Depth scaling on the simulator (3 instances per size)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(IMG / "depth_scaling.png", dpi=150); plt.close(fig)

# 3. Aer hybrid convergence
fig, ax = plt.subplots(figsize=(8, 3.6))
for p, col in ((1, Q), (2, C)):
    f = R / f"aer_hybrid_p{p}.json"
    if f.exists():
        h = json.loads(f.read_text())["history"]
        vals = np.minimum.accumulate([x[3] for x in h])
        ax.plot(range(1, len(vals) + 1), vals, color=col, label=f"p={p}")
ax.set_xlabel("Circuit evaluations on AerSimulator"); ax.set_ylabel("Best CVaR objective so far")
ax.set_title("Hybrid loop: SciPy updates (gamma, beta), Aer evaluates the circuit (12 qubits)", loc="left", fontweight="bold", fontsize=10)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(IMG / "aer_convergence.png", dpi=150); plt.close(fig)

# 4. decomposition
dec = json.loads((R / "decomposition_32.json").read_text())
fig, ax = plt.subplots(figsize=(8, 3.8))
for name, col in (("exact", C), ("qaoa", Q)):
    if name in dec["runs"]:
        lg = dec["runs"][name]["log"]
        ax.plot([l["iteration"] for l in lg], [l["best_cost"] for l in lg], marker="o", color=col,
                label=f"best feasible ({'exact' if name == 'exact' else 'QAOA'} sub-solver)")
        if name == "exact":
            ax.plot([l["iteration"] for l in lg], [l["dual_bound"] for l in lg], ls="--", color=col, alpha=0.6,
                    label="Lagrangian lower bound")
ax.axhline(dec["milp_cost"], color=RED, lw=1, label="MILP optimum")
ax.set_xlabel("Coordinator iteration"); ax.set_ylabel("Award cost")
ax.set_title("32-variable problem: two 16-qubit sub-QUBOs + classical coordinator", loc="left", fontweight="bold", fontsize=10)
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(IMG / "decomposition_32.png", dpi=150); plt.close(fig)
print("figures written to", IMG)
