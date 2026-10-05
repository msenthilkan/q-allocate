"""QAOA depth scaling on the simulator (guideline 2.3): p = 1 (hardware target), 2 and 3.

For each instance the penalty is fixed at max P_iv, angles for p+1 are warm-started
from p (INTERP), and we report the probability of sampling a feasible award, the
probability of sampling the optimum (relative to uniform sampling), and the best
sampled approximation ratio with 1024 shots.

    python scripts/02_depth_scaling.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qallocate import (generate, build_qubo, qubo_to_ising, all_energies, feasible_mask, solve_milp,
                       optimise_depths, sample, evaluate_samples)

SIZES = {12: (2, 3, 2), 16: (2, 4, 2)}
SEEDS = range(3)
SHOTS = 1024


def main():
    rows = []
    for n, cfg in SIZES.items():
        for seed in SEEDS:
            inst = generate(*cfg, seed=seed)
            milp = solve_milp(inst)
            qubo = build_qubo(inst, float(inst.price.max()))
            ising = qubo_to_ising(qubo)
            E = all_energies(qubo)
            mask = feasible_mask(inst, qubo.n)
            Ef = np.where(mask, E, np.inf)
            opt = np.isclose(Ef, Ef.min())
            res = optimise_depths(ising, (1, 2, 3), cvar_alpha=0.1, seed=seed)
            for p, r in res.items():
                ev = evaluate_samples(qubo, sample(r.probs, SHOTS, seed=seed))
                b = ev["best_feasible"]
                rows.append({"n_qubits": n, "seed": seed, "p": p,
                             "p_feasible": float(r.probs[mask].sum()),
                             "p_feasible_uniform": float(mask.mean()),
                             "p_optimal": float(r.probs[opt].sum()),
                             "p_optimal_uniform": float(opt.mean()),
                             "best_sample_ratio": None if b is None else milp["cost"] / b["cost"],
                             "evals": r.evals, "seconds": r.seconds,
                             "gammas": r.gammas.tolist(), "betas": r.betas.tolist()})
                x = rows[-1]
                print(f"N={n} seed={seed} p={p}: P(feasible)={x['p_feasible']:.4f} "
                      f"(uniform {x['p_feasible_uniform']:.4f})  P(opt)/uniform={x['p_optimal']/x['p_optimal_uniform']:.1f}x  "
                      f"best ratio={'n/a' if b is None else f'{x['best_sample_ratio']:.3f}'}  {r.seconds:.1f}s")
    (ROOT / "results" / "depth_scaling.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
