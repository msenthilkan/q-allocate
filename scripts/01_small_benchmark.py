"""Small instances (N <= 20 qubits): validity, MILP baseline, QAOA p=1 with penalty tuning.

Covers success criteria 1 (mathematical validity), 2 (classical baseline) and
4 (approximation ratio >= 95% for N <= 20). Every QAOA result is reported next
to the exact MILP optimum and next to a uniform-random sampling baseline with the
same number of shots, so any advantage of the variational state is visible.

    python scripts/01_small_benchmark.py            # full run (~5 min on a laptop / Colab)
    python scripts/01_small_benchmark.py --quick    # 2 sizes, 1 seed
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qallocate import (generate, build_qubo, all_energies, feasible_mask, solve_milp, tune_penalty,
                       sample, evaluate_samples, qubo_to_ising, ising_diagonal)
from qallocate.postprocess import refine_samples

CONFIGS = {8: (1, 4, 2), 12: (2, 3, 2), 16: (2, 4, 2), 20: (2, 5, 2)}   # N qubits -> (categories, suppliers, bits)
SHOTS = 1024
CVAR_ALPHA = 0.1


def run_one(nc, ns, bits, seed):
    inst = generate(nc, ns, bits=bits, seed=seed)
    milp = solve_milp(inst)

    # Criterion 1: QUBO ground state (penalty = max P_iv) is feasible and equals the MILP optimum
    qubo0 = build_qubo(inst, float(inst.price.max()))
    E = all_energies(qubo0)
    mask = feasible_mask(inst, qubo0.n)
    gs = int(np.argmin(E))
    ising = qubo_to_ising(qubo0)
    ising_ok = bool(np.allclose(ising_diagonal(ising) + ising.offset, E))

    # Criteria 3/4: hybrid QAOA (p=1) with dynamic penalty tuning
    t0 = time.perf_counter()
    tuned = tune_penalty(inst, p=1, shots=SHOTS, seed=seed, cvar_alpha=CVAR_ALPHA)
    qaoa_s = time.perf_counter() - t0
    best = tuned["eval"]["best_feasible"]
    probs = tuned["result"].probs

    # probability mass on optimal bitstrings vs uniform
    Et = all_energies(tuned["qubo"])
    Ef = np.where(mask, Et, np.inf)
    opt = np.isclose(Ef, Ef.min())
    p_opt, p_opt_uniform = float(probs[opt].sum()), float(opt.sum() / len(opt))
    p_feas, p_feas_uniform = float(probs[mask].sum()), float(mask.mean())

    # uniform random baseline with the same shots
    q_counts = sample(probs, SHOTS, seed=seed)
    r_counts = sample(np.full(len(probs), 1 / len(probs)), SHOTS, seed=seed + 999)
    rnd = evaluate_samples(tuned["qubo"], r_counts)
    rb = rnd["best_feasible"]
    q_ls = refine_samples(tuned["qubo"], q_counts)
    r_ls = refine_samples(tuned["qubo"], r_counts)
    ratio = lambda b: None if b is None else milp["cost"] / b["cost"]

    return {
        "instance": inst.name, "n_qubits": inst.n_vars, "seed": seed,
        "milp_cost": milp["cost"], "milp_seconds": milp["seconds"],
        "qubo_ground_state_feasible": bool(mask[gs]),
        "qubo_ground_state_equals_milp": bool(np.isclose(E[gs], milp["cost"])),
        "ising_equivalent": ising_ok,
        "feasible_states": int(mask.sum()), "total_states": int(len(mask)),
        "penalty_final": tuned["penalty"], "penalty_rounds": len(tuned["history"]),
        "qaoa_seconds": qaoa_s, "qaoa_evals": tuned["result"].evals,
        "gamma": tuned["result"].gammas.tolist(), "beta": tuned["result"].betas.tolist(),
        "qaoa_best_cost": None if best is None else best["cost"],
        "qaoa_approx_ratio": None if best is None else milp["cost"] / best["cost"],
        "qaoa_feasible_fraction": tuned["eval"]["feasible_fraction"],
        "p_feasible": p_feas, "p_feasible_uniform": p_feas_uniform,
        "p_optimal": p_opt, "p_optimal_uniform": p_opt_uniform,
        "random_best_cost": None if rb is None else rb["cost"],
        "random_approx_ratio": ratio(rb),
        "qaoa_plus_local_search_ratio": ratio(q_ls),
        "random_plus_local_search_ratio": ratio(r_ls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-list", type=int, nargs="+", help="run only these seeds")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    sizes = [8, 12] if a.quick else a.sizes
    seeds = a.seed_list or range(1 if a.quick else a.seeds)
    out = Path(__file__).resolve().parents[1] / "results" / "small_benchmark.json"
    rows = [r for r in (json.loads(out.read_text()) if out.exists() and not a.quick else [])
            if not (r["n_qubits"] in sizes and r["seed"] in seeds)]
    for cfg in [CONFIGS[n] for n in sizes]:
        for s in seeds:
            r = run_one(*cfg, seed=s)
            rows.append(r)
            ar = r["qaoa_approx_ratio"]
            rr = r["random_approx_ratio"]
            print(f"N={r['n_qubits']:>2} seed={s}  MILP={r['milp_cost']:>10.1f} ({r['milp_seconds']*1000:.0f} ms)  "
                  f"GS=MILP:{r['qubo_ground_state_equals_milp']}  "
                  f"QAOA ratio={'n/a' if ar is None else f'{ar:.3f}'}  random ratio={'n/a' if rr is None else f'{rr:.3f}'}  "
                  f"P(opt)={r['p_optimal']:.2e} vs uniform {r['p_optimal_uniform']:.2e}")
            out.write_text(json.dumps(sorted(rows, key=lambda r: (r["n_qubits"], r["seed"])), indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
