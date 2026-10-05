"""Hybrid loop demonstration on Qiskit AerSimulator (success criterion 3).

The classical optimiser (SciPy COBYLA) proposes (gamma, beta); every proposal is
executed as a Qiskit circuit on AerSimulator; the measured objective is fed back.
The run log records every angle update so convergence can be plotted.

    python scripts/03_aer_hybrid_loop.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qallocate import generate, build_qubo, qubo_to_ising, solve_milp, evaluate_samples
from qallocate.qaoa import AerEngine, optimise


def main():
    inst = generate(2, 3, bits=2, seed=0)          # 12 qubits
    milp = solve_milp(inst)
    qubo = build_qubo(inst, float(inst.price.max()))
    ising = qubo_to_ising(qubo)
    print(f"{inst.n_vars}-qubit instance, MILP optimum {milp['cost']:.1f}")
    print("Ising cost Hamiltonian (normalised) has", len(ising.J), "ZZ couplings and",
          int((ising.h != 0).sum()), "local fields")

    for p in (1, 2):
        init = None if p == 1 else ([res.gammas[0] / 2, res.gammas[0]], [res.betas[0], res.betas[0] / 2])
        res = optimise(ising, p, engine="aer", init=init, cvar_alpha=0.1, maxiter=150, grid=16)
        counts = AerEngine(ising, p).counts(res.gammas, res.betas, shots=2048, seed=7)
        ev = evaluate_samples(qubo, counts)
        b = ev["best_feasible"]
        first, last = res.history[0], res.history[-1]
        print(f"p={p}: {res.evals} circuit evaluations on AerSimulator in {res.seconds:.1f}s; "
              f"objective {first[3]:.3f} -> {min(h[3] for h in res.history):.3f}; "
              f"gamma={[round(float(g), 3) for g in res.gammas]}, beta={[round(float(x), 3) for x in res.betas]}; "
              f"best sample ratio={'n/a' if b is None else round(milp['cost'] / b['cost'], 4)}")
        out = {"n_qubits": inst.n_vars, "p": p, "milp_cost": milp["cost"], "evals": res.evals,
               "seconds": res.seconds, "gammas": res.gammas.tolist(), "betas": res.betas.tolist(),
               "best_sample_ratio": None if b is None else milp["cost"] / b["cost"],
               "feasible_fraction": ev["feasible_fraction"], "history": res.history}
        (ROOT / "results" / f"aer_hybrid_p{p}.json").write_text(json.dumps(out, indent=2))
    qc = AerEngine(ising, 1).circuit
    print(f"p=1 circuit: {qc.num_qubits} qubits, depth {qc.depth()}, gates {dict(qc.count_ops())}")


if __name__ == "__main__":
    main()
