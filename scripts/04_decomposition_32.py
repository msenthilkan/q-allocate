"""32-variable problem (4 categories x 4 suppliers x 2 bits) with supplier capacities.

A full 32-qubit statevector needs ~64 GB, so the problem is split by Lagrangian
decomposition into two 16-variable sub-QUBOs (2 categories each). The classical
coordinator prices supplier capacity; QAOA solves each sub-QUBO. The same loop
with exact sub-solvers gives a reference and a dual lower bound.

    python scripts/04_decomposition_32.py               # exact + QAOA (~2-4 min)
    python scripts/04_decomposition_32.py --solver exact
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qallocate import generate, solve_milp, lagrangian_decomposition, violations

SEED = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["exact", "qaoa", "both"], default="both")
    ap.add_argument("--iterations", type=int, default=12)
    a = ap.parse_args()

    inst = generate(4, 4, bits=2, capacity=True, seed=SEED, name="scale_32")
    inst.save(ROOT / "results" / "instance_32.json")
    milp = solve_milp(inst)
    relaxed = solve_milp(generate(4, 4, bits=2, capacity=False, seed=SEED))
    print(f"32-variable instance: MILP optimum {milp['cost']:.1f} in {milp['seconds']*1000:.0f} ms "
          f"(without capacity: {relaxed['cost']:.1f}, so capacity is binding)")

    out = {"instance": inst.name, "n_qubits": inst.n_vars, "milp_cost": milp["cost"],
           "milp_seconds": milp["seconds"], "milp_without_capacity": relaxed["cost"], "runs": {}}
    solvers = ["exact", "qaoa"] if a.solver == "both" else [a.solver]
    for s in solvers:
        res = lagrangian_decomposition(inst, block_size=2, iterations=a.iterations, solver=s,
                                       p=1, shots=2048, seed=SEED)
        best = res["best"]
        ratio = None if best is None else milp["cost"] / best["cost"]
        print(f"[{s}] sub-QUBO sizes {res['sub_qubo_sizes']}, best feasible "
              f"{'none' if best is None else f'{best['cost']:.1f} (iteration {best['iteration']})'}, "
              f"approx ratio {'n/a' if ratio is None else f'{ratio:.4f}'}, {res['seconds']:.1f} s")
        if best is not None:
            assert violations(inst, best["q"])["feasible"]
        out["runs"][s] = {"approx_ratio": ratio, "best_cost": None if best is None else best["cost"],
                          "best_q": None if best is None else np.asarray(best["q"]).tolist(),
                          "seconds": res["seconds"], "sub_qubo_sizes": res["sub_qubo_sizes"],
                          "log": res["log"]}
    path = ROOT / "results" / "decomposition_32.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update({k: v for k, v in out.items() if k != "runs"})
    prev.setdefault("runs", {}).update(out["runs"])
    path.write_text(json.dumps(prev, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
