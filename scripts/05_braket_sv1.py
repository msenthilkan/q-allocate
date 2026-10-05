"""Run the full 32-qubit QAOA circuit on Amazon Braket SV1 (managed statevector simulator).

A 32-qubit statevector does not fit in Colab's free tier, so this script submits
the circuit to Braket's SV1 (up to 34 qubits). Angles are transferred from the
small-instance runs (QAOA angles on normalised Hamiltonians concentrate across
similar instances), so only one SV1 task is needed per angle setting.

    # local check on a small instance (free, no AWS account needed)
    python scripts/05_braket_sv1.py --local --small

    # 32-qubit run on SV1 (needs AWS credentials and Braket access; billed per task-minute)
    python scripts/05_braket_sv1.py --device sv1 --shots 1000

Capacity couples the categories in the 32-variable instance. In the monolithic
QUBO it is priced with the Lagrange multipliers from results/decomposition_32.json
(if present), and every returned bitstring is checked against the true capacities.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qallocate import (Instance, generate, build_qubo, qubo_to_ising, solve_milp, evaluate_samples)

SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"


def braket_circuit(ising, gammas, betas):
    from braket.circuits import Circuit
    s = ising.scale()
    c = Circuit()
    for j in range(ising.n):
        c.h(j)
    for g, b in zip(gammas, betas):
        for j in range(ising.n):
            if ising.h[j]:
                c.rz(j, 2 * g * ising.h[j] / s)          # exp(-i g h Z)
        for (j, k), v in ising.J.items():
            c.zz(j, k, 2 * g * v / s)                    # exp(-i g J ZZ)
        for j in range(ising.n):
            c.rx(j, 2 * b)                               # exp(-i b X)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="use the free local Braket simulator")
    ap.add_argument("--small", action="store_true", help="12-qubit instance instead of the 32-qubit one")
    ap.add_argument("--device", default="sv1")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--gamma", type=float, nargs="+", default=None)
    ap.add_argument("--beta", type=float, nargs="+", default=None)
    a = ap.parse_args()

    if a.small:
        inst = generate(2, 3, bits=2, seed=0)
        lam = np.zeros(inst.n_sup)
    else:
        path = ROOT / "results" / "instance_32.json"
        inst = Instance.load(path) if path.exists() else generate(4, 4, bits=2, capacity=True, seed=5)
        dec = ROOT / "results" / "decomposition_32.json"
        lam = np.zeros(inst.n_sup)
        if dec.exists():
            lam = np.array(json.loads(dec.read_text())["runs"].get("exact", {}).get("log", [{}])[-1].get("lambda", lam))
    priced = inst.subset(list(range(inst.n_cat)), cost_adjust=lam)
    qubo = build_qubo(priced, float(priced.price.max()))
    ising = qubo_to_ising(qubo)

    # default angles: transferred from the AerSimulator p=1 run on the 12-qubit instance
    if a.gamma is None:
        aer = ROOT / "results" / "aer_hybrid_p1.json"
        prev = json.loads(aer.read_text()) if aer.exists() else {"gammas": [0.3], "betas": [0.4]}
        a.gamma, a.beta = prev["gammas"], prev["betas"]
    circ = braket_circuit(ising, a.gamma, a.beta)
    print(f"{ising.n}-qubit QAOA circuit, p={len(a.gamma)}, {len(circ.instructions)} gates")

    if a.local:
        from braket.devices import LocalSimulator
        if ising.n > 22:
            sys.exit("The local simulator is not suitable for more than ~22 qubits; use --device sv1")
        result = LocalSimulator().run(circ, shots=a.shots).result()
    else:
        from braket.aws import AwsDevice
        device = AwsDevice(SV1_ARN if a.device == "sv1" else a.device)
        result = device.run(circ, shots=a.shots).result()

    # Braket bitstrings list qubit 0 first; our index uses bit j for qubit j
    counts = {int(k[::-1], 2): v for k, v in result.measurement_counts.items()}
    ev = evaluate_samples(build_qubo(inst, float(inst.price.max())), counts)   # true constraints
    milp = solve_milp(inst)
    b = ev["best_feasible"]
    print(f"feasible fraction {ev['feasible_fraction']:.4f}; best feasible "
          f"{'none' if b is None else f'{b['cost']:.1f}'} vs MILP {milp['cost']:.1f}"
          f"{'' if b is None else f' (ratio {milp['cost'] / b['cost']:.4f})'}")


if __name__ == "__main__":
    main()
