"""QAOA hybrid loop: classical SciPy optimiser updates (gamma, beta); a quantum
simulator evaluates the circuit.

Two interchangeable engines evaluate the same circuit:
  * "aer"   - Qiskit circuit (H, RZ, RZZ, RX) executed on qiskit-aer AerSimulator.
  * "numpy" - exact statevector kernel (diagonal cost phase + per-qubit RX mixer),
              used for parameter sweeps because it is much faster. tests/ checks
              that both engines agree to numerical precision.

The cost Hamiltonian is normalised by its largest coefficient so the same angle
ranges work for every instance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .ising import Ising, ising_diagonal, qubo_to_ising
from .qubo import QUBO, award_cost, bits_from_index, decode, violations


# ------------------------------------------------------------- numpy engine
def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = np.cos(beta), -1j * np.sin(beta)
    for j in range(n):
        v = psi.reshape(2 ** (n - j - 1), 2, 2 ** j)
        a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = c * a0 + s * a1
        v[:, 1, :] = s * a0 + c * a1
    return psi


def numpy_statevector(diag: np.ndarray, gammas, betas) -> np.ndarray:
    n = int(np.log2(len(diag)))
    psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * diag)
        psi = _apply_mixer(psi, b, n)
    return psi


# --------------------------------------------------------------- aer engine
class AerEngine:
    """Parameterised Qiskit QAOA circuit evaluated on AerSimulator."""

    def __init__(self, ising: Ising, p: int):
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import ParameterVector
        from qiskit_aer import AerSimulator

        self.n, self.p = ising.n, p
        s = ising.scale()
        self.gamma = ParameterVector("gamma", p)
        self.beta = ParameterVector("beta", p)
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        for layer in range(p):
            for j in range(self.n):
                if ising.h[j]:
                    qc.rz(2 * self.gamma[layer] * ising.h[j] / s, j)
            for (j, k), v in ising.J.items():
                qc.rzz(2 * self.gamma[layer] * v / s, j, k)
            for j in range(self.n):
                qc.rx(2 * self.beta[layer], j)
        self.circuit = qc
        self.sim = AerSimulator(method="statevector")
        sv = qc.copy()
        sv.save_statevector()
        self._sv = transpile(sv, self.sim)
        meas = qc.copy()
        meas.measure_all()
        self._meas = transpile(meas, self.sim)

    def _bind(self, circ, gammas, betas):
        binds = {**dict(zip(self.gamma, gammas)), **dict(zip(self.beta, betas))}
        return circ.assign_parameters(binds)

    def statevector(self, gammas, betas) -> np.ndarray:
        res = self.sim.run(self._bind(self._sv, gammas, betas)).result()
        return np.asarray(res.get_statevector())

    def counts(self, gammas, betas, shots: int, seed: int | None = None) -> dict[int, int]:
        res = self.sim.run(self._bind(self._meas, gammas, betas), shots=shots, seed_simulator=seed).result()
        return {int(k.replace(" ", ""), 2): c for k, c in res.get_counts().items()}


# --------------------------------------------------------------- optimiser
@dataclass
class QAOAResult:
    p: int
    gammas: np.ndarray
    betas: np.ndarray
    expectation: float                 # <H_C>, normalised units
    probs: np.ndarray
    history: list = field(default_factory=list)  # (eval, gammas, betas, <H>)
    evals: int = 0
    seconds: float = 0.0
    engine: str = "numpy"


def _interp(params: np.ndarray) -> np.ndarray:
    """INTERP initialisation: extend optimal p-layer angles to p+1 layers."""
    p = len(params)
    new = np.zeros(p + 1)
    for i in range(p + 1):
        left = params[i - 1] if i - 1 >= 0 else 0.0
        right = params[i] if i < p else 0.0
        new[i] = (i / p) * left + ((p - i) / p) * right
    return new


def optimise(ising: Ising, p: int, engine: str = "numpy", init: tuple | None = None,
             maxiter: int = 300, grid: int = 24, seed: int = 0, cvar_alpha: float = 1.0) -> QAOAResult:
    """Minimise <H_C> (cvar_alpha = 1) or CVaR_alpha (mean of the lowest alpha
    probability mass, Barkoutsos et al. 2020), which rewards a good tail of samples
    rather than a good average and usually improves the best sampled solution."""
    t0 = time.perf_counter()
    diag = ising_diagonal(ising) / ising.scale()
    order = np.argsort(diag)
    d_sorted = diag[order]

    def objective(probs):
        if cvar_alpha >= 1.0:
            return float(probs @ diag)
        ps = probs[order]
        cum = np.cumsum(ps)
        k = int(np.searchsorted(cum, cvar_alpha))
        head = float(ps[:k] @ d_sorted[:k])
        rest = cvar_alpha - (cum[k - 1] if k > 0 else 0.0)
        return (head + rest * d_sorted[min(k, len(d_sorted) - 1)]) / cvar_alpha
    aer = AerEngine(ising, p) if engine == "aer" else None
    history = []

    def state(g, b):
        return aer.statevector(g, b) if aer else numpy_statevector(diag, g, b)

    def f(theta):
        g, b = theta[:p], theta[p:]
        val = objective(np.abs(state(g, b)) ** 2)
        history.append((len(history) + 1, g.tolist(), b.tolist(), val))
        return val

    if init is not None:
        x0 = np.concatenate([np.asarray(init[0], float), np.asarray(init[1], float)])
    elif p == 1:
        # coarse grid for a good starting point, then local refinement
        best, x0 = np.inf, None
        for g in np.linspace(0.02, 1.0, grid):
            for b in np.linspace(0.05, np.pi / 2, max(6, grid // 2)):
                val = f(np.array([g, b]))
                if val < best:
                    best, x0 = val, np.array([g, b])
    else:
        rng = np.random.default_rng(seed)
        x0 = np.concatenate([rng.uniform(0, 0.5, p), rng.uniform(0, np.pi / 2, p)])

    res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.1})
    g, b = res.x[:p], res.x[p:]
    probs = np.abs(state(g, b)) ** 2
    return QAOAResult(p=p, gammas=g, betas=b, expectation=float(probs @ diag), probs=probs,
                      history=history, evals=len(history), seconds=time.perf_counter() - t0,
                      engine=engine)


def optimise_depths(ising: Ising, depths=(1, 2, 3), **kw) -> dict[int, QAOAResult]:
    """Optimise p = 1..max with INTERP warm starts between depths."""
    out, prev = {}, None
    for p in depths:
        init = None if prev is None else (_interp(prev.gammas), _interp(prev.betas))
        out[p] = optimise(ising, p, init=init, **kw)
        prev = out[p]
    return out


# ------------------------------------------------------------ evaluation
def sample(probs: np.ndarray, shots: int, seed: int = 0) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, probs / probs.sum())
    return {int(i): int(c) for i, c in enumerate(counts) if c}


def evaluate_samples(qubo: QUBO, counts: dict[int, int]) -> dict:
    inst, n = qubo.instance, qubo.n
    shots = sum(counts.values())
    feasible, best, returned = 0, None, None
    for idx, c in counts.items():
        x = bits_from_index(idx, n)
        q = decode(inst, x)
        e = qubo.energy(x)
        is_ok = violations(inst, q)["feasible"]
        if returned is None or e < returned["energy"]:
            returned = {"energy": e, "feasible": is_ok, "index": idx}
        if is_ok:
            feasible += c
            cost = award_cost(inst, q)
            if best is None or cost < best["cost"]:
                best = {"cost": cost, "q": q, "index": idx}
    # "returned" = lowest-energy sampled bitstring, i.e. what the solver hands back
    return {"shots": shots, "feasible_fraction": feasible / shots, "best_feasible": best,
            "returned_feasible": bool(returned and returned["feasible"])}


def run_qaoa(qubo: QUBO, p: int = 1, shots: int = 1024, engine: str = "numpy", seed: int = 0,
             **kw) -> tuple[QAOAResult, dict]:
    ising = qubo_to_ising(qubo)
    res = optimise(ising, p, engine=engine, seed=seed, **kw)
    if engine == "aer":
        counts = AerEngine(ising, p).counts(res.gammas, res.betas, shots, seed=seed)
    else:
        counts = sample(res.probs, shots, seed=seed)
    return res, evaluate_samples(qubo, counts)
