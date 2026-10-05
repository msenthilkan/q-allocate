"""QUBO -> Ising cost Hamiltonian, H_C = sum J_jk Z_j Z_k + sum h_j Z_j + offset.

Substitution x_j = (1 - Z_j) / 2, with Z_j eigenvalues +1 (x=0) and -1 (x=1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qubo import QUBO


@dataclass
class Ising:
    h: np.ndarray          # (n,)
    J: dict                # {(j, k): J_jk} with j < k
    offset: float

    @property
    def n(self) -> int:
        return len(self.h)

    def energy_z(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        e = self.offset + float(self.h @ z)
        for (j, k), v in self.J.items():
            e += v * z[j] * z[k]
        return e

    def scale(self) -> float:
        """Largest absolute coefficient, used to normalise QAOA angles."""
        m = np.abs(self.h).max() if len(self.h) else 0.0
        if self.J:
            m = max(m, max(abs(v) for v in self.J.values()))
        return float(m) or 1.0

    def to_sparse_pauli_op(self, normalise: bool = True):
        """Qiskit SparsePauliOp of H_C without the constant offset."""
        from qiskit.quantum_info import SparsePauliOp
        s = self.scale() if normalise else 1.0
        terms = [("Z", [j], self.h[j] / s) for j in range(self.n) if self.h[j]]
        terms += [("ZZ", [j, k], v / s) for (j, k), v in self.J.items() if v]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=self.n)


def qubo_to_ising(qubo: QUBO) -> Ising:
    Q = np.triu(qubo.Q)
    n = Q.shape[0]
    diag = np.diag(Q).copy()
    off = Q - np.diag(diag)
    h = -diag / 2 - (off.sum(axis=1) + off.sum(axis=0)) / 4
    J = {(j, k): off[j, k] / 4 for j in range(n) for k in range(j + 1, n) if off[j, k]}
    offset = qubo.offset + diag.sum() / 2 + off.sum() / 4
    return Ising(h=h, J=J, offset=float(offset))


def ising_diagonal(ising: Ising) -> np.ndarray:
    """H_C (without offset) evaluated on every basis state; z_j = 1 - 2 x_j."""
    n = ising.n
    states = np.arange(2 ** n, dtype=np.uint32)
    Z = [1.0 - 2.0 * ((states >> j) & 1) for j in range(n)]
    E = np.zeros(2 ** n)
    for j in range(n):
        if ising.h[j]:
            E += ising.h[j] * Z[j]
    for (j, k), v in ising.J.items():
        E += v * Z[j] * Z[k]
    return E
