"""QUBO formulation of the split-award problem with binary quantity encoding.

    min  sum_iv c_iv q_iv  +  P * sum_i ( sum_v q_iv - D_i )^2
    q_iv = lot_iv * sum_b 2^b x_ivb

Energy convention: E(x) = sum_a Q[a,a] x_a + sum_{a<b} Q[a,b] x_a x_b + offset,
with Q stored upper-triangular. Bit a of a basis-state index is variable a
(little-endian, the same ordering Qiskit uses for qubits).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import Instance


@dataclass
class QUBO:
    Q: np.ndarray        # (n, n) upper triangular
    offset: float
    penalty: float
    weights: np.ndarray  # units represented by each variable
    instance: Instance

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ np.triu(self.Q) @ x + self.offset)


def build_qubo(inst: Instance, penalty: float) -> QUBO:
    n = inst.n_vars
    Q = np.zeros((n, n))
    w = np.zeros(n)
    c = inst.unit_cost
    for i in range(inst.n_cat):
        idx = []
        for v in range(inst.n_sup):
            for b in range(inst.bits):
                a = inst.var_index(i, v, b)
                w[a] = inst.lot[i, v] * 2 ** b
                idx.append(a)
        D = float(inst.demand[i])
        for a in idx:
            Q[a, a] += c[i, a // inst.bits % inst.n_sup] * w[a]          # cost
            Q[a, a] += penalty * (w[a] ** 2 - 2 * D * w[a])              # penalty, x^2 = x
        for p_, a in enumerate(idx):
            for b_ in idx[p_ + 1:]:
                Q[a, b_] += 2 * penalty * w[a] * w[b_]
    offset = penalty * float((inst.demand.astype(float) ** 2).sum())
    return QUBO(Q=Q, offset=offset, penalty=penalty, weights=w, instance=inst)


# ---------------------------------------------------------------- decoding
def bits_from_index(index: int, n: int) -> np.ndarray:
    return np.array([(index >> a) & 1 for a in range(n)], dtype=int)


def decode(inst: Instance, x: np.ndarray) -> np.ndarray:
    """Binary vector -> award quantities q (n_cat, n_sup) in units."""
    x = np.asarray(x).reshape(inst.n_cat, inst.n_sup, inst.bits)
    lots = (x * (2 ** np.arange(inst.bits))).sum(axis=2)
    return lots * inst.lot


def award_cost(inst: Instance, q: np.ndarray) -> float:
    return float((inst.unit_cost * q).sum())


def violations(inst: Instance, q: np.ndarray) -> dict:
    demand_gap = q.sum(axis=1) - inst.demand
    out = {"demand_gap": demand_gap.tolist(), "demand_ok": bool(np.all(demand_gap == 0))}
    if inst.capacity is not None:
        over = np.maximum(q.sum(axis=0) - inst.capacity, 0)
        out["capacity_over"] = over.tolist()
        out["capacity_ok"] = bool(np.all(over == 0))
    else:
        out["capacity_ok"] = True
    out["feasible"] = out["demand_ok"] and out["capacity_ok"]
    return out


# ------------------------------------------------------------- brute force
def all_energies(qubo: QUBO) -> np.ndarray:
    """Energy of every basis state (vectorised). Practical up to ~24 variables."""
    n = qubo.n
    if n > 26:
        raise ValueError("Brute force is limited to 26 variables")
    states = np.arange(2 ** n, dtype=np.uint32)
    X = [((states >> a) & 1).astype(np.float64) for a in range(n)]
    E = np.full(2 ** n, qubo.offset)
    Q = qubo.Q
    for a in range(n):
        if Q[a, a]:
            E += Q[a, a] * X[a]
        for b in range(a + 1, n):
            if Q[a, b]:
                E += Q[a, b] * (X[a] * X[b])
    return E


def feasible_mask(inst: Instance, n: int) -> np.ndarray:
    """Boolean mask over basis states that satisfy demand (and capacity)."""
    states = np.arange(2 ** n, dtype=np.uint32)
    ok = np.ones(2 ** n, dtype=bool)
    usage = np.zeros((inst.n_sup, 2 ** n))
    for i in range(inst.n_cat):
        tot = np.zeros(2 ** n)
        for v in range(inst.n_sup):
            q = np.zeros(2 ** n)
            for b in range(inst.bits):
                a = inst.var_index(i, v, b)
                q += ((states >> a) & 1) * inst.lot[i, v] * 2 ** b
            tot += q
            usage[v] += q
        ok &= tot == inst.demand[i]
    if inst.capacity is not None:
        ok &= np.all(usage <= inst.capacity[:, None], axis=0)
    return ok
