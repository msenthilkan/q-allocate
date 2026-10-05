"""Exact classical baseline: the same discrete split-award model as a MILP (PuLP + CBC)."""
from __future__ import annotations

import time

import numpy as np
import pulp

from .instance import Instance


def solve_milp(inst: Instance, time_limit: int = 120) -> dict:
    """Integer lots n_iv in [0, 2^bits - 1]; q_iv = lot_iv * n_iv.
    Demand equality per category; optional supplier capacity across categories."""
    t0 = time.perf_counter()
    m = pulp.LpProblem("split_award", pulp.LpMinimize)
    n = {(i, v): pulp.LpVariable(f"n_{i}_{v}", 0, inst.max_lots, cat="Integer")
         for i in range(inst.n_cat) for v in range(inst.n_sup)}
    c = inst.unit_cost
    m += pulp.lpSum(float(c[i, v] * inst.lot[i, v]) * n[i, v] for (i, v) in n)
    for i in range(inst.n_cat):
        m += pulp.lpSum(int(inst.lot[i, v]) * n[i, v] for v in range(inst.n_sup)) == int(inst.demand[i])
    if inst.capacity is not None:
        for v in range(inst.n_sup):
            m += pulp.lpSum(int(inst.lot[i, v]) * n[i, v] for i in range(inst.n_cat)) <= int(inst.capacity[v])
    status = m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    elapsed = time.perf_counter() - t0
    q = np.array([[int(round(n[i, v].value() or 0)) * inst.lot[i, v] for v in range(inst.n_sup)]
                  for i in range(inst.n_cat)])
    return {
        "status": pulp.LpStatus[status],
        "cost": float(pulp.value(m.objective)),
        "q": q,
        "seconds": elapsed,
    }
