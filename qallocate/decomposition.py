"""Classical-quantum decomposition (guideline success criterion 5).

The full problem couples categories only through supplier capacity
(sum_i q_iv <= Cap_v). Relaxing that constraint with Lagrange multipliers
lambda_v >= 0 splits the problem into independent category blocks:

    min  sum_iv (c_iv + lambda_v) q_iv   s.t. demand per category   - one sub-QUBO per block
    minus  sum_v lambda_v Cap_v

A classical coordinator updates lambda by projected subgradient steps; each
block is a 15-20 variable sub-QUBO solved by QAOA (or exactly, for reference).

Pure Lagrangian iterates oscillate around the capacity limits, so each iteration
also runs a primal recovery pass: blocks are fixed one at a time and each later
block picks its best candidate (QAOA sample, or exact state) that fits the
residual capacity left by earlier blocks. Both block orders are tried and the
best feasible stitched award is kept. With exact sub-solvers the loop also
yields a Lagrangian dual lower bound.
"""
from __future__ import annotations

import time

import numpy as np

from .instance import Instance
from .penalty import tune_penalty
from .qaoa import sample
from .qubo import award_cost, build_qubo, decode, feasible_mask, all_energies, bits_from_index


def _candidates_exact(sub: Instance) -> list[tuple[float, np.ndarray]]:
    """All demand-feasible awards of a block, cheapest (priced) first."""
    qubo = build_qubo(sub, penalty=float(sub.unit_cost.max()))
    E = all_energies(qubo)
    idx = np.flatnonzero(feasible_mask(sub, qubo.n))
    idx = idx[np.argsort(E[idx])]
    return [(float(E[i]), decode(sub, bits_from_index(int(i), qubo.n))) for i in idx]


def _candidates_qaoa(sub: Instance, p: int, shots: int, seed: int, cvar_alpha: float) -> list[tuple[float, np.ndarray]]:
    """Demand-feasible awards found among the QAOA samples, cheapest (priced) first."""
    out = tune_penalty(sub, p=p, shots=shots, seed=seed, cvar_alpha=cvar_alpha)
    counts = sample(out["result"].probs, shots, seed=seed)
    seen, cands = set(), []
    for idx in counts:
        q = decode(sub, bits_from_index(idx, sub.n_vars))
        if np.all(q.sum(axis=1) == sub.demand):
            key = q.tobytes()
            if key not in seen:
                seen.add(key)
                cands.append((award_cost(sub, q), q))
    return sorted(cands, key=lambda c: c[0])


def _recover(inst: Instance, blocks, cands, order) -> np.ndarray | None:
    """Fix blocks in `order`; each takes its cheapest candidate that fits residual capacity."""
    q = np.zeros((inst.n_cat, inst.n_sup), dtype=int)
    residual = inst.capacity.astype(int).copy()
    for k in order:
        pick = next((c for _, c in cands[k] if np.all(c.sum(axis=0) <= residual)), None)
        if pick is None:
            return None
        q[blocks[k]] = pick
        residual -= pick.sum(axis=0)
    return q


def lagrangian_decomposition(inst: Instance, block_size: int = 2, iterations: int = 15,
                             solver: str = "qaoa", p: int = 1, shots: int = 2048,
                             step_scale: float = 0.15, cvar_alpha: float = 0.1, seed: int = 0) -> dict:
    if inst.capacity is None:
        raise ValueError("Decomposition needs supplier capacities")
    t0 = time.perf_counter()
    blocks = [list(range(k, min(k + block_size, inst.n_cat))) for k in range(0, inst.n_cat, block_size)]
    lam = np.zeros(inst.n_sup)
    step0 = step_scale * float(inst.unit_cost.mean())
    best, log = None, []

    for t in range(iterations):
        q = np.zeros((inst.n_cat, inst.n_sup), dtype=int)
        dual = -float(lam @ inst.capacity)
        cands, ok_blocks = [], True
        for k, blk in enumerate(blocks):
            sub = inst.subset(blk, cost_adjust=lam)
            c = (_candidates_exact(sub) if solver == "exact"
                 else _candidates_qaoa(sub, p, shots, seed + 100 * t + k, cvar_alpha))
            cands.append(c)
            if not c:
                ok_blocks = False
                continue
            q[blk] = c[0][1]                       # relaxed (Lagrangian) choice
            dual += c[0][0]
        usage = q.sum(axis=0)
        g = usage - inst.capacity

        # primal recovery: both block orders, keep the cheapest capacity-feasible award
        recovered = None
        if ok_blocks:
            for order in (range(len(blocks)), reversed(range(len(blocks)))):
                r = _recover(inst, blocks, cands, list(order))
                if r is not None and (recovered is None or award_cost(inst, r) < award_cost(inst, recovered)):
                    recovered = r
        cost = None if recovered is None else award_cost(inst, recovered)
        if recovered is not None and (best is None or cost < best["cost"]):
            best = {"cost": cost, "q": recovered.copy(), "iteration": t + 1}
        log.append({"iteration": t + 1, "lambda": lam.round(3).tolist(),
                    "max_overload_relaxed": int(max(g.max(), 0)),
                    "recovered_cost": cost, "best_cost": None if best is None else best["cost"],
                    "dual_bound": dual if (solver == "exact" and ok_blocks) else None})
        step = step0 / (t + 1)
        lam = np.maximum(0.0, lam + step * g / max(1.0, float(np.abs(g).max())))

    return {"best": best, "log": log, "blocks": blocks,
            "sub_qubo_sizes": [len(b) * inst.n_sup * inst.bits for b in blocks],
            "seconds": time.perf_counter() - t0, "solver": solver}
