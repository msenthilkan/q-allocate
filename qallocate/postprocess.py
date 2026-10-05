"""Classical post-processing of quantum samples (reported separately from raw QAOA).

Steepest-descent local search on the QUBO energy using single- and two-bit flips,
started from the best distinct sampled bitstrings. Two-bit flips let the search
move units between suppliers while keeping demand satisfied.
"""
from __future__ import annotations

import numpy as np

from .qubo import QUBO, award_cost, bits_from_index, decode, violations


def _local_search(qubo: QUBO, x: np.ndarray, max_steps: int = 200) -> np.ndarray:
    Q = np.triu(qubo.Q)
    S = Q + Q.T - np.diag(np.diag(Q))          # symmetric couplings, diagonal = linear
    n = len(x)
    x = x.astype(float).copy()
    for _ in range(max_steps):
        field = S @ x - np.diag(S) * x            # sum_{b != a} S_ab x_b
        d1 = (1 - 2 * x) * (np.diag(S) + field)   # energy change of flipping a
        best_d, move = d1.min(), ("1", int(d1.argmin()))
        sgn = 1 - 2 * x
        pair = d1[:, None] + d1[None, :] + np.outer(sgn, sgn) * (S - np.diag(np.diag(S)))
        np.fill_diagonal(pair, np.inf)
        if pair.min() < best_d:
            a, b = np.unravel_index(int(pair.argmin()), pair.shape)
            best_d, move = pair[a, b], ("2", (int(a), int(b)))
        if best_d >= -1e-9:
            break
        if move[0] == "1":
            x[move[1]] = 1 - x[move[1]]
        else:
            a, b = move[1]
            x[a], x[b] = 1 - x[a], 1 - x[b]
    return x.astype(int)


def refine_samples(qubo: QUBO, counts: dict[int, int], top_k: int = 20) -> dict | None:
    inst = qubo.instance
    starts = sorted(counts, key=lambda i: qubo.energy(bits_from_index(i, qubo.n)))[:top_k]
    best = None
    for idx in starts:
        x = _local_search(qubo, bits_from_index(idx, qubo.n))
        q = decode(inst, x)
        if violations(inst, q)["feasible"]:
            c = award_cost(inst, q)
            if best is None or c < best["cost"]:
                best = {"cost": c, "q": q}
    return best
