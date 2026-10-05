"""Dynamic penalty tuning (guideline 2.2).

Start with P_penalty = max base price P_iv and scale it up whenever the quantum
solver returns a constraint-violating bitstring (the lowest-energy sample is
infeasible). Too small a penalty lets the optimiser under-buy; too large a
penalty flattens the cost signal, so we stop at the first value that works.
"""
from __future__ import annotations

from .instance import Instance
from .qaoa import run_qaoa
from .qubo import build_qubo


def tune_penalty(inst: Instance, p: int = 1, shots: int = 1024, growth: float = 2.0,
                 max_rounds: int = 6, engine: str = "numpy",
                 seed: int = 0, **kw) -> dict:
    penalty = float(inst.price.max())
    history = []
    for r in range(max_rounds):
        qubo = build_qubo(inst, penalty)
        res, ev = run_qaoa(qubo, p=p, shots=shots, engine=engine, seed=seed + r, **kw)
        ok = ev["returned_feasible"]
        history.append({"round": r + 1, "penalty": penalty,
                        "feasible_fraction": ev["feasible_fraction"],
                        "best_cost": None if ev["best_feasible"] is None else ev["best_feasible"]["cost"],
                        "accepted": ok})
        if ok:
            return {"penalty": penalty, "qubo": qubo, "result": res, "eval": ev, "history": history}
        penalty *= growth
    return {"penalty": penalty, "qubo": qubo, "result": res, "eval": ev, "history": history}
