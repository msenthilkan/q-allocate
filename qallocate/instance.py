"""Split-award problem instances (synthetic, reproducible).

A supplier v awarded in category i receives q_iv units, where

    q_iv = lot_iv * n_iv,   n_iv in {0, 1, ..., 2**bits - 1}

so every non-zero award is a whole number of lots. Setting the lot size to the
supplier's minimum order quantity (MOQ) enforces the MOQ by construction, and
`bits` caps the largest award (a split-award rule: no supplier can take more
than lot_iv * (2**bits - 1) units).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Instance:
    categories: list[str]
    suppliers: list[str]
    price: np.ndarray       # P_iv  base unit cost
    tariff: np.ndarray      # T_iv  tariff / duty per unit
    leadtime: np.ndarray    # L_iv  lead-time penalty per unit
    risk: np.ndarray        # R_iv  supplier risk score in [0, 1]
    alpha: float            # risk-tolerance weight
    lot: np.ndarray         # lot size (= MOQ) per category-supplier, units
    demand: np.ndarray      # D_i  units
    bits: int = 2           # binary-encoding bits per category-supplier pair
    capacity: np.ndarray | None = None  # optional supplier capacity across categories
    name: str = "instance"
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ sizes
    @property
    def n_cat(self) -> int:
        return len(self.categories)

    @property
    def n_sup(self) -> int:
        return len(self.suppliers)

    @property
    def n_vars(self) -> int:
        return self.n_cat * self.n_sup * self.bits

    @property
    def unit_cost(self) -> np.ndarray:
        """Risk-adjusted landed cost per unit: P + T + L + alpha * R."""
        return self.price + self.tariff + self.leadtime + self.alpha * self.risk

    @property
    def max_lots(self) -> int:
        return 2 ** self.bits - 1

    def var_index(self, i: int, v: int, b: int) -> int:
        return (i * self.n_sup + v) * self.bits + b

    def subset(self, cat_idx: list[int], cost_adjust: np.ndarray | None = None) -> "Instance":
        """Sub-problem with a subset of categories and no capacity constraint.
        `cost_adjust` (per supplier) is added to the tariff term, which is how the
        Lagrangian coordinator prices supplier capacity."""
        adj = np.zeros(self.n_sup) if cost_adjust is None else np.asarray(cost_adjust, float)
        return Instance(
            categories=[self.categories[i] for i in cat_idx],
            suppliers=list(self.suppliers),
            price=self.price[cat_idx].copy(),
            tariff=self.tariff[cat_idx] + adj[None, :],
            leadtime=self.leadtime[cat_idx].copy(),
            risk=self.risk[cat_idx].copy(),
            alpha=self.alpha,
            lot=self.lot[cat_idx].copy(),
            demand=self.demand[cat_idx].copy(),
            bits=self.bits,
            capacity=None,
            name=f"{self.name}[{','.join(map(str, cat_idx))}]",
        )

    # --------------------------------------------------------------- json io
    def to_dict(self) -> dict:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Instance":
        d = json.loads(Path(path).read_text())
        for k in ("price", "tariff", "leadtime", "risk", "lot", "demand"):
            d[k] = np.array(d[k], dtype=float if k != "lot" else int)
        d["demand"] = d["demand"].astype(int)
        if d.get("capacity") is not None:
            d["capacity"] = np.array(d["capacity"], dtype=int)
        return cls(**d)


def generate(n_categories: int, n_suppliers: int, bits: int = 2, alpha: float = 20.0,
             capacity: bool = False, seed: int = 0, name: str | None = None) -> Instance:
    """Random but always-feasible instance.

    Demand is built from a random feasible award (at least two suppliers per
    category, i.e. a genuine split award), so a solution is guaranteed to exist.
    """
    rng = np.random.default_rng(seed)
    nc, ns = n_categories, n_suppliers
    base = rng.uniform(80, 120, size=nc)                       # category price level
    sup_factor = rng.uniform(0.85, 1.15, size=ns)              # supplier competitiveness
    price = np.round(base[:, None] * sup_factor[None, :] * rng.uniform(0.95, 1.05, (nc, ns)), 2)
    tariff_rate = rng.choice([0.0, 0.05, 0.10, 0.15], size=ns)  # by supplier country
    tariff = np.round(price * tariff_rate[None, :], 2)
    leadtime = np.round(rng.uniform(0, 8, (nc, ns)), 2)
    risk = np.round(np.repeat(rng.uniform(0, 1, ns)[None, :], nc, axis=0), 3)
    lot = rng.choice([10, 20, 25, 50], size=(nc, ns))

    max_lots = 2 ** bits - 1
    award = np.zeros((nc, ns), dtype=int)
    for i in range(nc):
        k = rng.integers(2, min(3, ns) + 1)                    # 2-3 suppliers per category
        chosen = rng.choice(ns, size=k, replace=False)
        award[i, chosen] = rng.integers(1, max_lots + 1, size=k)
    demand = (award * lot).sum(axis=1)

    cap = None
    if capacity:
        usage = (award * lot).sum(axis=0)
        cap = np.maximum(usage, np.round(0.30 * demand.sum()).astype(int))

    return Instance(
        categories=[f"C{i + 1}" for i in range(nc)],
        suppliers=[f"S{v + 1}" for v in range(ns)],
        price=price, tariff=tariff, leadtime=leadtime, risk=risk, alpha=alpha,
        lot=lot, demand=demand, bits=bits, capacity=cap,
        name=name or f"syn_c{nc}_s{ns}_b{bits}_seed{seed}",
        meta={"seed": seed, "synthetic": True},
    )
