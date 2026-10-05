# Formulation

## Notation

| Symbol | Meaning |
| --- | --- |
| $i$, $v$ | category, supplier |
| $P_{iv}, T_{iv}, L_{iv}, R_{iv}$ | base price, tariff/duty, lead-time penalty (per unit), risk score in [0, 1] |
| $\alpha$ | risk-tolerance weight |
| $c_{iv} = P_{iv} + T_{iv} + L_{iv} + \alpha R_{iv}$ | risk-adjusted landed unit cost |
| $\text{lot}_{iv}$ | lot size, equal to the supplier MOQ |
| $D_i$ | demand of category *i* (units) |
| $\text{Cap}_v$ | supplier capacity across categories (32-variable instance only) |
| $B$ | bits per category-supplier pair |
| $x_{ivb}\in\{0,1\}$ | binary decision variables (one qubit each) |

## 1. Encoding

$$q_{iv} = \text{lot}_{iv}\, n_{iv}, \qquad n_{iv} = \sum_{b=0}^{B-1} 2^b x_{ivb} \in \{0,\dots,2^B-1\}$$

* **Qubits:** $N = |\text{categories}| \times |\text{suppliers}| \times B$. One-hot encoding of $2^B - 1$ quantity bands would need $2^B - 1$ qubits per pair instead of $B$.
* **MOQ:** any $n_{iv} \ge 1$ gives $q_{iv} \ge \text{lot}_{iv} = \text{MOQ}_{iv}$, so MOQ holds for every bitstring.
* **Split cap:** $q_{iv} \le (2^B - 1)\,\text{lot}_{iv}$.

| Instance | Categories × suppliers × bits | Qubits |
| --- | --- | --- |
| small | 1×4×2, 2×3×2, 2×4×2, 2×5×2 | 8, 12, 16, 20 |
| scale | 4×4×2 | 32 (two 16-qubit blocks) |

## 2. QUBO

$$E(x) = \sum_{i,v} c_{iv} q_{iv} + P \sum_i \Big(\sum_{v} q_{iv} - D_i\Big)^2$$

Let $w_a = \text{lot}_{iv} 2^b$ be the units carried by variable $a = (i, v, b)$. Expanding the square within category *i*, and using $x_a^2 = x_a$:

$$Q_{aa} = c_{iv} w_a + P\,(w_a^2 - 2 D_i w_a), \qquad Q_{ab} = 2 P\, w_a w_b \;\;(a<b,\ \text{same category}), \qquad \text{offset} = P\sum_i D_i^2$$

**Why $P = \max P_{iv}$ is a safe start.** Every award is a multiple of a lot (at least 10 units), so any demand gap $\delta$ satisfies $|\delta| \ge 10$. Under-buying by $\delta$ saves at most $c_{\max}\delta$ but costs $P\delta^2$. With $P$ of the order of the unit cost, the penalty dominates, and the QUBO ground state is the cheapest feasible award. `tests/test_core.py` checks this against the MILP.

**Dynamic tuning.** The QAOA landscape gets flatter as $P$ grows, so $P$ is only increased (×2) when the lowest-energy sampled bitstring violates demand.

## 3. Ising Hamiltonian

With $x_a = (1 - Z_a)/2$:

$$h_a = -\frac{Q_{aa}}{2} - \frac14\sum_{b\ne a} Q_{ab}, \qquad J_{ab} = \frac{Q_{ab}}{4}, \qquad \text{offset}_{Z} = \text{offset} + \frac12\sum_a Q_{aa} + \frac14\sum_{a<b} Q_{ab}$$

$$H_C = \sum_{a<b} J_{ab} Z_a Z_b + \sum_a h_a Z_a$$

Couplings exist only between variables of the same category, so each category forms a dense block.

## 4. QAOA

$$|\gamma,\beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l \sum_a X_a}\, e^{-i\gamma_l \tilde H_C}\, |+\rangle^{\otimes N}, \qquad \tilde H_C = H_C / \max(|h|, |J|)$$

* **Gates:** $R_Z(2\gamma h_a)$, $R_{ZZ}(2\gamma J_{ab})$ and $R_X(2\beta)$. At p = 1 a 12-qubit instance has depth 12 with 30 RZZ gates.
* **Objective:** $\text{CVaR}_{0.1}$, the mean energy of the best 10% of probability mass. It rewards a good tail of samples.
* **Optimiser:** grid start for p = 1, then COBYLA. Depth p+1 is warm-started from p using INTERP interpolation.
* **Engines:** Qiskit AerSimulator (circuit) and an exact numpy statevector kernel for sweeps. Tests confirm that both, and the Braket circuit, give identical probabilities.

## 5. Decomposition for 32 variables

Supplier capacity $\sum_i q_{iv} \le \text{Cap}_v$ couples categories. Relaxing it with multipliers $\lambda_v \ge 0$ gives

$$L(\lambda) = \sum_{\text{blocks } k} \min_{q^{(k)}} \sum_{i\in k,v} (c_{iv} + \lambda_v) q_{iv} \;-\; \sum_v \lambda_v \text{Cap}_v$$

* Each block (2 categories, 16 qubits) is an independent QUBO with priced costs, which fits the guideline's 15–20 variable target.
* **Coordinator:** projected subgradient, $\lambda \leftarrow \max(0, \lambda + s_t\, g/\|g\|_\infty)$ with $g = \text{usage} - \text{Cap}$ and $s_t = 0.15\,\bar c/(t+1)$.
* **Primal recovery:** the relaxed stitched award usually overloads some supplier. Blocks are therefore fixed in sequence, and each later block takes its cheapest candidate (QAOA sample or exact state) that fits the residual capacity. Both block orders are tried.
* **Lower bound:** with exact block solvers $L(\lambda)$ is a valid lower bound. On the 32-variable instance it reaches within 0.9% of the optimum.
