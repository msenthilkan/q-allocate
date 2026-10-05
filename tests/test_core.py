import numpy as np
import pytest

from qallocate import (generate, build_qubo, decode, award_cost, violations, all_energies,
                       feasible_mask, qubo_to_ising, ising_diagonal, solve_milp)
from qallocate.qaoa import AerEngine, numpy_statevector


@pytest.fixture(scope="module")
def inst():
    return generate(2, 3, bits=2, seed=3)


def test_qubo_energy_equals_cost_plus_penalty(inst):
    qubo = build_qubo(inst, penalty=7.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 2, qubo.n)
        q = decode(inst, x)
        expected = award_cost(inst, q) + 7.0 * float(((q.sum(1) - inst.demand) ** 2).sum())
        assert np.isclose(qubo.energy(x), expected)


def test_ising_matches_qubo(inst):
    qubo = build_qubo(inst, penalty=5.0)
    ising = qubo_to_ising(qubo)
    E = all_energies(qubo)
    assert np.allclose(ising_diagonal(ising) + ising.offset, E)


def test_ground_state_equals_milp(inst):
    qubo = build_qubo(inst, penalty=float(inst.price.max()))
    E = all_energies(qubo)
    gs = int(np.argmin(E))
    assert feasible_mask(inst, qubo.n)[gs], "ground state must satisfy demand"
    milp = solve_milp(inst)
    x = [(gs >> a) & 1 for a in range(qubo.n)]
    assert np.isclose(award_cost(inst, decode(inst, x)), milp["cost"])


def test_moq_respected_by_construction(inst):
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = decode(inst, rng.integers(0, 2, inst.n_vars))
        nz = q > 0
        assert np.all(q[nz] >= inst.lot[nz])
        assert np.all(q % inst.lot == 0)


def test_numpy_engine_matches_aer():
    small = generate(1, 3, bits=2, seed=5)
    ising = qubo_to_ising(build_qubo(small, 50.0))
    diag = ising_diagonal(ising) / ising.scale()
    g, b = [0.3, 0.1], [0.7, 0.4]
    psi_np = numpy_statevector(diag, g, b)
    psi_aer = AerEngine(ising, 2).statevector(g, b)
    assert np.allclose(np.abs(psi_np) ** 2, np.abs(psi_aer) ** 2, atol=1e-8)


def test_braket_circuit_matches_numpy():
    pytest.importorskip("braket")
    import importlib.util, pathlib
    from braket.devices import LocalSimulator
    spec = importlib.util.spec_from_file_location("b", pathlib.Path(__file__).parents[1] / "scripts" / "05_braket_sv1.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    small = generate(1, 3, bits=2, seed=2)
    ising = qubo_to_ising(build_qubo(small, 40.0))
    circ = mod.braket_circuit(ising, [0.4], [0.6])
    circ.state_vector()
    sv = np.asarray(LocalSimulator().run(circ, shots=0).result().values[0])
    n = ising.n
    # Braket orders amplitudes with qubit 0 as the most significant bit
    probs_braket = np.abs(sv) ** 2
    idx = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(2 ** n)])
    reordered = np.empty_like(probs_braket); reordered[idx] = probs_braket
    probs_np = np.abs(numpy_statevector(ising_diagonal(ising) / ising.scale(), [0.4], [0.6])) ** 2
    assert np.allclose(reordered, probs_np, atol=1e-8)
