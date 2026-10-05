"""Q-Allocate: hybrid quantum-classical QUBO solver for supplier split-award optimisation."""
from .instance import Instance, generate
from .qubo import QUBO, build_qubo, decode, award_cost, violations, all_energies, feasible_mask
from .ising import Ising, qubo_to_ising, ising_diagonal
from .classical import solve_milp
from .qaoa import optimise, optimise_depths, run_qaoa, sample, evaluate_samples
from .penalty import tune_penalty
from .decomposition import lagrangian_decomposition

__version__ = "0.1.0"
