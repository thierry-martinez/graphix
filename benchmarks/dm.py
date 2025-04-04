import numpy as np
import graphix
import cProfile, pstats, io
import sys
import argparse

import typer
from graphix.sim.density_matrix import DensityMatrix, RustDensityMatrix
from graphix.noise_models.depolarising_noise_model import DepolarisingNoiseModel


def simple_random_circuit(nqubit, depth):
    r"""Generate a test circuit for benchmarking.

    This function generates a circuit with nqubit qubits and depth layers,
    having layers of CNOT and Rz gates with random placements.

    Parameters
    ----------
    nqubit : int
        number of qubits
    depth : int
        number of layers

    Returns
    -------
    circuit : graphix.transpiler.Circuit object
        generated circuit
    """
    qubit_index = [i for i in range(nqubit)]
    circuit = graphix.Circuit(nqubit)
    for _ in range(depth):
        np.random.shuffle(qubit_index)
        for j in range(len(qubit_index) // 2):
            circuit.cnot(qubit_index[2 * j], qubit_index[2 * j + 1])
        for j in range(len(qubit_index)):
            circuit.rz(qubit_index[j], 2 * np.pi * np.random.random())
    return circuit


def random_pattern(nqubits, depth):
    circuit = simple_random_circuit(nqubits, depth)
    pattern = circuit.transpile().pattern
    pattern.minimize_space()
    return pattern


class TimeSuite:
    def setup(self, ncircuits, nqubits, depth, noise_model):
        self.patterns = [random_pattern(nqubits, depth) for _ in range(ncircuits)]
        self.noise_model = noise_model

    def test_consistency(self):
        for pattern in self.patterns:
            numpy_result = pattern.simulate_pattern(backend="densitymatrix", impl=DensityMatrix, noise_model=self.noise_model).rho.flatten()
            rust_result = pattern.simulate_pattern(backend="densitymatrix", impl=RustDensityMatrix, noise_model=self.noise_model).flatten()
            np.testing.assert_equal(len(numpy_result), len(rust_result))
            np.testing.assert_almost_equal(numpy_result, rust_result, decimal=2)

    def time_impl(self, impl):
        print(f"Benchmarkin {impl}")
        for pattern in self.patterns:
            pattern.simulate_pattern(backend="densitymatrix", impl=impl, noise_model=self.noise_model)
        print()

def benchmark(ts, impl, identifier):
    """ 
    Returns the profiling stats as a string.
    """
    pr = cProfile.Profile()

    pr.enable()
    ts.time_impl(impl)
    pr.disable()

    # Capture profiling stats in a string
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(identifier)
    profiling_output = s.getvalue()
    
    return profiling_output

def main(nqubits: int = 2, ncircuits: int = 10, depth: int = 2, check_consistency: bool = False) -> None:

    print(f"========== Running benchmark with parameters with: max-nqubits={nqubits}, ncircuits={ncircuits}, depth={depth}, check-consistency={check_consistency} ==========")
    
    ts = TimeSuite()
    
    noise_model = DepolarisingNoiseModel(entanglement_error_prob=0.5)
    ts.setup(ncircuits=ncircuits, nqubits=nqubits, depth=depth, noise_model=noise_model)

    if check_consistency:
        ts.test_consistency()

    np = benchmark(ts, impl=DensityMatrix, identifier="density_matrix")
    print(np)
    rs = benchmark(ts, impl=RustDensityMatrix, identifier="dm_simu_rs|density_matrix")
    print(rs)

if __name__ == "__main__":
    typer.run(main)

    #parser = argparse.ArgumentParser(description="Graphix rust density matrix back-end benchmark")
    #
    #parser.add_argument("--nqubits", type=int, help="Maximum number of qubits for the benchmark", default=2, required=False)
    #parser.add_argument("--ncircuits", type=int, help="Number of simulations to perform to get a mean of the runtime", default=10, required=False)
    #parser.add_argument("--depth", type=int, help="Depth of the circuits that will be transpiled to MBQC patterns", default=2, required=False)
    #parser.add_argument("--check-consistency", help="Whether or not check for consistent results between the two implementations. This will take more time to run the benchmark.", action="store_true", required=False)
    #
    #args = parser.parse_args()
    #
    #nqubits = args.nqubits
    #ncircuits = args.ncircuits
    #depth = args.depth
    #check_consistency = args.check_consistency
