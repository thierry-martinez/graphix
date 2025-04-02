import numpy as np
import graphix
import cProfile, pstats, io
import sys

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
        print(f"Running {self.patterns[0].n_node} nodes patterns for {impl}")
        for pattern in self.patterns:
            print(".", end="")
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

def parse_tot_time(profiling_output):
    """
    Parse the total elapsed time from the profiling output string.
    """
    # Extract the total time from the profiling stats
    total_time_line = next(
        (line for line in profiling_output.splitlines() if "function calls" in line), None
    )
    total_time = None
    if total_time_line:
        total_time = float(total_time_line.split()[-2])  # Get the time in seconds

    return total_time

def benchmark_by_simulation_size(ncircuits=20, max_nqubits=10, circuit_depth=2, noise_model=None):
    times = []  # Elapsed simulation times according to the number of qubits of the circuits.
    ts = TimeSuite()
    for nqubit in range(1, max_nqubits + 1):   # For each number of qubits, create n Circuits and benchmark the two backends to compare their performance.
        ts.setup(ncircuits, nqubit, circuit_depth, noise_model=noise_model)

        ts.test_consistency()   # Ensure we get the same results.

        graphix_bench_output = benchmark(ts, impl=DensityMatrix, identifier="density_matrix")
        rs_bench_output = benchmark(ts, impl=RustDensityMatrix, identifier="dm_simu_rs|density_matrix")

        print(f"Benchmark with {nqubit} qubits graphix:\n{graphix_bench_output}")
        print(f"Benchmark with {nqubit} qubits rs:\n{rs_bench_output}")
        print("========================================================")

        graphix_tot_time = parse_tot_time(graphix_bench_output) / ncircuits
        rs_tot_time = parse_tot_time(rs_bench_output) / ncircuits

        # Append total times as a tuple of (graphix_tot_time, rs_total_time)
        times.append((graphix_tot_time, rs_tot_time))

    return times

if __name__ == "__main__":
    # TODO: add arguments --nqubits, --npatterns --depth
    nqubits = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # By default, if no integer argument is passed, set the number of qubits to 2
    ts = TimeSuite()
    
    noise_model = DepolarisingNoiseModel(entanglement_error_prob=0.5)

    ts.setup(10, nqubits, 2, noise_model=noise_model)
    # ts.test_consistency() # We could keep test_consistency here, but it would make it really long to run for big simulations

    np = benchmark(ts, impl=DensityMatrix, identifier="density_matrix")
    print(np)
    rs = benchmark(ts, impl=RustDensityMatrix, identifier="dm_simu_rs|density_matrix")
    print(rs)
