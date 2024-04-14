import time
import numpy as np
import graphix
import statistics
import cProfile, pstats, io


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


def get_perf(f):
    start = time.perf_counter()
    f()
    end = time.perf_counter()
    return end - start


def random_pattern(nqubits, depth):
    circuit = simple_random_circuit(nqubits, depth)
    pattern = circuit.transpile()
    pattern.standardize()
    pattern.minimize_space()
    return pattern


class TimeSuite:
    def setup(self, ncircuits, nqubits, depth):
        self.patterns = [random_pattern(nqubits, depth) for _ in range(ncircuits)]

    def test_consistency(self):
        for pattern in self.patterns:
            numpy_result = pattern.simulate_pattern(impl=graphix.sim.statevec.Statevec)
            rust_result = pattern.simulate_pattern(impl=graphix.sim.statevec.RustStatevec)
            np.testing.assert_almost_equal(numpy_result.flatten(), rust_result.flatten(), decimal=2)

    def time_impl(self, impl):
        for pattern in self.patterns:
            pattern.simulate_pattern(impl=impl)


ts = TimeSuite()
ts.setup(20, 16, 2)
ts.test_consistency()


def benchmark(ts, impl):
    pr = cProfile.Profile()
    pr.enable()
    ts.time_impl(impl)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    print(s.getvalue())


benchmark(ts, graphix.sim.statevec.Statevec)
benchmark(ts, graphix.sim.statevec.RustStatevec)
