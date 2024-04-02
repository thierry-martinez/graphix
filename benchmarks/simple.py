import time
import numpy as np
import graphix
import statistics


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


class TimeSuite:
    def setup(self):
        circuit = simple_random_circuit(18, 1)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.minimize_space()
        self.pattern = pattern

    def time_pattern(self):
        self.pattern.simulate_pattern()


ts = TimeSuite()
ts.setup()
timings = []
for _ in range(10):
    timings.append(get_perf(lambda: ts.time_pattern()))
print(statistics.mean(timings))
