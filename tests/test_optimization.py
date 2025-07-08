from __future__ import annotations

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.gflow import gflow_from_pattern
from graphix.optimization import incorporate_pauli_results
from graphix.random_objects import rand_circuit


@pytest.mark.parametrize("jumps", range(1, 11))
def test_incorporate_pauli_results(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 3
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.perform_pauli_measurements()
    pattern2 = incorporate_pauli_results(pattern)
    state = pattern.simulate_pattern(rng=rng)
    state2 = pattern2.simulate_pattern(rng=rng)
    assert np.abs(np.dot(state.flatten().conjugate(), state2.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_flow_after_pauli_preprocessing(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 3
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.perform_pauli_measurements()
    pattern2 = incorporate_pauli_results(pattern)
    pattern2.standardize()
    f, _l = gflow_from_pattern(pattern2)
    assert f is not None
