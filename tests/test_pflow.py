from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest

from graphix.fundamentals import Plane
from graphix.generator import _pflow2pattern
from graphix.linalg import MatGF2
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pflow import (
    OpenGraphIndex,
    _back_substitute,
    _find_pflow_simple,
    _get_pflow_matrices,
    _get_reduced_adj,
    find_pflow,
)
from graphix.states import PlanarState

if TYPE_CHECKING:
    from numpy.random import Generator


class OpenGraphTestCase(NamedTuple):
    ogi: OpenGraphIndex
    radj: MatGF2 | None
    flow_demand_mat: MatGF2 | None
    order_demand_mat: MatGF2 | None
    has_pflow: bool


class BackSubsTestCase(NamedTuple):
    mat: MatGF2
    b: MatGF2


def prepare_test_og() -> list[OpenGraphTestCase]:
    test_cases: list[OpenGraphTestCase] = []

    # Trivial open graph with pflow and nI = nO
    def get_og_1() -> OpenGraph:
        """Return an open graph with Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-1-(2)
        """
        graph: nx.Graph[int] = nx.Graph([(0, 1), (1, 2)])
        inputs = [0]
        outputs = [2]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.5, Plane.YZ),  # Y
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_1()),
            radj=MatGF2([[1, 0], [0, 1]]),
            flow_demand_mat=MatGF2([[1, 0], [1, 1]]),
            order_demand_mat=MatGF2([[0, 0], [0, 0]]),
            has_pflow=True,
        )
    )

    # Non-trivial open graph without pflow and nI = nO
    def get_og_2() -> OpenGraph:
        """Return an open graph without Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-2-4-(6)
            | |
        [1]-3-5-(7)
        """
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)])
        inputs = [1, 0]
        outputs = [6, 7]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.XZ),  # X
            3: Measurement(0.5, Plane.YZ),  # Y
            4: Measurement(0.5, Plane.YZ),  # Y
            5: Measurement(0.1, Plane.YZ),  # YZ
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_2()),
            radj=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1],
                ]
            ),
            flow_demand_mat=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [1, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
            order_demand_mat=MatGF2(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1],
                ]
            ),
            has_pflow=False,
        )
    )

    # Non-trivial open graph with pflow and nI = nO
    def get_og_3() -> OpenGraph:
        """Return an open graph with Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-2-4-(6)
            | |
        [1]-3-5-(7)
        """
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)])
        inputs = [0, 1]
        outputs = [6, 7]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XY),  # XY
            2: Measurement(0.0, Plane.XY),  # X
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0.0, Plane.XY),  # X
            5: Measurement(0.5, Plane.XY),  # Y
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_3()),
            radj=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1],
                ]
            ),
            flow_demand_mat=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1],
                ]
            ),
            order_demand_mat=MatGF2(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            has_pflow=True,
        )
    )

    # Non-trivial open graph with pflow and nI != nO
    def get_og_4() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs.

        Example from Fig. 1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        graph: nx.Graph[int] = nx.Graph(
            [(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)]
        )
        inputs = [0]
        outputs = [5, 6]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.YZ),  # Y
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0, Plane.XZ),  # Z
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_4()),
            radj=MatGF2(
                [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1]]
            ),
            flow_demand_mat=MatGF2(
                [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0]]
            ),
            order_demand_mat=MatGF2(
                [[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            ),
            has_pflow=True,
        )
    )

    # The following tests check the final result only, not the intermediate steps.

    # Non-trivial open graph with pflow and nI != nO
    def get_og_5() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)])
        inputs = [0, 1]
        outputs = [5, 6, 8]
        meas = {
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.0, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.5, Plane.XY),
            7: Measurement(0, Plane.XY),
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_5()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    # Non-trivial open graph with pflow and nI != nO
    def get_og_6() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 2), (2, 3), (3, 4)])
        inputs = [0, 1]
        outputs = [1, 3, 4]
        meas = {0: Measurement(0.1, Plane.XY), 2: Measurement(0.5, Plane.YZ)}

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_6()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    # Non-trivial open graph with pflow and nI != nO
    def get_og_7() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7)])
        inputs = [1]
        outputs = [6, 2, 7]
        meas = {
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.YZ),
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_7()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    # Disconnected open graph with pflow and nI != nO
    def get_og_8() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (4, 6)])
        inputs: list[int] = []
        outputs = [1, 3, 4]
        meas = {0: Measurement(0.5, Plane.XZ), 2: Measurement(0, Plane.YZ), 6: Measurement(0.2, Plane.XY)}

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_8()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    # Non-trivial open graph without pflow and nI != nO
    def get_og_9() -> OpenGraph:
        """Return an open graph without Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph(
            [(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7)]
        )
        inputs = [1]
        outputs = [6, 2, 7]
        meas = {
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.XY),
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_9()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=False,
        )
    )

    # Disconnected open graph without pflow and nI != nO
    def get_og_10() -> OpenGraph:
        """Return an open graph without Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (4, 6)])
        inputs = [0]
        outputs = [1, 3, 4]
        meas = {0: Measurement(0.1, Plane.XZ), 2: Measurement(0, Plane.YZ), 6: Measurement(0.2, Plane.XY)}

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_10()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=False,
        )
    )

    return test_cases


def prepare_test_back_subs() -> list[BackSubsTestCase]:
    test_cases: list[BackSubsTestCase] = []

    # `mat` must be in row echelon form.
    # `b` must have zeros in the indices corresponding to the zero rows of `mat`.

    test_cases.extend(
        (
            BackSubsTestCase(mat=MatGF2([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]]), b=MatGF2([1, 0, 0])),
            BackSubsTestCase(
                mat=MatGF2([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
                b=MatGF2([0, 1, 1, 0, 0]),
            ),
            BackSubsTestCase(
                mat=MatGF2([[1, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
                b=MatGF2([0, 0, 0, 0]),
            ),
            BackSubsTestCase(
                mat=MatGF2([[1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]]),
                b=MatGF2([1, 0, 1, 0]),
            ),
            BackSubsTestCase(
                mat=MatGF2([[1, 0, 1], [0, 1, 0], [0, 0, 1]]),
                b=MatGF2([1, 1, 1]),
            ),
        )
    )

    return test_cases


class TestPflow:
    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_get_reduced_adj(self, test_case: OpenGraphTestCase) -> None:
        if test_case.radj is not None:
            ogi = test_case.ogi
            radj = _get_reduced_adj(ogi)
            assert radj == test_case.radj

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_get_pflow_matrices(self, test_case: OpenGraphTestCase) -> None:
        if test_case.flow_demand_mat is not None and test_case.order_demand_mat is not None:
            ogi = test_case.ogi
            flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(ogi)

            assert flow_demand_matrix == test_case.flow_demand_mat
            assert order_demand_matrix == test_case.order_demand_mat

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_find_pflow_simple(self, test_case: OpenGraphTestCase) -> None:
        if test_case.flow_demand_mat is not None:
            ogi = test_case.ogi
            if len(ogi.og.outputs) == len(ogi.og.inputs):
                pflow_algebraic = _find_pflow_simple(ogi)

                if not test_case.has_pflow:
                    assert pflow_algebraic is None
                else:
                    assert pflow_algebraic is not None
                    correction_matrix, _ = pflow_algebraic
                    ident = MatGF2(np.eye(len(ogi.non_outputs), dtype=np.int_))
                    assert test_case.flow_demand_mat @ correction_matrix == ident

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_find_pflow_determinism(self, test_case: OpenGraphTestCase, fx_rng: Generator) -> None:
        og = test_case.ogi.og

        pflow = find_pflow(og)

        if not test_case.has_pflow:
            assert pflow is None
        else:
            assert pflow is not None

            pattern = _pflow2pattern(
                graph=og.inside,
                inputs=set(og.inputs),
                meas_planes={i: m.plane for i, m in og.measurements.items()},
                angles={i: m.angle for i, m in og.measurements.items()},
                p=pflow[0],
                l_k=pflow[1],
            )
            pattern.reorder_output_nodes(og.outputs)

            alpha = 2 * np.pi * fx_rng.random()
            state_ref = pattern.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))

            n_shots = 5
            results = []
            for _ in range(n_shots):
                state = pattern.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))
                results.append(np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())))

            avg = sum(results) / n_shots

            assert avg == pytest.approx(1)

    @pytest.mark.parametrize("test_case", prepare_test_back_subs())
    def test_back_substitute(self, test_case: BackSubsTestCase) -> None:
        mat = test_case.mat
        b = test_case.b

        x = _back_substitute(mat, b)

        assert mat @ x == b
