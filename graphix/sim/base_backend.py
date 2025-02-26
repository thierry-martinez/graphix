"""Abstract base class for simulation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.ops import Ops
from graphix.rng import ensure_rng
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.random import Generator
    import numpy.typing as npt

    from graphix.fundamentals import Plane
    from graphix.measurements import Measurement


class NodeIndex:
    """A class for managing the mapping between node numbers and qubit indices in the internal state of the backend.

    This allows for efficient access and manipulation of qubit orderings throughout the execution of a pattern.

    Attributes
    ----------
        __list (list): A private list of the current active node (labelled with integers).
        __dict (dict): A private dictionary mapping current node labels (integers) to their corresponding qubit indices
                       in the backend's internal quantum state.
    """

    def __init__(self) -> None:
        self.__dict = {}
        self.__list = []

    def __getitem__(self, index: int) -> int:
        """Return the qubit node associated with the specified index."""
        return self.__list[index]

    def index(self, node: int) -> int:
        """Return the qubit index associated with the specified node label."""
        return self.__dict[node]

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over indices."""
        return iter(self.__list)

    def __len__(self) -> int:
        """Return the number of currently active nodes."""
        return len(self.__list)

    def extend(self, nodes: Iterable[int]) -> None:
        """Extend the list with a sequence of node labels, updating the dictionary by assigning them sequential qubit indices."""
        base = len(self)
        self.__list.extend(nodes)
        # The following loop iterates over `self.__list[base:]` instead of `nodes`
        # because the iterable `nodes` can be transient and consumed by the
        # `self.__list.extend` on the line just above.
        for index, node in enumerate(self.__list[base:]):
            self.__dict[node] = base + index

    def remove(self, node: int) -> None:
        """Remove the specified node label from the list and dictionary, and re-attributes qubit indices for the remaining nodes."""
        index = self.__dict[node]
        del self.__list[index]
        del self.__dict[node]
        for new_index, u in enumerate(self.__list[index:], start=index):
            self.__dict[u] = new_index

    def swap(self, i: int, j: int) -> None:
        """Swap two nodes given their indices."""
        node_i = self.__list[i]
        node_j = self.__list[j]
        self.__list[i] = node_j
        self.__list[j] = node_i
        self.__dict[node_i] = j
        self.__dict[node_j] = i


class State:
    """Base class for backend state."""


def _op_mat_from_result(vec: tuple[float, float, float], result: bool) -> np.ndarray:
    op_mat = np.eye(2, dtype=np.complex128) / 2
    sign = (-1) ** result
    for i in range(3):
        op_mat += sign * vec[i] * Clifford(i + 1).matrix / 2
    return op_mat


def perform_measure(qubit_node: int, qubit_loc: int, plane: Plane, angle: float, state, selector: BranchSelector) -> bool:
    """Perform measurement of a qubit."""
    vec = plane.polar(angle)
    # op_mat_0 may contain the matrix operator associated with the outcome 0,
    # but the value is computed lazily, i.e., only if needed.
    op_mat_0 = None
    def get_op_mat_0() -> np.ndarray:
        nonlocal op_mat_0
        if op_mat_0 is None:
            op_mat_0 = _op_mat_from_result(vec, False)
        return op_mat_0
    def compute_expectation_0() -> float:
        return state.expectation_single(get_op_mat_0(), qubit_loc)
    result = selector.measure(qubit_node, compute_expectation_0)
    if result:
        op_mat = _op_mat_from_result(vec, True)
    else:
        op_mat = get_op_mat_0()
    state.evolve_single(op_mat, qubit_loc)
    return result


class BranchSelector(ABC):
    """Abstract class for branch selectors.

    Branch selectors determine the computation branch that is explored
    during a simulation, meaning the choice of measurement outcomes.
    The branch selection can be random (see :class:`RandomBranchSelector`)
    or deterministic (see :class:`ConstBranchSelector`).

    A branch selector provides the method `measure`, which returns the
    measurement outcome (0 or 1) for a given qubit.
    """

    @abstractmethod
    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> bool:
        """
        Return the measurement outcome of `qubit`.

        This method may use the function `compute_expectation_0` to
        retrieve the expected probability of outcome 0. The probability is
        computed only if this function is called (lazy computation),
        ensuring no unnecessary computational cost.
        """


@dataclass
class RandomBranchSelector(BranchSelector):
    """Random branch selector.

    Parameters
    ----------
    pr_calc : bool, optional
        Whether to compute the probability distribution before selecting the measurement result.
        If False, measurements yield 0/1 with equal probability (50% each).
        Default is `True`.
    rng : Generator | None, optional
        Random-number generator for measurements.
        If `None`, a default random-number generator is used.
        Default is `None`.
    """

    pr_calc: bool = True
    rng: Generator | None = None

    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> bool:
        """
        Return the measurement outcome of `qubit`.

        If `pr_calc` is `True`, the measurement outcome is determined based on the
        computed probability of outcome 0. Otherwise, the result is randomly chosen
        with a 50% chance for either outcome.
        """
        self.rng = ensure_rng(self.rng)
        if self.pr_calc:
            prob_0 = compute_expectation_0()
            return self.rng.random() > prob_0
        return self.rng.choice([0, 1]) == 1


@dataclass
class FixedBranchSelector(BranchSelector):
    """Branch selector with predefined measurement outcomes.

    The mapping is fixed in `results`. By default, an error is raised if
    a qubit is measured without a predefined outcome. However, another
    branch selector can be specified in `default` to handle such cases.

    Parameters
    ----------
    results : Mapping[int, bool]
        A dictionary mapping qubits to their measurement outcomes.
        If a qubit is not present in this mapping, the `default` branch
        selector is used.
    default : BranchSelector | None, optional
        Branch selector to use for qubits not present in `results`.
        If `None`, an error is raised when an unmapped qubit is measured.
        Default is `None`.
    """

    results: Mapping[int, bool]
    default: BranchSelector | None = None

    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> bool:
        """
        Return the predefined measurement outcome of `qubit`, if available.

        If the qubit is not present in `results`, the `default` branch selector
        is used. If no default is provided, an error is raised.
        """
        result = self.results.get(qubit)
        if result is None:
            if self.default is None:
                raise ValueError(f"Unexpected measurement of qubit {qubit}.")
            return self.default.measure(qubit, compute_expectation_0)
        return result


@dataclass
class ConstBranchSelector(BranchSelector):
    """Branch selector with a constant measurement outcome.

    The value `result` is returned for every qubit.

    Parameters
    ----------
    result : bool
        The fixed measurement outcome for all qubits.
    """

    result: bool

    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> bool:
        """
        Return the constant measurement outcome `result` for any qubit.
        """
        return self.result


class Backend:
    """Base class for backends."""

    def __init__(
        self,
        state: State,
        node_index: NodeIndex | None = None,
        branch_selector: BranchSelector | None = None,
        pr_calc: bool | None = None,
        rng: Generator | None = None,
    ):
        """Construct a backend.

        Parameters
        ----------
            pr_calc : bool
                whether or not to compute the probability distribution before choosing the measurement result.
                if False, measurements yield results 0/1 with 50% probabilities each.
            node_index : NodeIndex
                mapping between node numbers and qubit indices in the internal state of the backend.
            state : State
                internal state of the backend: instance of Statevec, DensityMatrix, or MBQCTensorNet.

        """
        self.__state = state
        if node_index is None:
            self.__node_index = NodeIndex()
        else:
            self.__node_index = node_index.copy()
        # whether to compute the probability
        if branch_selector is None:
            if pr_calc is None:
                pr_calc = True
            self.__branch_selector = RandomBranchSelector(pr_calc=pr_calc, rng=rng)
        else:
            if pr_calc is not None or rng is not None:
                raise ValueError("Cannot specify both branch selector and pr_calc/rng")
            self.__branch_selector = branch_selector

    def copy(self) -> Backend:
        """Return a copy of the backend."""
        return Backend(self.__state, self.__node_index, self.__pr_calc, self.__rng)

    @property
    def rng(self) -> Generator:
        """Return the associated random-number generator."""
        return self.__rng

    @property
    def state(self) -> State:
        """Return the state of the backend."""
        return self.__state

    @property
    def node_index(self) -> NodeIndex:
        """Return the node index table of the backend."""
        return self.__node_index

    def add_nodes(self, nodes, data=BasicStates.PLUS) -> None:
        """Add new qubit(s) to statevector in argument and assign the corresponding node number to list self.node_index.

        Parameters
        ----------
        nodes : list of node indices
        """
        self.state.add_nodes(nqubit=len(nodes), data=data)
        self.node_index.extend(nodes)

    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """Apply CZ gate to two connected nodes.

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.state.entangle((target, control))

    def measure(self, node: int, measurement: Measurement) -> bool:
        """Perform measurement of a node and trace out the qubit.

        Parameters
        ----------
        node: int
        measurement: Measurement
        """
        loc = self.node_index.index(node)
        result = perform_measure(node, loc, measurement.plane, measurement.angle, self.state, self.__branch_selector)
        self.node_index.remove(node)
        self.state.remove_qubit(loc)
        return result

    def correct_byproduct(self, cmd, measure_method) -> None:
        """Byproduct correction correct for the X or Z byproduct operators, by applying the X or Z gate."""
        if np.mod(sum([measure_method.get_measure_result(j) for j in cmd.domain]), 2) == 1:
            if cmd.kind == CommandKind.X:
                op = Ops.X
            elif cmd.kind == CommandKind.Z:
                op = Ops.Z
            self.apply_single(node=cmd.node, op=op)

    def apply_single(self, node, op) -> None:
        """Apply a single gate to the state."""
        index = self.node_index.index(node)
        self.state.evolve_single(op=op, i=index)

    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """Apply single-qubit Clifford gate, specified by vop index specified in graphix.clifford.CLIFFORD."""
        loc = self.node_index.index(node)
        self.state.evolve_single(clifford.matrix, loc)

    def sort_qubits(self, output_nodes) -> None:
        """Sort the qubit order in internal statevector."""
        for i, ind in enumerate(output_nodes):
            if self.node_index.index(ind) != i:
                move_from = self.node_index.index(ind)
                self.state.swap((i, move_from))
                self.node_index.swap(i, move_from)

    def finalize(self, output_nodes) -> None:
        """To be run at the end of pattern simulation."""
        self.sort_qubits(output_nodes)

    @property
    def nqubit(self) -> int:
        """Return the number of qubits of the current state."""
        return self.state.nqubit
