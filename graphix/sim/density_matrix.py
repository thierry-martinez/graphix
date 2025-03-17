"""Density matrix simulator.

Simulate MBQC with density matrix representation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import sys
import numbers

import copy

import numpy as np

from graphix import linalg_validations as lv
from graphix import states
from graphix.channels import KrausChannel
from graphix.sim.base_backend import Backend, State
from graphix.sim.statevec import CNOT_TENSOR, CZ_TENSOR, SWAP_TENSOR, Statevec
from graphix.states import BasicStates, PlanarState
import dm_simu_rs

if TYPE_CHECKING:
    from numpy.random import Generator


class DensityMatrix(State):
    """DensityMatrix object."""

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
    ):
        """Initialize density matrix objects.

        The behaviour builds on the one of `graphix.statevec.Statevec`.
        `data` can be:
        - a single :class:`graphix.states.State` (classical description of a quantum state)
        - an iterable of :class:`graphix.states.State` objects
        - an iterable of iterable of scalars (A `2**n x 2**n` numerical density matrix)
        - a `graphix.statevec.DensityMatrix` object
        - a `graphix.statevec.Statevector` object

        If `nqubit` is not provided, the number of qubit is inferred from `data` and checked for consistency.
        If only one :class:`graphix.states.State` is provided and nqubit is a valid integer, initialize the statevector
        in the tensor product state.
        If both `nqubit` and `data` are provided, consistency of the dimensions is checked.
        If a `graphix.statevec.Statevec` or `graphix.statevec.DensityMatrix` is passed, returns a copy.


        :param data: input data to prepare the state. Can be a classical description or a numerical input, defaults to graphix.states.BasicStates.PLUS
        :type data: Data
        :param nqubit: number of qubits to prepare, defaults to `None`
        :type nqubit: int, optional
        """
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        def check_size_consistency(mat):
            if nqubit is not None and mat.shape != (2**nqubit, 2**nqubit):
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the shape of the provided density matrix = {mat.shape}."
                )

        if isinstance(data, DensityMatrix):
            check_size_consistency(data)
            # safe: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html
            self.rho = data.rho.copy()
            return
        if isinstance(data, Iterable):
            input_list = list(data)
            if len(input_list) != 0:
                # needed since Object is iterable but not subscribable!
                try:
                    if isinstance(input_list[0], Iterable) and isinstance(input_list[0][0], numbers.Number):
                        self.rho = np.array(input_list)
                        if not lv.is_qubitop(self.rho):
                            raise ValueError("Cannot interpret the provided density matrix as a qubit operator.")
                        check_size_consistency(self.rho)
                        if not lv.is_unit_trace(self.rho):
                            raise ValueError("Density matrix must have unit trace.")
                        if not lv.is_psd(self.rho):
                            raise ValueError("Density matrix must be positive semi-definite.")
                        return
                except TypeError:
                    pass
        statevec = Statevec(data, nqubit)
        # NOTE this works since np.outer flattens the inputs!
        self.rho = np.outer(statevec.psi, statevec.psi.conj())

    @property
    def nqubit(self) -> int:
        """Return the number of qubits."""
        return self.rho.shape[0].bit_length() - 1

    def __str__(self) -> str:
        """Return a string description."""
        return f"DensityMatrix object, with density matrix {self.rho} and shape {self.dims()}."

    def add_nodes(self, nqubit, data) -> None:
        """Add nodes to the density matrix."""
        dm_to_add = DensityMatrix(nqubit=nqubit, data=data)
        self.tensor(dm_to_add)

    def evolve_single(self, op, i) -> None:
        """Single-qubit operation.

        Parameters
        ----------
            op : np.ndarray
                2*2 matrix.
            i : int
                Index of qubit to apply operator.
        """
        assert i >= 0 and i < self.nqubit
        if op.shape != (2, 2):
            raise ValueError("op must be 2*2 matrix.")

        rho_tensor = self.rho.reshape((2,) * self.nqubit * 2)
        rho_tensor = np.tensordot(np.tensordot(op, rho_tensor, axes=(1, i)), op.conj().T, axes=(i + self.nqubit, 0))
        rho_tensor = np.moveaxis(rho_tensor, (0, -1), (i, i + self.nqubit))
        self.rho = rho_tensor.reshape((2**self.nqubit, 2**self.nqubit))

    def evolve(self, op, qargs) -> None:
        """Multi-qubit operation.

        Args:
            op (np.array): 2^n*2^n matrix
            qargs (list of ints): target qubits' indexes
        """
        d = op.shape
        # check it is a matrix.
        if len(d) == 2:
            # check it is square
            if d[0] == d[1]:
                pass
            else:
                raise ValueError(f"The provided operator has shape {op.shape} and is not a square matrix.")
        else:
            raise ValueError(f"The provided data has incorrect shape {op.shape}.")

        nqb_op = np.log2(len(op))
        if not np.isclose(nqb_op, int(nqb_op)):
            raise ValueError("Incorrect operator dimension: not consistent with qubits.")
        nqb_op = int(nqb_op)

        if nqb_op != len(qargs):
            raise ValueError("The dimension of the operator doesn't match the number of targets.")

        if not all(0 <= i < self.nqubit for i in qargs):
            raise ValueError("Incorrect target indices.")
        if len(set(qargs)) != nqb_op:
            raise ValueError("A repeated target qubit index is not possible.")

        op_tensor = op.reshape((2,) * 2 * nqb_op)

        rho_tensor = self.rho.reshape((2,) * self.nqubit * 2)

        rho_tensor = np.tensordot(
            np.tensordot(op_tensor, rho_tensor, axes=[tuple(nqb_op + i for i in range(len(qargs))), tuple(qargs)]),
            op.conj().T.reshape((2,) * 2 * nqb_op),
            axes=[tuple(i + self.nqubit for i in qargs), tuple(i for i in range(len(qargs)))],
        )
        rho_tensor = np.moveaxis(
            rho_tensor,
            [i for i in range(len(qargs))] + [-i for i in range(1, len(qargs) + 1)],
            [i for i in qargs] + [i + self.nqubit for i in reversed(list(qargs))],
        )
        self.rho = rho_tensor.reshape((2**self.nqubit, 2**self.nqubit))

    def expectation_single(self, op, i) -> complex:
        """Return the expectation value of single-qubit operator.

        Args:
            op (np.array): 2*2 Hermite operator
            loc (int): Index of qubit on which to apply operator.

        Returns
        -------
            complex: expectation value (real for hermitian ops!).
        """
        if not (0 <= i < self.nqubit):
            raise ValueError(f"Wrong target qubit {i}. Must between 0 and {self.nqubit-1}.")

        if op.shape != (2, 2):
            raise ValueError("op must be 2x2 matrix.")

        st1 = copy.copy(self)
        st1.normalize()

        rho_tensor = st1.rho.reshape((2,) * st1.nqubit * 2)
        rho_tensor = np.tensordot(op, rho_tensor, axes=[1, i])
        rho_tensor = np.moveaxis(rho_tensor, 0, i)

        return np.trace(rho_tensor.reshape((2**self.nqubit, 2**self.nqubit)))

    def dims(self):
        """Return the dimensions of the density matrix."""
        return self.rho.shape

    def tensor(self, other) -> None:
        r"""Tensor product state with other density matrix.

        Results in self :math:`\otimes` other.

        Parameters
        ----------
            other : :class: `DensityMatrix` object
                DensityMatrix object to be tensored with self.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        self.rho = np.kron(self.rho, other.rho)

    def cnot(self, edge) -> None:
        """Apply CNOT gate to density matrix.

        Parameters
        ----------
            edge : (int, int) or [int, int]
                Edge to apply CNOT gate.
        """
        self.evolve(CNOT_TENSOR.reshape(4, 4), edge)

    def swap(self, edge) -> None:
        """Swap qubits.

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubits indices.
        """
        self.evolve(SWAP_TENSOR.reshape(4, 4), edge)

    def entangle(self, edge) -> None:
        """Connect graph nodes.

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubit indices.
        """
        self.evolve(CZ_TENSOR.reshape(4, 4), edge)

    def normalize(self) -> None:
        """Normalize density matrix."""
        self.rho = self.rho / np.trace(self.rho)

    def remove_qubit(self, loc) -> None:
        """Remove a qubit."""
        self.ptrace(loc)
        self.normalize()

    def ptrace(self, qargs) -> None:
        """Partial trace.

        Parameters
        ----------
            qargs : list of ints or int
                Indices of qubit to trace out.
        """
        n = int(np.log2(self.rho.shape[0]))
        if isinstance(qargs, int):
            qargs = [qargs]
        assert isinstance(qargs, (list, tuple))
        qargs_num = len(qargs)
        nqubit_after = n - qargs_num
        assert n > 0
        assert all([qarg >= 0 and qarg < n for qarg in qargs])

        rho_res = self.rho.reshape((2,) * n * 2)
        # ket, bra indices to trace out
        trace_axes = list(qargs) + [n + qarg for qarg in qargs]
        rho_res = np.tensordot(
            np.eye(2**qargs_num).reshape((2,) * qargs_num * 2), rho_res, axes=(list(range(2 * qargs_num)), trace_axes)
        )

        self.rho = rho_res.reshape((2**nqubit_after, 2**nqubit_after))

    def fidelity(self, statevec):
        """Calculate the fidelity against reference statevector.

        Parameters
        ----------
            statevec : numpy array
                statevector (flattened numpy array) to compare with
        """
        return np.abs(statevec.transpose().conj() @ self.rho @ statevec)

    def apply_channel(self, channel: KrausChannel, qargs) -> None:
        """Apply a channel to a density matrix.

        Parameters
        ----------
        :rho: density matrix.
        channel: :class:`graphix.channel.KrausChannel` object
            KrausChannel to be applied to the density matrix
        qargs: target qubit indices

        Returns
        -------
        nothing

        Raises
        ------
        ValueError
            If the final density matrix is not normalized after application of the channel.
            This shouldn't happen since :class:`graphix.channel.KrausChannel` objects are normalized by construction.
        ....
        """
        if not isinstance(channel, KrausChannel):
            raise TypeError("Can't apply a channel that is not a Channel object.")

        if channel.nqubit == 0:
            if len(qargs) != 0:
                raise ValueError("Inconsistently applying an empty Kraus channel to some qubits.")
            return

        result_array = np.zeros((2**self.nqubit, 2**self.nqubit), dtype=np.complex128)

        for k_op in channel:
            dm = copy.copy(self)
            dm.evolve(k_op.operator, qargs)
            result_array += k_op.coef * np.conj(k_op.coef) * dm.rho
            # reinitialize to input density matrix

        if not np.allclose(result_array.trace(), 1.0):
            raise ValueError("The output density matrix is not normalized, check the channel definition.")

        self.rho = result_array

class RustDensityMatrix(State):
    """Rust density matrix simulator"""
    def __init__(self, data=BasicStates.PLUS, nqubit: int | None = None):
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        def check_size_consistency(mat):
            if nqubit is not None and mat.shape != (2**nqubit, 2**nqubit):
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the shape of the provided density matrix = {mat.shape}."
                )

        if isinstance(data, DensityMatrix):
            check_size_consistency(data)
            # safe: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html
            self.rho = dm_simu_rs.new_dm_from_vec(data.rho)
            return
        if isinstance(data, RustDensityMatrix):
            check_size_consistency(data)
            # safe: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html
            self.rho = data.rho
            return
        if isinstance(data, PlanarState):
            if nqubit == 0 or nqubit is None:
                self.rho = dm_simu_rs.new_empty_dm()
                return
            
            elif data == BasicStates.PLUS:
                self.rho = dm_simu_rs.new_dm(nqubit, dm_simu_rs.Plus)
            elif data == BasicStates.MINUS:
                self.rho = dm_simu_rs.new_dm(nqubit, dm_simu_rs.Minus)
            elif data == BasicStates.ZERO:
                self.rho = dm_simu_rs.new_dm(nqubit, dm_simu_rs.Zero)
            elif data == BasicStates.ONE:
                self.rho = dm_simu_rs.new_dm(nqubit, dm_simu_rs.One)       
            else:
                raise NotImplementedError()     
            return
        if isinstance(data, Statevec):
            self.rho = dm_simu_rs.new_dm_from_statevec(data.psi.flatten())
            return
        if isinstance(data, Iterable):  # Either a sv, dm or planar state iterable.
            input_list = list(data)
            if len(input_list) == 0:
                self.rho = dm_simu_rs.new_empty_dm()
                return

            # Ensure all elements have a consistent numeric structure
            def is_numeric_sequence(seq):
                """Recursively check if all elements are numeric or numeric sequences."""
                if isinstance(seq, Iterable) and not isinstance(seq, (str, bytes)):
                    return all(is_numeric_sequence(sub) for sub in seq)
                return isinstance(seq, numbers.Number)

            if (all(isinstance(sub, PlanarState) for sub in input_list)):   # Having a list of planar states: init with Statevector backend
                statevec = Statevec(data, nqubit)
                self.rho = dm_simu_rs.new_dm_from_statevec(statevec.psi)
                return

            elif is_numeric_sequence(input_list):
                # Convert to a NumPy array and validate as a density matrix OR statevector
                try:
                    array = np.array(input_list, dtype=np.complex128)
                except ValueError as e:
                    raise TypeError("Failed to interpret iterable as a valid density matrix.") from e
                
                try:    # In case we are having an iterable representing a statevector
                    sv = Statevec(array, nqubit)
                    self.rho = dm_simu_rs.new_dm_from_statevec(sv.psi)  # Initialize density matrix from state vector representation.
                    return
                except Exception as e:
                    pass

                # Check if the iterable has a good density matrix representation.
                if not lv.is_qubitop(array):    
                    raise ValueError("Cannot interpret the provided density matrix as a qubit operator.")
                check_size_consistency(array)
                if not lv.is_unit_trace(array):
                    raise ValueError("Density matrix must have unit trace.")
                if not lv.is_psd(array):
                    raise ValueError("Density matrix must be positive semi-definite.")
                self.rho = dm_simu_rs.new_dm_from_vec(array)    # Initialize density matrix from density matrix representation.
                return
            else:
                raise TypeError("Inconsistent or invalid structure in the provided iterable.")
                return
        raise TypeError("Invalid data passed as argument.")

    @property
    def dims(self):
        nqubits = self.nqubit
        return (2 ** nqubits, 2 ** nqubits)


    @property
    def matrix(self):
        mat = dm_simu_rs.get_dm(self.rho)
        nqubits = self.nqubit
        mat = np.array(mat, dtype=np.complex128)
        return mat.reshape((2 ** nqubits, 2 ** nqubits))

    @property
    def nqubit(self):
        return dm_simu_rs.get_nqubits(self.rho)

    def __repr__(self):
        dim_size = len(self.matrix)
        return f"RustDensityMatrix object. Matrix size: {(dim_size, dim_size)}, nqubits:{self.nqubit}"

    def add_nodes(self, nqubit, data) -> None:
        """Add nodes to the density matrix."""
        dm_to_add = RustDensityMatrix(nqubit=nqubit, data=data)
        self.tensor(dm_to_add)

    def _evolve_single(self, op, target: int):
        assert target >= 0 and target < self.nqubit
        if op.shape != (2, 2):
            raise ValueError("op must be 2*2 matrix.")

        return dm_simu_rs.evolve_single(self.rho, op.flatten(), target)

    def evolve_single(self, op, target: int):
        try:
            res = self._evolve_single(op, target)
        except Exception as e:
            raise e
        dm_simu_rs.set(self.rho, res)

    def _evolve(self, op: np.ndarray, qargs: list[int]):
        d = op.shape
        # check it is a matrix.
        if len(d) == 2:
            # check it is square
            if d[0] == d[1]:
                pass
            else:
                raise ValueError(f"The provided operator has shape {op.shape} and is not a square matrix.")
        else:
            raise ValueError(f"The provided data has incorrect shape {op.shape}.")

        nqb_op = np.log2(len(op))
        if not np.isclose(nqb_op, int(nqb_op)):
            raise ValueError("Incorrect operator dimension: not consistent with qubits.")
        nqb_op = int(nqb_op)

        if nqb_op != len(qargs):
            raise ValueError("The dimension of the operator doesn't match the number of targets.")

        if not all(0 <= i < self.nqubit for i in qargs):
            raise ValueError("Incorrect target indices.")
        if len(set(qargs)) != nqb_op:
            raise ValueError("A repeated target qubit index is not possible.")
        return dm_simu_rs.evolve(self.rho, op.flatten(), qargs)

    def evolve(self, op: np.ndarray, qargs: list[int]):
        try:
            res = self._evolve(op, qargs)
        except Exception as e:
            raise e
        dm_simu_rs.set(self.rho, res)

    def normalize(self):
        """normalize density matrix"""
        dm_simu_rs.normalize(self.rho)

    def apply_channel(self, channel: KrausChannel, qargs) -> None:
        """Applies a channel to a density matrix.

        Parameters
        ----------
        :rho: density matrix.
        channel: :class:`graphix.channel.KrausChannel` object
            KrausChannel to be applied to the density matrix
        qargs: target qubit indices

        Returns
        -------
        nothing

        Raises
        ------
        ValueError
            If the final density matrix is not normalized after application of the channel.
            This shouldn't happen since :class:`graphix.channel.KrausChannel` objects are normalized by construction.
        ....
        """
        if not isinstance(channel, KrausChannel):
            raise TypeError(f"Wrong channel passed. Got {type(KrausChannel)}, expected KraussChannel.")

        channel = [
            (complex(data.coef), [complex(d) for d in data.operator.flatten()])
            for data in channel
        ]
        dm_simu_rs.apply_channel(self.rho, channel, qargs)

    def remove_qubit(self, loc) -> None:
        """Remove a qubit."""
        self.ptrace(loc)
        self.normalize()

    def tensor(self, other):
        r"""Tensor product state with other density matrix.
        Results in self :math:`\otimes` other.

        Parameters
        ----------
            other : :class: `DensityMatrix` object
                DensityMatrix object to be tensored with self.
        """
        if not isinstance(other, RustDensityMatrix):
            other = RustDensityMatrix(other)

        dm_simu_rs.tensor_dm(self.rho, other.rho)

    def _check_arg_positive(self, edge) -> bool:
        if isinstance(edge, tuple):
            return all(self._check_arg_positive(i) for i in edge)
        return edge >= 0

    def cnot(self, edge) -> None:
        if not self._check_arg_positive(edge):
            raise ValueError("Argument are not consistent.")
        new_dm = dm_simu_rs.cnot(self.rho, edge)
        dm_simu_rs.set(self.rho, new_dm)

    def entangle(self, edge):
        if not self._check_arg_positive(edge):
            raise ValueError("Argument are not consistent.")
        new_dm = dm_simu_rs.entangle(self.rho, edge)
        dm_simu_rs.set(self.rho, new_dm)

    def swap(self, edge):
        if not self._check_arg_positive(edge):
            raise ValueError("Argument are not consistent.")
        new_dm = dm_simu_rs.swap(self.rho, edge)
        dm_simu_rs.set(self.rho, new_dm)
        
        
    def expectation_single(self, op, i):
        """Expectation value of single-qubit operator.

        Args:
            op (np.array): 2*2 Hermite operator
            loc (int): Index of qubit on which to apply operator.
        Returns:
            complex: expectation value (real for hermitian ops!).
        """
        if not (0 <= i < self.nqubit):
            raise ValueError(f"Wrong target qubit {i}. Must between 0 and {self.nqubit-1}.")

        if op.shape != (2, 2):
            raise ValueError("op must be 2x2 matrix.")

        op_rs = dm_simu_rs.new_op(op.flatten())
        result = dm_simu_rs.expectation_single(self.rho, op_rs, i)
        return result

    
    def ptrace(self, qargs):
        """partial trace

        Parameters
        ----------
            qargs : list of ints or int
                Indices of qubit to trace out.
        """
        n = self.nqubit
        if isinstance(qargs, int):
            qargs = [qargs]
        assert isinstance(qargs, (list, tuple))
        assert n > 0
        assert all([qarg >= 0 and qarg < n for qarg in qargs])
        
        dm_simu_rs.ptrace(self.rho, qargs)

class DensityMatrixBackend(Backend):
    """MBQC simulator with density matrix method."""

    def __init__(self, pr_calc=False, impl=DensityMatrix, rng: Generator | None = None) -> None:
        """Construct a density matrix backend.

        Parameters
        ----------
        pattern : :class:`graphix.pattern.Pattern` object
            Pattern to be simulated.
        pr_calc : bool
            whether or not to compute the probability distribution before choosing the measurement result.
            if False, measurements yield results 0/1 with 50% probabilities each. 
        rng: :class:`np.random.Generator` (default: `None`)
            random number generator to use for measurements
        """
        super().__init__(state=impl(nqubit=0), pr_calc=pr_calc, rng=rng)

    def apply_channel(self, channel: KrausChannel, qargs) -> None:
        """Apply channel to the state.

        Parameters
        ----------
            qargs : list of ints. Target qubits
        """
        indices = [self.node_index.index(i) for i in qargs]
        self.state.apply_channel(channel, indices)


if sys.version_info >= (3, 10):
    from collections.abc import Iterable

    Data = (
        states.State
        | DensityMatrix
        | Statevec
        | Iterable[states.State]
        | Iterable[numbers.Number]
        | Iterable[Iterable[numbers.Number]]
    )
else:
    from typing import Iterable, Union

    Data = Union[
        states.State,
        DensityMatrix,
        Statevec,
        Iterable[states.State],
        Iterable[numbers.Number],
        Iterable[Iterable[numbers.Number]],
    ]
