"""24 Unique single-qubit Clifford gates and their multiplications, conjugations and Pauli conjugations."""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, ClassVar

from graphix._db import (
    CLIFFORD,
    CLIFFORD_CONJ,
    CLIFFORD_HSZ_DECOMPOSITION,
    CLIFFORD_LABEL,
    CLIFFORD_MEASURE,
    CLIFFORD_MUL,
    CLIFFORD_TO_QASM3,
)
from graphix.pauli import IXYZ, ComplexUnit, Pauli, Sign

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt


@dataclasses.dataclass
class Domains:
    """
    Represent `X^sZ^t` where s and t are XOR of results from given sets of indices.

    This representation is used in `Clifford.commute_domains`.
    """

    s_domain: set[int]
    t_domain: set[int]


_TABLE = None


class CliffordMeta(type):
    """Meta-class for Clifford."""

    def __len__(cls) -> int:
        """Return the number of Clifford gates."""
        return len(_TABLE)

    def __iter__(cls) -> Iterator[Clifford]:
        """
        Return an iterator over Clifford gates.

        To get a list of all Clifford gates, one can write `list(Clifford)`.
        """
        return iter(_TABLE)


@dataclasses.dataclass(frozen=True)
class Clifford(metaclass=CliffordMeta):
    """Clifford gate."""

    index: int

    I: ClassVar[Clifford]
    X: ClassVar[Clifford]
    Y: ClassVar[Clifford]
    Z: ClassVar[Clifford]
    S: ClassVar[Clifford]
    H: ClassVar[Clifford]

    def __new__(cls, index: int) -> Clifford:
        """
        Return the Clifford gate corresponding to the given index.

        Clifford instances are singleton values: two calls to the
        constructor `Clifford` with the same `index` returns the same
        instance, e.g., `Clifford.I is Clifford(0)`.
        """
        if _TABLE is None:
            # This branch is only run on module initialization (construction of _TABLE).
            return super().__new__(cls)
        # The following line raises an exception if index is out of bound.
        return _TABLE[index]

    def __copy__(self) -> Clifford:
        """
        Return the Clifford gate itself.

        Clifford instances are singleton values: they are never duplicated.
        """
        return self

    def __deepcopy__(self, _memo) -> Clifford:
        """
        Return the Clifford gate itself.

        Clifford instances are singleton values: they are never duplicated.
        """
        return self

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return the matrix of the Clifford gate."""
        return CLIFFORD[self.index]

    def __repr__(self) -> str:
        """Return the Clifford expression on the form of HSZ decomposition."""
        return " @ ".join([f"Clifford.{gate}" for gate in self.hsz])

    def __str__(self) -> str:
        """Return the name of the Clifford gate."""
        return CLIFFORD_LABEL[self.index]

    @property
    def conj(self) -> Clifford:
        """Return the conjugate of the Clifford gate."""
        return Clifford(CLIFFORD_CONJ[self.index])

    @property
    def hsz(self) -> list[Clifford]:
        """Return a decomposition of the Clifford gate with the gates `H`, `S`, `Z`."""
        return [Clifford(i) for i in CLIFFORD_HSZ_DECOMPOSITION[self.index]]

    @property
    def qasm3(self) -> tuple[str, ...]:
        """Return a decomposition of the Clifford gate as qasm3 gates."""
        return CLIFFORD_TO_QASM3[self.index]

    def __matmul__(self, other: Clifford) -> Clifford:
        """Multiplication within the Clifford group (modulo unit factor)."""
        if isinstance(other, Clifford):
            return Clifford(CLIFFORD_MUL[self.index][other.index])
        return NotImplemented

    def measure(self, pauli: Pauli) -> Pauli:
        """Compute C† P C."""
        if pauli.symbol == IXYZ.I:
            return copy.deepcopy(pauli)
        table = CLIFFORD_MEASURE[self.index]
        symbol, sign = table[pauli.symbol.value]
        return pauli.unit * Pauli(IXYZ[symbol], ComplexUnit(Sign(sign), False))

    def commute_domains(self, domains: Domains) -> Domains:
        """
        Commute `X^sZ^t` with `C`.

        Given `X^sZ^t`, return `X^s'Z^t'` such that `X^sZ^tC = CX^s'Z^t'`.

        Note that applying the method to `self.conj` computes the reverse commutation:
        indeed, `C†X^sZ^t = (X^sZ^tC)† = (CX^s'Z^t')† = X^s'Z^t'C†`.
        """
        s_domain = domains.s_domain.copy()
        t_domain = domains.t_domain.copy()
        for gate in self.hsz:
            if gate == Clifford.I:
                pass
            elif gate == Clifford.H:
                t_domain, s_domain = s_domain, t_domain
            elif gate == Clifford.S:
                t_domain ^= s_domain
            elif gate == Clifford.Z:
                pass
            else:
                raise RuntimeError(f"{gate} should be either I, H, S or Z.")
        return Domains(s_domain, t_domain)


_TABLE = [Clifford(index) for index in range(len(CLIFFORD))]

Clifford.I = Clifford(0)
Clifford.X = Clifford(1)
Clifford.Y = Clifford(2)
Clifford.Z = Clifford(3)
Clifford.S = Clifford(4)
Clifford.H = Clifford(6)
