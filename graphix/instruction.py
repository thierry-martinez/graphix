"""Instruction classes."""

from __future__ import annotations

import dataclasses
import enum
import sys
from enum import Enum
from typing import ClassVar, Literal, Union

import numpy as np

from graphix import type_utils
from graphix.pauli import Plane


class InstructionKind(Enum):
    """Tag for instruction kind."""

    CCX = enum.auto()
    RZZ = enum.auto()
    CNOT = enum.auto()
    SWAP = enum.auto()
    H = enum.auto()
    S = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()
    I = enum.auto()
    M = enum.auto()
    RX = enum.auto()
    RY = enum.auto()
    RZ = enum.auto()
    CZ = enum.auto()
    J = enum.auto()
    # The two following instructions are used internally by the transpiler
    _XC = enum.auto()
    _ZC = enum.auto()


class _KindChecker:
    """Enforce tag field declaration."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        type_utils.check_kind(cls, {"InstructionKind": InstructionKind, "Plane": Plane})


@dataclasses.dataclass
class CCX(_KindChecker):
    """Toffoli circuit instruction."""

    target: int
    controls: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CCX]] = dataclasses.field(default=InstructionKind.CCX, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The decomposition of Toffoli gate into H, CNOT, T and T-dagger gates can be found in
        Michael A. Nielsen and Isaac L. Chuang,
        Quantum Computation and Quantum Information,
        Cambridge University Press, 2000
        (p. 182 in the 10th Anniversary Edition).
        These gates are in turn decomposed into J and ∧z gates.
        """
        return [
            jcz
            for gate in [
                H(target=self.target),
                CNOT(control=self.controls[1], target=self.target),
                RZ(target=self.target, angle=-np.pi / 4),
                CNOT(control=self.controls[0], target=self.target),
                RZ(target=self.target, angle=np.pi / 4),
                CNOT(control=self.controls[1], target=self.target),
                RZ(target=self.target, angle=-np.pi / 4),
                CNOT(control=self.controls[0], target=self.target),
                RZ(target=self.controls[1], angle=np.pi / 4),
                RZ(target=self.target, angle=np.pi / 4),
                CNOT(control=self.controls[0], target=self.controls[1]),
                H(target=self.target),
                RZ(target=self.controls[0], angle=np.pi / 4),
                RZ(target=self.controls[1], angle=-np.pi / 4),
                CNOT(control=self.controls[0], target=self.controls[1]),
            ]
            for jcz in gate.transpile_jcz()
        ]


@dataclasses.dataclass
class RZZ(_KindChecker):
    """RZZ circuit instruction."""

    target: int
    control: int
    angle: float
    # FIXME: Remove `| None` from `meas_index`
    # - `None` makes codes messy/type-unsafe
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RZZ]] = dataclasses.field(default=InstructionKind.RZZ, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The RZZ(α) gate is decomposed as CNOT(control, target)·Rz(target, α)·CNOT(control, target), which in turn decomposed into J and ∧z gates.
        """
        return [
            jcz
            for gate in [
                CNOT(control=self.control, target=self.target),
                RZ(target=self.target, angle=self.angle),
                CNOT(control=self.control, target=self.target),
            ]
            for jcz in gate.transpile_jcz()
        ]


@dataclasses.dataclass
class CNOT(_KindChecker):
    """CNOT circuit instruction."""

    target: int
    control: int
    kind: ClassVar[Literal[InstructionKind.CNOT]] = dataclasses.field(default=InstructionKind.CNOT, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The CNOT gate is decomposed as H·∧z·H, and Hadamard gates are in turn decomposed into J gates.
        """
        return [
            jcz
            for gate in [H(target=self.target), CZ(targets=(self.control, self.target)), H(target=self.target)]
            for jcz in gate.transpile_jcz()
        ]

@dataclasses.dataclass
class SWAP(_KindChecker):
    """SWAP circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.SWAP]] = dataclasses.field(default=InstructionKind.SWAP, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The gate is decomposed into CNOT(0, 1)·CNOT(1, 0)·CNOT(0, 1), that are in turn decomposed into J and ∧z gates.
        """
        return [
            jcz
            for gate in [
                CNOT(control=self.targets[0], target=self.targets[1]),
                CNOT(control=self.targets[1], target=self.targets[0]),
                CNOT(control=self.targets[0], target=self.targets[1]),
            ]
            for jcz in gate.transpile_jcz()
        ]


@dataclasses.dataclass
class CZ(_KindChecker):
    """CZ circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CZ]] = dataclasses.field(default=InstructionKind.CZ, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The decomposition of the ∧z gate is the gate itself.
        """
        return [self]


@dataclasses.dataclass
class J(_KindChecker):
    """J circuit instruction."""

    target: int
    angle: float
    kind: ClassVar[Literal[InstructionKind.J]] = dataclasses.field(default=InstructionKind.J, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The decomposition of the J gate is the gate itself.
        """
        return [self]


@dataclasses.dataclass
class H(_KindChecker):
    """H circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.H]] = dataclasses.field(default=InstructionKind.H, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The decomposition of the H gate is J(0).
        Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.
        """
        return [J(target=self.target, angle=0)]


@dataclasses.dataclass
class S(_KindChecker):
    """S circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.S]] = dataclasses.field(default=InstructionKind.S, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The S gate is first rewritten as Rz(π/2), which is in turn decomposed into J and ∧z gates.
        """
        return [jcz for jcz in RZ(target=self.target, angle=np.pi / 2).transpile_jcz()]


@dataclasses.dataclass
class X(_KindChecker):
    """X circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.X]] = dataclasses.field(default=InstructionKind.X, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The X gate is first rewritten as Rx(π), which is in turn decomposed into J and ∧z gates.
        """
        return [jcz for jcz in RX(target=self.target, angle=np.pi).transpile_jcz()]


@dataclasses.dataclass
class Y(_KindChecker):
    """Y circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.Y]] = dataclasses.field(default=InstructionKind.Y, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The Y gate is first rewritten as X·Z, which is in turn decomposed into J and ∧z gates.
        """
        return [jcz for gate in reversed((X, Z)) for jcz in gate(target=self.target).transpile_jcz()]


@dataclasses.dataclass
class Z(_KindChecker):
    """Z circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.Z]] = dataclasses.field(default=InstructionKind.Z, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The Z gate is first rewritten as Rz(π), which is in turn decomposed into J and ∧z gates.
        """
        return [jcz for jcz in RZ(target=self.target, angle=np.pi).transpile_jcz()]


@dataclasses.dataclass
class I(_KindChecker):
    """I circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.I]] = dataclasses.field(default=InstructionKind.I, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The identity is translated into the empty list of gates.
        """
        return []


@dataclasses.dataclass
class M(_KindChecker):
    """M circuit instruction."""

    target: int
    plane: Plane
    angle: float
    kind: ClassVar[Literal[InstructionKind.M]] = dataclasses.field(default=InstructionKind.M, init=False)


@dataclasses.dataclass
class RX(_KindChecker):
    """X rotation circuit instruction."""

    target: int
    angle: float
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RX]] = dataclasses.field(default=InstructionKind.RX, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The Rx(α) gate is decomposed into J(α)·H (that is to say, J(α)·J(0)).
        Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.
        """
        return [J(target=self.target, angle=angle) for angle in reversed((self.angle, 0))]


@dataclasses.dataclass
class RY(_KindChecker):
    """Y rotation circuit instruction."""

    target: int
    angle: float
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RY]] = dataclasses.field(default=InstructionKind.RY, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The Ry(α) gate is decomposed into J(0)·J(π/2)·J(α)·J(-π/2).
        Vincent Danos, Elham Kashefi, Prakash Panangaden, Robust and parsimonious realisations of unitaries in the one-way model, 2004.
        """
        return [J(target=self.target, angle=angle) for angle in reversed((0, np.pi / 2, self.angle, -np.pi / 2))]


@dataclasses.dataclass
class RZ(_KindChecker):
    """Z rotation circuit instruction."""

    target: int
    angle: float
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RZ]] = dataclasses.field(default=InstructionKind.RZ, init=False)

    def transpile_jcz(self) -> list[J | CZ]:
        """Return the J-∧z decomposition of the instruction.

        The Rz(α) gate is decomposed into H·J(α) (that is to say, J(0)·J(α)).
        Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.
        """
        return [J(target=self.target, angle=angle) for angle in reversed((0, self.angle))]


@dataclasses.dataclass
class _XC(_KindChecker):
    """X correction circuit instruction. Used internally by the transpiler."""

    target: int
    domain: set[int]
    kind: ClassVar[Literal[InstructionKind._XC]] = dataclasses.field(default=InstructionKind._XC, init=False)


@dataclasses.dataclass
class _ZC(_KindChecker):
    """Z correction circuit instruction. Used internally by the transpiler."""

    target: int
    domain: set[int]
    kind: ClassVar[Literal[InstructionKind._ZC]] = dataclasses.field(default=InstructionKind._ZC, init=False)


if sys.version_info >= (3, 10):
    Instruction = CCX | RZZ | CNOT | SWAP | H | S | X | Y | Z | I | M | RX | RY | RZ | _XC | _ZC
else:
    Instruction = Union[CCX, RZZ, CNOT, SWAP, H, S, X, Y, Z, I, M, RX, RY, RZ, _XC, _ZC]
