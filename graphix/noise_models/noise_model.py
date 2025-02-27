"""Abstract base class for all noise models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphix.channels import KrausChannel

if TYPE_CHECKING:
    from graphix.command import Command


class NoiseElement(ABC):
    @abstractmethod
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""

    @abstractmethod
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""


Noise = list[tuple[NoiseElement, list[int]]]
"""
A noise is specified by a list of tuples `(element, qubits)`,
where `element` is a noise element to apply to the given `qubits`.
"""


class NoiseModel(ABC):
    """Abstract base class for all noise models."""

    @abstractmethod
    def input_nodes(self, nodes: list[int]) -> Noise:
        """Return the noise to apply to input nodes."""

    @abstractmethod
    def command(self, cmd: Command) -> Noise:
        """Return the noise to apply to the command `cmd`."""

    @abstractmethod
    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""


@dataclass(frozen=True)
class ComposeNoiseModel(NoiseModel):
    """Compose noise models."""

    l: list[NoiseModel]

    def input_nodes(self, nodes: list[int]) -> Noise:
        """Return the noise to apply to input nodes."""
        return [channel_qubits for m in self.l for channel_qubits in m.input_nodes(nodes)]

    def command(self, cmd: Command) -> Noise:
        """Return the noise to apply to the command `cmd`."""
        return [channel_qubits for m in self.l for channel_qubits in m.command(cmd)]

    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""
        for m in self.l:
            result = m.confuse_result(result)
        return result
