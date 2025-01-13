"""Abstract base class for all noise models."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphix.channels import KrausChannel

if TYPE_CHECKING:
    from graphix.command import Command


Noise = list[tuple[KrausChannel, list[int]]]
"""
A noise is specified by a list of tuples `(channel, qubits)`,
where `channel` is a Kraus channel to apply to the given `qubits`.
"""


class NoiseModel(abc.ABC):
    """Abstract base class for all noise models."""

    @abc.abstractmethod
    def input_nodes(self, nodes: list[int]) -> Noise:
        """Return the noise to apply to input nodes."""

    @abc.abstractmethod
    def command(self, cmd: Command) -> Noise:
        """Return the noise to apply to the command `cmd`."""

    @abc.abstractmethod
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
