"""Abstract base class for all noise models."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphix.channels import KrausChannel
    from graphix.command import Command


class NoiseModel(abc.ABC):
    """Abstract base class for all noise models."""

    @abc.abstractmethod
    def input_nodes(self, nodes: list[int]) -> list[tuple[KrausChannel, list[int]]]:
        """Return the noise to apply to input nodes.

        The noise is specified by a list of tuples `(channel, qubits)`,
        where `channel` is a Kraus channel to apply to the given `qubits`.
        """

    @abc.abstractmethod
    def command(self, cmd: Command) -> tuple[KrausChannel, list[int]]:
        """Return the noise to apply to the command `cmd`.

        The noise is specified by a tuple `(channel, qubits)`, where
        `channel` is a Kraus channel to apply to the given `qubits`.
        """

    @abc.abstractmethod
    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""
