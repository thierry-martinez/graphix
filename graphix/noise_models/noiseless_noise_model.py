"""Noiseless noise model for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.noise_models.noise_model import Noise, NoiseModel

if TYPE_CHECKING:
    from graphix.command import Command


class NoiselessNoiseModel(NoiseModel):
    """Noiseless noise model for testing.

    Only return the identity channel.
    """

    def input_nodes(self, nodes: list[int]) -> Noise:
        """Return the noise to apply to input nodes."""
        return []

    def command(self, cmd: Command) -> Noise:
        """Return the noise to apply to the command `cmd`."""
        return []

    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""
        return result
