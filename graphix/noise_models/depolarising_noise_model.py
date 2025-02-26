"""Depolarising noise model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typing_extensions

from graphix.channels import KrausChannel, depolarising_channel, two_qubit_depolarising_channel
from graphix.command import Command, CommandKind
from graphix.noise_models.noise_model import Noise, NoiseModel
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from numpy.random import Generator


class DepolarisingNoiseModel(NoiseModel):
    """Depolarising noise model.

    Only return the identity channel.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        prepare_error_prob: float = 0.0,
        x_error_prob: float = 0.0,
        z_error_prob: float = 0.0,
        entanglement_error_prob: float = 0.0,
        measure_channel_prob: float = 0.0,
        measure_error_prob: float = 0.0,
        rng: Generator = None,
    ) -> None:
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob
        self.rng = ensure_rng(rng)

    def input_nodes(self, nodes: list[int]) -> Noise:
        """Return the noise to apply to input nodes."""
        return [(depolarising_channel(self.prepare_error_prob), [node]) for node in nodes]

    def command(self, cmd: Command) -> Noise:
        """Return the noise to apply to the command `cmd`."""
        kind = cmd.kind
        if kind == CommandKind.N:
            return [(depolarising_channel(self.prepare_error_prob), [cmd.node])]
        if kind == CommandKind.E:
            return [(two_qubit_depolarising_channel(self.entanglement_error_prob), cmd.nodes)]
        if kind == CommandKind.M:
            return [(depolarising_channel(self.measure_channel_prob), [cmd.node])]
        if kind == CommandKind.X:
            return [(depolarising_channel(self.x_error_prob), [cmd.node])]
        if kind == CommandKind.Z:
            return [(depolarising_channel(self.z_error_prob), [cmd.node])]
        if kind == CommandKind.C or kind == CommandKind.T:
            return [(KrausChannel([]), [])]
        typing_extensions.assert_never(kind)

    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        if self.rng.uniform() < self.measure_error_prob:
            return not result
        else:
            return result
