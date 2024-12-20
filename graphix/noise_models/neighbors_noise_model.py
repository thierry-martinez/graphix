from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import typing_extensions

from graphix.channels import KrausChannel
from graphix.command import Command, CommandKind
from graphix.noise_models.noise_model import Noise, NoiseModel
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from numpy.random import Generator


class NeighborsNoiseModel(NoiseModel):
    """Neighbor noise model.

    Tracks the neighbor nodes of each nodes using a state graph.
    Apply channels specified in `channel_selector`.
    The entries within `channel_selector` are CommandKind.
    """

    def __init__(
        self, channel_specifier: dict[str, KrausChannel], rng: Generator = None, input_graph: nx.Graph = None
    ) -> None:
        self._state_graph: nx.Graph = input_graph  # Global tracking of the neighbors (entanglement and position)
        self.rng = ensure_rng(rng)
        self.channel_specifier = channel_specifier

    def _get_neighbors(self, nodes: list[int]) -> list[int]:
        """Return the neighbors of multiple nodes."""
        return [neighbor for n in nodes for neighbor in self._state_graph.neighbors(n)]

    def input_nodes(
        self,
        nodes: list[int],
    ) -> Noise:
        """Return the noise to apply to the input nodes' neighbors.

        For each nodes, check if they have neighbors.
        If so, compose the nodes on which the channel will be applied
        according to its number of qubits.
        """
        if self._state_graph is None:
            self._state_graph = nx.Graph(nodes)

        channel = self.channel_specifier["input"]
        noise = Noise()
        for n in nodes:  # ITerate through each nodes
            neighbors = self._state_graph.neighbors(n)  # Get neighbors
            if len(neighbors) == 0:
                continue
            channel_nqubits = channel.nqubit
            target_neighbors = []

            assert channel_nqubits <= len(
                neighbors
            ), f"Krauss channel with {channel_nqubits} qubits can not be applied to {len(neighbors) + 1} qubits."

            for i in range(
                len(neighbors) - channel_nqubits + 1
            ):  # Compose targets according to the channel's number of qubits
                target_neighbors += tuple(neighbors[i : i + channel_nqubits - 1])

            noise.extend([(channel, list(target)) for target in target_neighbors])  # Update noise

        return noise

    def command(self, cmd: Command) -> Noise:
        """Return the noise to apply to the command `cmd`.

        N: adds a node to the state graph.
        E: adds an edge to the state graph.
        M: removes a node from the state graph.
        """
        kind = cmd.kind
        if kind == CommandKind.N:
            if cmd.node not in self._state_graph:
                self._state_graph.add_node(cmd.node)

            neighbors = self._get_neighborshbors([cmd.node])

            channel = self.channel_specifier["N"]
            channel_nqubits = channel.nqubit

            target_neighbors = []

            assert channel_nqubits <= len(
                neighbors
            ), f"Krauss channel with {channel_nqubits} qubits can not be applied to {len(neighbors) + 1} qubits."

            for i in range(
                len(neighbors) - channel_nqubits + 1
            ):  # Compose targets according to the channel's number of qubits
                target_neighbors += tuple(neighbors[i : i + channel_nqubits - 1])

        if kind == CommandKind.E:
            # TODO
            return []

        if kind == CommandKind.M:
            # TODO
            return []

        if kind == CommandKind.X:
            # TODO
            return []

        if kind == CommandKind.Z:
            # TODO
            return []

        if kind == CommandKind.C or kind == CommandKind.T:
            return [(KrausChannel([]), [])]

        typing_extensions.assert_never(kind)

    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""
        if self.rng.uniform() < self.measure_error_prob:
            return not result
        else:
            return result
