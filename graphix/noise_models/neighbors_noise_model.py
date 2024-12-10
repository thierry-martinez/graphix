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
    The Krauss channel that is applied to the targetted nodes should be specified when initializing the NeighborsNoiseModel.
    """

    def __init__(
        self,
        one_qubit_channel: function,
        two_qubits_channel: function,
        prepare_error_prob: float = 0.0,
        x_error_prob: float = 0.0,
        z_error_prob: float = 0.0,
        entanglement_error_prob: float = 0.0,
        measure_channel_prob: float = 0.0,
        measure_error_prob: float = 0.0,
        rng: Generator = None,
    ) -> None:
        self._state_graph: nx.Graph = nx.Graph()
        self.one_qubit_channel = one_qubit_channel
        self.two_qubit_channel = two_qubits_channel
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob
        self.rng = ensure_rng(rng)

    def _get_neighbors(self, nodes: list[int]) -> list[int]:
        """Return the neighbors of multiple nodes"""
        return [neighbor for n in nodes for neighbor in self._state_graph.neighbors(n)]

    def input_nodes(self, nodes: list[int]) -> Noise:
        """Return the noise to apply to input nodes and their neighbor nodes.

        We assume that the noise is applied to every neighbors,
        regardless the fact that one node can be the neighbor of
        two other nodes.
        The noise will be applied as many time as it the node is
        the neighbor of other nodes.
        """
        self._state_graph.add_nodes_from(nodes)  # Add initial nodes into the graph
        return [(self.one_qubit_channel(self.prepare_error_prob), [node]) for node in nodes] + [
            (self.one_qubit_channel(self.prepare_error_prob), [neighbor]) for neighbor in self._get_neighbors(nodes)
        ]

    def command(self, cmd: Command) -> Noise:
        """Return the noise to apply to the command `cmd`.

        N: adds a node to the state graph.
        E: adds an edge to the state graph.
        M: removes a node from the state graph.
        """
        kind = cmd.kind
        if kind == CommandKind.N:
            self._state_graph.add_node(cmd.node)
            return [(self.one_qubit_channel(self.prepare_error_prob), [cmd.node])] + [
                (self.one_qubit_channel(self.prepare_error_prob), [neighbor])
                for neighbor in self._get_neighbors([cmd.node])
            ]

        if kind == CommandKind.E:
            self._state_graph.add_edge(*cmd.nodes)
            return (
                [(self.two_qubit_channel(self.entanglement_error_prob), list(cmd.nodes))]
                + [
                    (self.two_qubit_channel(self.entanglement_error_prob), [cmd.nodes[0], neighbor])
                    for neighbor in self._get_neighbors([cmd.nodes[0]])
                ]
                + [
                    (self.two_qubit_channel(self.entanglement_error_prob), [cmd.nodes[1], neighbor])
                    for neighbor in self._get_neighbors([cmd.nodes[1]])
                ]
            )

        if kind == CommandKind.M:
            noise = [(self.one_qubit_channel(self.measure_channel_prob), [cmd.node])] + [
                (self.one_qubit_channel(self.measure_channel_prob), [neighbor])
                for neighbor in self._get_neighbors([cmd.node])
            ]
            self._state_graph.remove_node(cmd.node)
            return noise

        if kind == CommandKind.X:
            return [(self.one_qubit_channel(self.x_error_prob), [cmd.node])] + [
                (self.one_qubit_channel(self.x_error_prob), [neighbor]) for neighbor in self._get_neighbors([cmd.node])
            ]

        if kind == CommandKind.Z:
            return [(self.one_qubit_channel(self.z_error_prob), [cmd.node])] + [
                (self.one_qubit_channel(self.z_error_prob), [neighbor]) for neighbor in self._get_neighbors([cmd.node])
            ]

        if kind == CommandKind.C or kind == CommandKind.T:
            return [(KrausChannel([]), [])]

        typing_extensions.assert_never(kind)

    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""
        if self.rng.uniform() < self.measure_error_prob:
            return not result
        else:
            return result
