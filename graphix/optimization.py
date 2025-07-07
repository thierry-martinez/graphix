"""Optimization procedures for patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix import Pattern, command
from graphix.clifford import Clifford
from graphix.command import CommandKind

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet


def _incorporate_pauli_results_in_domain(
    results: Mapping[int, int], domain: AbstractSet[int]
) -> tuple[bool, set[int]] | None:
    if not (results.keys() & domain):
        return None
    new_domain = set(domain - results.keys())
    odd_outcome = sum(outcome for node, outcome in results.items() if node in domain) % 2
    return odd_outcome == 1, new_domain


def incorporate_pauli_results(pattern: Pattern) -> Pattern:
    """Return an equivalent pattern where results from Pauli presimulation are integrated in corrections."""
    result = Pattern(input_nodes=pattern.input_nodes)
    for cmd in pattern:
        if cmd.kind == CommandKind.M:
            s = _incorporate_pauli_results_in_domain(pattern.results, cmd.s_domain)
            t = _incorporate_pauli_results_in_domain(pattern.results, cmd.t_domain)
            if s or t:
                if s:
                    apply_x, new_s_domain = s
                else:
                    apply_x = False
                    new_s_domain = cmd.s_domain
                if t:
                    apply_z, new_t_domain = t
                else:
                    apply_z = False
                    new_t_domain = cmd.t_domain
                new_cmd = command.M(cmd.node, cmd.plane, cmd.angle, new_s_domain, new_t_domain)
                if apply_x:
                    new_cmd = new_cmd.clifford(Clifford.X)
                if apply_z:
                    new_cmd = new_cmd.clifford(Clifford.Z)
                result.add(new_cmd)
            else:
                result.add(cmd)
        # Use == for mypy
        elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
            signal = _incorporate_pauli_results_in_domain(pattern.results, cmd.domain)
            if signal:
                apply_c, new_domain = signal
                if new_domain:
                    cmd_cstr = command.X if cmd.kind == CommandKind.X else command.Z
                    result.add(cmd_cstr(cmd.node, new_domain))
                if apply_c:
                    c = Clifford.X if cmd.kind == CommandKind.X else Clifford.Z
                    result.add(command.C(cmd.node, c))
            else:
                result.add(cmd)
        else:
            result.add(cmd)
    result.reorder_output_nodes(pattern.output_nodes)
    return result
