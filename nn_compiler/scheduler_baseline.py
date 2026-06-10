"""
scheduler_baseline.py -- Reference scheduling heuristics for the NN Compiler.

Provides deterministic baseline schedulers whose cycle counts serve as
competitive targets and reward normalizers for the learned policy.
"""

from typing import Dict, List, Tuple
from .compiler_env import ComputeGraph, NodeType


def critical_path_height(graph: ComputeGraph) -> Dict[str, int]:
    """
    Compute the critical path height for every node in the graph.

    Height = longest path from this node to any output node
    (measured in number of downstream operations).

    Used by the Critical Path First (CPF) scheduler to prioritize
    nodes with the deepest dependency chains.
    """
    topo = graph.get_topological_order()

    # Build forward adjacency: parent -> [(child, slot)]
    children: Dict[str, List[Tuple[str, int]]] = {nid: [] for nid in graph.nodes}
    for child_id, child_node in graph.nodes.items():
        for parent_id, slot in child_node.dependencies:
            if parent_id in children:
                children[parent_id].append((child_id, slot))

    # Compute heights in reverse topological order
    height: Dict[str, int] = {}

    # Output nodes have height 0
    for out_id in graph.outputs:
        height[out_id] = 0

    for node_id in reversed(topo):
        if node_id in height:
            continue
        max_child = 0
        for child_id, _ in children.get(node_id, []):
            max_child = max(max_child, height.get(child_id, 0) + 1)
        height[node_id] = max_child

    # Inputs: height = 1 + max child height
    for in_id in graph.inputs:
        if in_id not in height:
            max_child = 0
            for child_id, _ in children.get(in_id, []):
                max_child = max(max_child, height.get(child_id, 0) + 1)
            height[in_id] = max_child

    return height


def schedule_cpf(graph: ComputeGraph) -> List[str]:
    """
    Critical Path First scheduling heuristic.

    At each step, from the set of ready nodes (all deps satisfied),
    pick the one with the highest critical path height. Break ties
    by node name for determinism.

    Returns the schedule order (list of op node IDs).
    """
    topo = graph.get_topological_order()

    # Build dependency tracking.
    # Only count dependencies on other OP nodes (not Input nodes),
    # since Input nodes are always available and never need scheduling.
    op_set = set(topo)
    remaining_deps: Dict[str, int] = {}
    switch_to_fpmul: Dict[str, str] = {}  # Switch -> fp_mul implicit deps
    for node_id in topo:
        deps = graph.nodes[node_id].dependencies
        remaining_deps[node_id] = sum(1 for dep_id, _ in deps if dep_id in op_set)

    # Add implicit dependencies: fp_mul nodes depend on their Switch
    # via send_false routing, which isn't in the graph's dependency edges.
    for node_id in topo:
        node = graph.nodes[node_id]
        if node.type == "Switch":
            for child_id, child_node in graph.nodes.items():
                if child_node.type == "Merge":
                    has_switch = any(d == node_id for d, _ in child_node.dependencies)
                    if has_switch:
                        for other, _ in child_node.dependencies:
                            if other != node_id and other.startswith("fp_mul"):
                                if other in remaining_deps:
                                    remaining_deps[other] += 1
                                    switch_to_fpmul[node_id] = other

    # Also track deps on Input nodes for ready-set calculation
    # (Input nodes are always "ready" but don't appear in topo)
    height = critical_path_height(graph)
    executed: set = set()
    schedule: List[str] = []

    while len(schedule) < len(topo):
        # Find ready nodes: all deps satisfied, not yet executed
        ready = []
        for node_id in topo:
            if node_id in executed:
                continue
            if remaining_deps[node_id] == 0:
                ready.append(node_id)

        if not ready:
            # Deadlock -- shouldn't happen for valid DAGs
            break

        # Pick the ready node with highest critical path height
        # Break ties by node name (deterministic)
        best = max(ready, key=lambda nid: (height.get(nid, 0), nid))

        schedule.append(best)
        executed.add(best)

        # Free dependents of the scheduled node (including implicit deps)
        for child_id, child_node in graph.nodes.items():
            for parent_id, _ in child_node.dependencies:
                if parent_id == best and child_id in remaining_deps:
                    remaining_deps[child_id] -= 1
        # Free implicit fp_mul dependency when Switch is scheduled
        if best in switch_to_fpmul:
            fpmul = switch_to_fpmul[best]
            if fpmul in remaining_deps:
                remaining_deps[fpmul] -= 1

    return schedule


def schedule_topological(graph: ComputeGraph) -> List[str]:
    """
    Baseline: emit nodes in topological order.
    Returns the graph's topological order as-is.
    """
    return graph.get_topological_order()
