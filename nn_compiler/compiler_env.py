"""
compiler_env.py -- Graph representation, VM bridge, Gym environment, and
baseline targets for the NN Compiler v0.

Supports conditional routing (send_true / send_false) for Switch nodes.
"""

import os
import re
import math
import random
import subprocess
from typing import List, Dict, Tuple, Optional, Any


# ═══════════════════════════════════════════════════════════════
#  Step 1: Computation Graph Representation
# ═══════════════════════════════════════════════════════════════

class NodeType:
    INPUT = "Input"
    MUL = "Mul"
    ADD = "Add"
    SUB = "Sub"
    DIV = "Div"
    SWITCH = "Switch"
    CMPGEZ = "CmpGeZ"
    MERGE = "Merge"
    OUTPUT = "Output"

    @staticmethod
    def to_opcode(node_type: str) -> str:
        """Map a NodeType string to the DF-ASM opcode mnemonic."""
        mapping = {
            NodeType.MUL: "Mul",
            NodeType.ADD: "Add",
            NodeType.SUB: "Sub",
            NodeType.DIV: "Div",
            NodeType.SWITCH: "Switch",
            NodeType.CMPGEZ: "CmpGeZ",
            NodeType.MERGE: "Merge",
        }
        return mapping.get(node_type, node_type)


class ComputeNode:
    """A single node in the high-level mathematical computation graph."""

    def __init__(self, node_id: str, node_type: str):
        self.id = node_id
        self.type = node_type
        # List of (parent_node_id, target_operand_slot)
        self.dependencies: List[Tuple[str, int]] = []

    def __repr__(self):
        return f"ComputeNode({self.id}, {self.type}, deps={self.dependencies})"


class ComputeGraph:
    """Represents the high-level mathematical intent as a directed graph."""

    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []

    def add_node(self, node_id: str, node_type: str) -> ComputeNode:
        node = ComputeNode(node_id, node_type)
        self.nodes[node_id] = node
        if node_type == NodeType.INPUT:
            self.inputs.append(node_id)
        elif node_type == NodeType.OUTPUT:
            self.outputs.append(node_id)
        return node

    def add_edge(self, from_node: str, to_node: str, slot: int):
        """Wire a data dependency: from_node's result feeds into to_node at operand `slot`."""
        if to_node in self.nodes:
            self.nodes[to_node].dependencies.append((from_node, slot))

    def get_topological_order(self) -> List[str]:
        """
        Returns operation nodes (skipping Input/Output) in dependency order,
        computed via DFS from each output node.
        """
        visited = set()
        order = []

        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node is None:
                return
            for dep_id, _ in node.dependencies:
                dfs(dep_id)
            if node.type not in (NodeType.INPUT, NodeType.OUTPUT):
                order.append(node_id)

        for out_id in self.outputs:
            dfs(out_id)

        return order


# ═══════════════════════════════════════════════════════════════
#  Routing Types
# ═══════════════════════════════════════════════════════════════

class RoutingType:
    ALWAYS = "send"
    IF_TRUE = "send_true"
    IF_FALSE = "send_false"

    @staticmethod
    def from_int(val: int) -> str:
        return [RoutingType.ALWAYS, RoutingType.IF_TRUE, RoutingType.IF_FALSE][val]

    @staticmethod
    def to_int(val: str) -> int:
        mapping = {RoutingType.ALWAYS: 0, RoutingType.IF_TRUE: 1, RoutingType.IF_FALSE: 2}
        return mapping.get(val, 0)


# ═══════════════════════════════════════════════════════════════
#  Step 2: VM Bridge (Python <-> Rust release binary)
# ═══════════════════════════════════════════════════════════════

class VMBridge:
    """
    Writes .dfasm files to disk, invokes the Rust VM binary via subprocess,
    and parses the METRICS and OUTPUTS lines from stdout.

    Expected VM output format:
      OUTPUTS: name=value name=value ...
      METRICS: CYCLES=14, MAX_QUEUE=3
    """

    def __init__(self, vm_binary_path: str):
        self.vm_binary_path = vm_binary_path

    def run_program(
        self, dfasm_content: str, inputs: Dict[str, float],
        expected_outputs: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict]:
        """
        Compile and execute a .dfasm program.

        Args:
            dfasm_content: The .dfasm source to run.
            inputs: Dict of input name -> value.
            expected_outputs: If given, validates output values match.
                              Mismatch triggers the -1000 failure penalty.

        Returns:
            (reward, telemetry_dict)
            reward = -(cycles + 0.1 * max_queue) on success, -1000 on failure.
        """
        temp_filename = "runtime_eval.dfasm"
        with open(temp_filename, "w", encoding="utf-8") as f:
            f.write(dfasm_content)

        cmd = [self.vm_binary_path, temp_filename]
        for name, value in inputs.items():
            cmd.extend(["--input", f"{name}={value}"])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=2.0
            )

            if result.returncode != 0:
                return -1000.0, {
                    "error": f"VM exited {result.returncode}: {result.stderr.strip()}"
                }

            stdout = result.stdout

            # ── Parse output values ───────────────────────
            outputs_match = re.search(r"OUTPUTS:\s+(.+)$", stdout, re.MULTILINE)
            actual_outputs: Dict[str, float] = {}
            if outputs_match:
                for pair in outputs_match.group(1).split():
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        actual_outputs[k] = float(v)

            # ── Correctness validation ────────────────────
            if expected_outputs is not None:
                for name, expected_val in expected_outputs.items():
                    actual_val = actual_outputs.get(name)
                    if actual_val is None:
                        return -1000.0, {
                            "error": f"Missing output '{name}'.",
                            "actual_outputs": actual_outputs,
                        }
                    # Check for NaN explicitly (NaN comparison always False in IEEE 754)
                    # Use relative tolerance: accept 1e-4 absolute OR 1e-5 relative
                    # error. The matching store computes in a different order than
                    # the evaluator, so floating-point accumulation can differ
                    # for large intermediate values.
                    rel_err = abs(actual_val - expected_val) / max(1.0, abs(expected_val))
                    if math.isnan(actual_val) or (
                        abs(actual_val - expected_val) > 1e-4
                        and rel_err > 1e-5
                    ):
                        return -1000.0, {
                            "error": (
                                f"Output '{name}' mismatch: "
                                f"expected {expected_val}, got {actual_val}"
                            ),
                            "actual_outputs": actual_outputs,
                            "expected_outputs": expected_outputs,
                        }

            # ── Parse metrics ─────────────────────────────
            cycles_match = re.search(r"CYCLES=(\d+)", stdout)
            queue_match = re.search(r"MAX_QUEUE=(\d+)", stdout)

            if not cycles_match or not queue_match:
                return -1000.0, {
                    "error": "Failed to parse METRICS from VM output.",
                    "stdout": stdout[:200],
                }

            cycles = int(cycles_match.group(1))
            max_queue = int(queue_match.group(1))

            reward = -(cycles + 0.1 * max_queue)
            info: Dict = {"cycles": cycles, "max_queue": max_queue}
            if actual_outputs:
                info["outputs"] = actual_outputs
            return reward, info

        except subprocess.TimeoutExpired:
            return -1000.0, {"error": "VM execution timed out (deadlock/livelock)."}
        finally:
            # Windows file-lock race: the subprocess may still be releasing
            # the file handle when we try to delete. Suppress the error.
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except OSError:
                pass

    def batch_evaluate(
        self, dfasm_content: str,
        test_suite: List[Tuple[Dict[str, float], Dict[str, float]]],
    ) -> Tuple[float, Dict]:
        """
        Evaluate a single DF-ASM program against a suite of test cases.

        Args:
            dfasm_content: The .dfasm source to run.
            test_suite: List of (input_dict, expected_output_dict) pairs.
                        At minimum one pair must be provided.

        Returns:
            (worst_reward, aggregate_info)
            Uses worst-case (minimum) aggregation across all test cases.
            If ANY test case fails correctness validation or deadlocks,
            the entire batch scores -1000.
        """
        worst_reward = float("inf")
        aggregate_info: Dict = {
            "cases": [],
            "cycles": None,
            "max_queue": None,
        }

        for i, (inputs, expected_outputs) in enumerate(test_suite):
            reward, info = self.run_program(
                dfasm_content, inputs, expected_outputs
            )

            case_result = {
                "index": i,
                "inputs": inputs,
                "reward": reward,
                "info": info,
            }
            aggregate_info["cases"].append(case_result)

            if reward < worst_reward:
                worst_reward = reward

        # Pick the metrics from the worst-performing case for summary
        worst_idx = min(
            range(len(test_suite)),
            key=lambda i: aggregate_info["cases"][i]["reward"],
        )
        worst_case = aggregate_info["cases"][worst_idx]
        aggregate_info["cycles"] = worst_case["info"].get("cycles")
        aggregate_info["max_queue"] = worst_case["info"].get("max_queue")
        aggregate_info["worst_case_idx"] = worst_idx

        if worst_reward == float("inf"):
            worst_reward = -1000.0

        return worst_reward, aggregate_info


# ═══════════════════════════════════════════════════════════════
#  Step 3: Gym-Style RL Environment
# ═══════════════════════════════════════════════════════════════

class DataflowGymEnv:
    """
    A Gym-compatible environment where the agent incrementally builds a
    DF-ASM program by making routing decisions for each operation node
    in topological order.

    Each routing entry is: (target_name, slot, routing_type)
      where routing_type is "send", "send_true", or "send_false"

    The reward is computed via batch evaluation: the compiled DF-ASM is
    run against a test suite of multiple inputs, and the worst-case
    (minimum) reward across all test cases is returned. This prevents
    the agent from spec-gaming on a single input.
    """

    def __init__(
        self,
        graph: ComputeGraph,
        vm_bridge: VMBridge,
        test_suite: List[Tuple[Dict[str, float], Dict[str, float]]],
    ):
        """
        Args:
            test_suite: List of (input_dict, expected_output_dict) pairs.
                        The agent's compiled program must pass ALL test cases
                        with correct output values to receive a non-failure reward.
        """
        self.graph = graph
        self.vm_bridge = vm_bridge
        self.test_suite = test_suite
        self.topo_order = graph.get_topological_order()
        self.reset()

    def reset(self) -> List[str]:
        """Reset the environment. Returns the initial state."""
        self.current_node_idx = 0
        self.compiled_nodes: Dict[str, Dict] = {}
        # (target_name, slot, routing_type) triples
        self.forward_routing: Dict[str, List[Tuple[str, int, str]]] = {
            node_id: [] for node_id in self.graph.nodes
        }
        return self._get_state()

    def _get_state(self) -> List[str]:
        if self.current_node_idx < len(self.topo_order):
            return [self.topo_order[self.current_node_idx]]
        return []

    def step(
        self, action: Dict
    ) -> Tuple[Optional[List[str]], float, bool, Dict]:
        """
        Apply routing actions for the current node.

        Action format:
          {'target_destinations': [(target, slot, routing_type), ...]}
          where routing_type is 'send', 'send_true', or 'send_false'
        """
        if self.current_node_idx >= len(self.topo_order):
            return None, -1000.0, True, {"error": "Step past end of compilation."}

        current_id = self.topo_order[self.current_node_idx]
        compute_node = self.graph.nodes[current_id]

        # Register routing decisions
        for entry in action.get("target_destinations", []):
            if len(entry) == 3:
                # (target, slot, routing_type)
                target, slot, rt = entry
                self.forward_routing[current_id].append((target, slot, rt))
            elif len(entry) == 2:
                # Legacy: (target, slot) -> default to Always
                target, slot = entry
                self.forward_routing[current_id].append(
                    (target, slot, RoutingType.ALWAYS)
                )

        self.compiled_nodes[current_id] = {
            "id": current_id,
            "opcode": NodeType.to_opcode(compute_node.type),
            "dependencies": compute_node.dependencies,
        }

        self.current_node_idx += 1
        done = self.current_node_idx == len(self.topo_order)

        reward = 0.0
        info = {}

        if done:
            # Auto-wire outputs (use Always routing, dedup by target name)
            for out_id in self.graph.outputs:
                out_node = self.graph.nodes[out_id]
                for src_id, src_slot in out_node.dependencies:
                    already_wired = any(
                        t == out_id for t, _, _ in self.forward_routing[src_id]
                    )
                    if not already_wired:
                        self.forward_routing[src_id].append(
                            (out_id, src_slot, RoutingType.ALWAYS)
                        )

            dfasm_program = self._generate_dfasm()
            reward, info = self.vm_bridge.batch_evaluate(
                dfasm_program, self.test_suite
            )

        return self._get_state(), reward, done, info

    def _generate_dfasm(self) -> str:
        """Serialize the compiled node graph into a .dfasm S-expression string.

        For non-Switch nodes, send directives for outgoing data dependencies
        are auto-generated (using Always routing). This ensures correct data
        flow without the agent having to manually route every edge.

        For Switch nodes, the agent must provide ALL outgoing routing
        explicitly (using send_true / send_false) since the Switch's
        routing is conditional.
        """
        lines = []

        inputs_str = " ".join(self.graph.inputs)
        lines.append(f"(input {inputs_str})")
        lines.append("")

        # Build map: node_id -> [(child_name, slot)] of outgoing data edges
        outgoing_edges: Dict[str, List[Tuple[str, int]]] = {}
        for child_name, child_node in self.graph.nodes.items():
            for parent_id, slot in child_node.dependencies:
                if parent_id not in outgoing_edges:
                    outgoing_edges[parent_id] = []
                outgoing_edges[parent_id].append((child_name, slot))

        for node_id in self.topo_order:
            meta = self.compiled_nodes[node_id]
            compute_node = self.graph.nodes[node_id]
            opcode = meta["opcode"]
            is_switch = compute_node.type == "Switch"

            lines.append(f"(node {node_id} {opcode}")

            # Wait clauses from data dependencies
            for parent_id, slot in meta["dependencies"]:
                lines.append(f"    (wait {parent_id} {slot})")

            # Auto-generate sends for non-Switch nodes' outgoing data edges.
            # Skip edges already covered by forward_routing (e.g. output
            # auto-wiring) to avoid duplicate send directives.
            if not is_switch:
                fwd_entries = set(
                    (t, s) for t, s, _ in self.forward_routing[node_id]
                )
                for child_name, slot in outgoing_edges.get(node_id, []):
                    if (child_name, slot) not in fwd_entries:
                        lines.append(f"    (send {child_name} {slot})")

            # Agent's routing decisions (extra / conditional sends)
            for target_id, slot, rt in self.forward_routing[node_id]:
                lines.append(f"    ({rt} {target_id} {slot})")

            lines.append(")")
            lines.append("")

        outputs_str = " ".join(self.graph.outputs)
        lines.append(f"(output {outputs_str})")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Step 4: Scheduling Gym Environment
# ═══════════════════════════════════════════════════════════════

# Instruction latencies (cycles) for resource-constrained simulation.
DEFAULT_LATENCY = {
    NodeType.ADD: 1,
    NodeType.MUL: 3,
    NodeType.SUB: 1,
    NodeType.DIV: 3,
    NodeType.CMPGEZ: 1,
    NodeType.SWITCH: 1,
    NodeType.MERGE: 1,
}


class SchedulingGymEnv:
    """
    A Gym-compatible environment where the agent schedules nodes by
    priority: at each step it picks the next ready node to emit.

    Unlike DataflowGymEnv (which does routing), this env only requires
    the agent to choose WHICH ready node to emit next. All send
    directives are auto-generated from data dependencies.

    Reward = -(cycles + 0.1 * max_queue) from batch evaluation.

    Args:
        max_exec_units: If None (default), use the VM's unlimited dataflow
            model for cycle counts. If set (e.g., 1), use a Python-side
            single-issue simulator where only K instructions can issue
            per cycle. The VM still validates output correctness.
        latency: Dict mapping node type -> execution cycles. Used only
            when max_exec_units is set.
    """

    def __init__(
        self,
        graph: ComputeGraph,
        vm_bridge: VMBridge,
        test_suite: List[Tuple[Dict[str, float], Dict[str, float]]],
        max_exec_units: Optional[int] = None,
        latency: Optional[Dict[str, int]] = None,
        max_registers: Optional[int] = None,
        register_penalty_alpha: float = 0.0,
        unit_limit: Optional[Dict[str, int]] = None,
        latency_distribution: Optional[Dict[str, Tuple[int, int]]] = None,
        mem_latency: int = 10,
    ):
        self.graph = graph
        self.vm_bridge = vm_bridge
        self.test_suite = test_suite
        self.topo_order = graph.get_topological_order()
        self.max_exec_units = max_exec_units
        self.latency = latency if latency is not None else DEFAULT_LATENCY
        self.max_registers = max_registers
        self.register_penalty_alpha = register_penalty_alpha
        self.unit_limit = unit_limit if unit_limit is not None else {}
        self.latency_distribution = latency_distribution if latency_distribution is not None else {}
        self.mem_latency = mem_latency
        self.reset()

    def reset(self):
        """Reset the environment."""
        self.executed: set = set()
        self.schedule: List[Tuple[str, str]] = []  # (action_type, node_id)
        self.stack_pool: Dict[str, int] = {}  # node_id -> spill_cycle
        self.spilled_nodes: set = set()
        self.reload_in_progress: set = set()
        # Track remaining dependency count for each op node.
        # Only count dependencies on other op nodes (not Input nodes),
        # since Input nodes are always available.
        op_set = set(self.topo_order)
        self.remaining_deps: Dict[str, int] = {}
        for node_id in self.topo_order:
            deps = self.graph.nodes[node_id].dependencies
            self.remaining_deps[node_id] = sum(
                1 for dep_id, _ in deps if dep_id in op_set
            )
        # Add implicit dependencies: fp_mul nodes depend on their Switch
        # via send_false routing, which isn't in the graph's dependency edges.
        self.switch_to_fpmul: Dict[str, str] = {}
        for node_id in self.topo_order:
            node = self.graph.nodes[node_id]
            if node.type == NodeType.SWITCH:
                for child_id, child_node in self.graph.nodes.items():
                    if child_node.type == NodeType.MERGE:
                        has_switch = any(
                            d == node_id for d, _ in child_node.dependencies
                        )
                        if has_switch:
                            for other, _ in child_node.dependencies:
                                if other != node_id and other.startswith("fp_mul"):
                                    if other in self.remaining_deps:
                                        self.remaining_deps[other] += 1
                                        self.switch_to_fpmul[node_id] = other

        # Register pressure tracking (for observation feature)
        if self.max_registers is not None:
            self.node_consumers: Dict[str, int] = {}
            for nid in self.topo_order:
                count = 0
                for child_id, child_node in self.graph.nodes.items():
                    if child_id in self.topo_order:
                        for parent_id, _ in child_node.dependencies:
                            if parent_id == nid:
                                count += 1
                self.node_consumers[nid] = count
            # Implicit fp_mul consumer counts
            for nid in self.topo_order:
                node = self.graph.nodes[nid]
                if node.type == NodeType.SWITCH:
                    for child_id, child_node in self.graph.nodes.items():
                        if child_node.type == NodeType.MERGE:
                            has = any(d == nid for d, _ in child_node.dependencies)
                            if has:
                                for other, _ in child_node.dependencies:
                                    if other != nid and other.startswith("fp_mul"):
                                        if other in self.node_consumers:
                                            self.node_consumers[other] += 1
            self.outstanding: set = set()
            self.max_outstanding: int = 0
        else:
            self.node_consumers = {}
            self.outstanding = set()
            self.max_outstanding = 0
        self._sim_deadlock_warning = None

        return self._ready_set()

    def _ready_set(self) -> List[str]:
        """Return the set of nodes ready to fire (all deps satisfied).

        A node is NOT ready if any of its dependencies are spilled (their
        output is in the stack and must be reloaded first). This prevents
        the model from issuing consumers before spilt values are restored.
        """
        ready = []
        for node_id in self.topo_order:
            if node_id not in self.executed and self.remaining_deps[node_id] == 0:
                node = self.graph.nodes[node_id]
                blocked = any(
                    dep_id in self.spilled_nodes
                    for dep_id, _ in node.dependencies
                )
                if not blocked:
                    ready.append(node_id)
        return ready

    @property
    def done(self) -> bool:
        return len(self.executed) == len(self.topo_order)

    def step(self, action_idx: int) -> Tuple[List[str], float, bool, Dict]:
        """
        Schedule one action in the flat 2N action space.

        Args:
            action_idx: 0..N-1 = Issue (if node unissued+ready) or
                        Reload (if node spilled); N..2N-1 = Spill
                        (if node's output is live in a register).

        Returns:
            ready_set: List of node IDs now ready (for next step).
            reward: 0.0 if not done; terminal reward when done.
            done: True if all nodes scheduled AND no spills/reloads pending.
            info: Dict with metrics (only populated when done).
        """
        n = len(self.topo_order)

        if action_idx >= n:
            # ── Spill action (N..2N-1) ─────────────────────────
            sp_idx = action_idx - n
            if sp_idx >= n:
                return self._ready_set(), -1000.0, True, {
                    "error": f"Spill action {action_idx} out of range [N, 2N)."
                }
            node_id = self.topo_order[sp_idx]
            if node_id not in self.outstanding or node_id in self.spilled_nodes:
                return self._ready_set(), -1000.0, True, {
                    "error": f"Cannot spill '{node_id}': output not live in register."
                }
            self.schedule.append(("spill", node_id))
            self.outstanding.discard(node_id)
            self.spilled_nodes.add(node_id)
            self.stack_pool[node_id] = len(self.schedule)

        elif self.topo_order[action_idx] in self.spilled_nodes:
            # ── Reload action (0..N-1, node is spilled) ────────
            node_id = self.topo_order[action_idx]
            if node_id in self.reload_in_progress:
                return self._ready_set(), -1000.0, True, {
                    "error": f"Node '{node_id}' is already being reloaded."
                }
            self.schedule.append(("reload", node_id))
            self.spilled_nodes.discard(node_id)
            self.reload_in_progress.add(node_id)
            self.outstanding.add(node_id)

        else:
            # ── Issue action (0..N-1, node is unissued+ready) ──
            node_id = self.topo_order[action_idx]
            if node_id in self.executed or node_id not in self._ready_set():
                return self._ready_set(), -1000.0, True, {
                    "error": f"Node '{node_id}' is not ready for issue."
                }

            self.executed.add(node_id)
            self.schedule.append(("issue", node_id))

            # Free dependents
            for child_id, child_node in self.graph.nodes.items():
                for parent_id, _ in child_node.dependencies:
                    if parent_id == node_id and child_id in self.remaining_deps:
                        self.remaining_deps[child_id] -= 1
            # Free implicit fp_mul dependency when Switch is scheduled
            if node_id in self.switch_to_fpmul:
                fpmul = self.switch_to_fpmul[node_id]
                if fpmul in self.remaining_deps:
                    self.remaining_deps[fpmul] -= 1

            # Register pressure tracking (step-loop proxy)
            if self.max_registers is not None:
                node = self.graph.nodes[node_id]
                for dep_id, _ in node.dependencies:
                    if dep_id in self.outstanding and dep_id in self.node_consumers:
                        self.node_consumers[dep_id] -= 1
                        if self.node_consumers[dep_id] <= 0:
                            self.outstanding.discard(dep_id)
                            self.node_consumers.pop(dep_id, None)
                if node_id in self.switch_to_fpmul:
                    fpmul = self.switch_to_fpmul[node_id]
                    if fpmul in self.outstanding and fpmul in self.node_consumers:
                        self.node_consumers[fpmul] -= 1
                        if self.node_consumers[fpmul] <= 0:
                            self.outstanding.discard(fpmul)
                            self.node_consumers.pop(fpmul, None)
                if self.node_consumers.get(node_id, 0) > 0:
                    self.outstanding.add(node_id)
                self.max_outstanding = max(self.max_outstanding, len(self.outstanding))

        done = self.done
        reward = 0.0
        info: Dict = {}

        if done:
            if self.vm_bridge is not None and len(self.test_suite) > 0:
                dfasm = self._generate_dfasm()
                reward, info = self.vm_bridge.batch_evaluate(dfasm, self.test_suite)
                # Override cycles with simulated model when active            if self.max_exec_units is not None and reward > -900:
                sim_cycles = self._simulate_cycles()
                spill_penalty = 0
                if self.max_registers is not None and self.register_penalty_alpha > 0:
                    spill_total = sum(
                        max(0, n - self.max_registers)
                        for n in self._sim_live_regs_history
                    )
                    spill_penalty = self.register_penalty_alpha * spill_total
                reward = -(sim_cycles + 0.1 * info.get("max_queue", 0) + spill_penalty)
                info["cycles"] = sim_cycles
                info["cycles_source"] = "simulated"
                info["spill_penalty"] = spill_penalty
                info["struct_stalls"] = self._sim_struct_stalls
                if self._sim_deadlock_warning:
                    info["deadlock_warning"] = self._sim_deadlock_warning
                if self.max_registers is not None:
                    info["max_live_registers"] = self._sim_max_live
                    info["registers"] = self.max_registers
                    info["live_regs_history"] = self._sim_live_regs_history
                    info["spill_steps"] = sum(
                        1 for n in self._sim_live_regs_history if n > self.max_registers
                    )
            elif self.max_exec_units is not None:
                # Simulator-only mode: no VM, use cycle simulator directly
                sim_cycles = self._simulate_cycles()
                spill_penalty = 0
                if self.max_registers is not None and self.register_penalty_alpha > 0:
                    spill_total = sum(
                        max(0, n - self.max_registers)
                        for n in self._sim_live_regs_history
                    )
                    spill_penalty = self.register_penalty_alpha * spill_total
                info = {"cycles": sim_cycles, "max_queue": 0, "cycles_source": "simulated"}
                info["spill_penalty"] = spill_penalty
                info["struct_stalls"] = self._sim_struct_stalls
                if self._sim_deadlock_warning:
                    info["deadlock_warning"] = self._sim_deadlock_warning
                if self.max_registers is not None:
                    info["max_live_registers"] = self._sim_max_live
                    info["registers"] = self.max_registers
                    info["live_regs_history"] = self._sim_live_regs_history
                    info["spill_steps"] = sum(
                        1 for n in self._sim_live_regs_history if n > self.max_registers
                    )
                reward = -(sim_cycles + spill_penalty)
            else:
                # No VM and no simulator — should not happen in practice
                reward = -1000.0
                info = {"error": "No VM bridge and no simulator configured."}

        return self._ready_set(), reward, done, info

    def _simulate_cycles(self) -> int:
        """
        Simulate execution of the schedule (with spill/reload support).

        Model: the schedule is a heterogeneous list of actions:
          ("issue", nid):  issue computation node nid (uses existing OoO model)
          ("spill", nid):  store nid's output to stack (1-cycle latency)
          ("reload", nid): load nid's output from stack (mem_latency-cycle latency)

        At each cycle, up to max_exec_units issue entries are scanned and
        issued in OoO order. Spill/reload entries are processed as in-order
        serialization points (they don't participate in the OoO scan but
        fire as completions that affect the register file).

        Register effects:
          - Spill completes  -> nid removed from live_regs (register freed)
          - Reload completes -> nid added back to live_regs (register occupied)
        """
        # Deterministic seed per schedule (reproducible across process restarts).
        h = 0
        for action_type, nid in self.schedule:
            for ch in nid:
                h = (h * 127 + ord(ch)) & 0x7FFFFFFF
        random.seed(h)
        K = self.max_exec_units
        graph = self.graph
        lat = self.latency

        # Recompute remaining_deps (same logic as reset())
        op_set = set(self.topo_order)
        remaining: Dict[str, int] = {}
        for nid in self.topo_order:
            deps = graph.nodes[nid].dependencies
            remaining[nid] = sum(1 for d, _ in deps if d in op_set)

        # Implicit fp_mul deps
        switch_to_fpmul: Dict[str, str] = {}
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.type == "Switch":
                for child_id, child_node in graph.nodes.items():
                    if child_node.type == "Merge":
                        has = any(d == nid for d, _ in child_node.dependencies)
                        if has:
                            for other, _ in child_node.dependencies:
                                if other != nid and other.startswith("fp_mul"):
                                    if other in remaining:
                                        remaining[other] += 1
                                        switch_to_fpmul[nid] = other

        # Extract issue-only schedule for the OoO ready-scan
        issue_order: List[str] = [nid for action, nid in self.schedule if action == "issue"]
        total_issue = len(issue_order)

        # completion: issue entries only (computation results)
        completion: Dict[str, int] = {}
        # Spill/reload completions: cycle when store/load finishes
        spill_completions: Dict[str, int] = {}   # nid -> cycle
        reload_completions: Dict[str, int] = {}  # nid -> cycle

        in_flight_counts: Dict[str, int] = {}
        struct_stalls = 0

        cycle = 0
        freed: set = set()  # issue entries whose dependents have been freed
        MAX_CYCLES = 10_000

        # ── Register pressure tracking ────────────────────
        max_regs = self.max_registers
        if max_regs is not None:
            reg_consumers: Dict[str, int] = {}
            for nid in self.topo_order:
                count = 0
                for child_id, child_node in graph.nodes.items():
                    if child_id not in self.topo_order:
                        continue
                    for parent_id, _ in child_node.dependencies:
                        if parent_id == nid:
                            count += 1
                if count > 0:
                    reg_consumers[nid] = count
            for nid in self.topo_order:
                node = graph.nodes[nid]
                if node.type == "Switch":
                    for child_id, child_node in graph.nodes.items():
                        if child_node.type == "Merge":
                            has = any(d == nid for d, _ in child_node.dependencies)
                            if has:
                                for other, _ in child_node.dependencies:
                                    if other != nid and other.startswith("fp_mul"):
                                        if other in reg_consumers:
                                            reg_consumers[other] += 1
            live_regs: set = set()
            max_live = 0
            live_regs_history: List[int] = []
        else:
            reg_consumers = {}
            live_regs = set()
            max_live = 0
            live_regs_history: List[int] = []

        # Track schedule position for in-order spill/reload processing
        sched_idx = 0
        sched = self.schedule

        while (len(completion) < total_issue or sched_idx < len(sched)
               or spill_completions or reload_completions) and cycle < MAX_CYCLES:
            # Record register pressure at start of cycle
            if max_regs is not None:
                live_regs_history.append(len(live_regs))

            # Phase 1: Process pending spill/reload completions
            for nid, end_cycle in list(spill_completions.items()):
                if end_cycle <= cycle:
                    live_regs.discard(nid)
                    del spill_completions[nid]
            for nid, end_cycle in list(reload_completions.items()):
                if end_cycle <= cycle:
                    if nid in reg_consumers:
                        live_regs.add(nid)
                        max_live = max(max_live, len(live_regs))
                    del reload_completions[nid]

            # Phase 2: Issue up to K items (process schedule entries in order)
            issued = 0
            while issued < K and sched_idx < len(sched):
                action_type, nid = sched[sched_idx]
                sched_idx += 1

                if action_type == "issue":
                    # Standard OoO issue: check if nid is ready
                    if remaining.get(nid, -1) != 0:
                        # Not ready — stall on this entry. Re-queue for next cycle.
                        sched_idx -= 1
                        break

                    ntype = graph.nodes[nid].type
                    # Port constraint
                    if ntype in self.unit_limit:
                        if in_flight_counts.get(ntype, 0) >= self.unit_limit[ntype]:
                            sched_idx -= 1
                            struct_stalls += 1
                            break
                    # Register pressure
                    if max_regs is not None and len(live_regs) > max_regs:
                        would_free = False
                        for dep_id, _ in graph.nodes[nid].dependencies:
                            if reg_consumers.get(dep_id, 0) == 1:
                                would_free = True
                                break
                        if nid in switch_to_fpmul:
                            fpmul = switch_to_fpmul[nid]
                            if reg_consumers.get(fpmul, 0) == 1:
                                would_free = True
                        if not would_free:
                            sched_idx -= 1
                            break

                    # Issue: schedule completion
                    if ntype in self.latency_distribution:
                        min_l, max_l = self.latency_distribution[ntype]
                        l = random.randint(min_l, max_l)
                    else:
                        l = lat.get(ntype, 1)
                    completion[nid] = cycle + l
                    issued += 1
                    in_flight_counts[ntype] = in_flight_counts.get(ntype, 0) + 1

                    # Decrement consumer counts
                    if max_regs is not None:
                        for dep_id, _ in graph.nodes[nid].dependencies:
                            if dep_id in reg_consumers:
                                reg_consumers[dep_id] -= 1
                                if reg_consumers[dep_id] <= 0:
                                    live_regs.discard(dep_id)
                                    reg_consumers.pop(dep_id, None)
                        if nid in switch_to_fpmul:
                            fpmul = switch_to_fpmul[nid]
                            if fpmul in reg_consumers:
                                reg_consumers[fpmul] -= 1
                                if reg_consumers[fpmul] <= 0:
                                    live_regs.discard(fpmul)
                                    reg_consumers.pop(fpmul, None)

                elif action_type == "spill":
                    # 1-cycle store latency
                    spill_completions[nid] = cycle + 1
                    issued += 1  # counts as an issued slot (uses memory port)

                elif action_type == "reload":
                    # mem_latency-cycle load latency
                    reload_completions[nid] = cycle + self.mem_latency
                    issued += 1  # counts as an issued slot

            # Advance clock
            cycle += 1

            # Free dependents of issue entries that completed
            for nid in list(completion.keys()):
                if nid not in freed and completion[nid] <= cycle:
                    freed.add(nid)
                    completed_type = graph.nodes[nid].type
                    if completed_type in self.unit_limit:
                        in_flight_counts[completed_type] = max(
                            0, in_flight_counts.get(completed_type, 0) - 1
                        )
                    for child_id, child_node in graph.nodes.items():
                        for parent_id, _ in child_node.dependencies:
                            if parent_id == nid and child_id in remaining:
                                remaining[child_id] -= 1
                    if nid in switch_to_fpmul:
                        fpmul = switch_to_fpmul[nid]
                        if fpmul in remaining:
                            remaining[fpmul] -= 1
                    if max_regs is not None and nid in reg_consumers:
                        live_regs.add(nid)
                        max_live = max(max_live, len(live_regs))

        # Deadlock/stall diagnostic
        if len(completion) < total_issue:
            stranded = [nid for nid in issue_order if nid not in completion]
            stranded_types = [f"{n}({graph.nodes[n].type})" for n in stranded]
            self._sim_deadlock_warning = (
                f"MAX_CYCLES reached: {total_issue - len(completion)}/{total_issue} "
                f"ops stranded {stranded_types}, live_regs={len(live_regs)}, "
                f"struct_stalls={struct_stalls}"
            )
        else:
            self._sim_deadlock_warning = None

        self._sim_max_live = max_live
        self._sim_live_regs_history = live_regs_history
        self._sim_struct_stalls = struct_stalls
        return max(completion.values()) if completion else MAX_CYCLES

    def _generate_dfasm(self) -> str:
        """
        Serialize the graph into a DF-ASM program in schedule order.

        Only "issue" actions generate DF-ASM nodes; spill/reload are
        meta-operations that affect the register file but don't appear
        in the final program.

        All send directives are auto-generated from data dependencies.
        For Switch nodes, send_true / send_false are generated based
        on the graph topology:
          - Send IfTrue to the Merge that depends on this Switch (slot 0)
          - Send IfFalse to the fp_mul node on the false path (slot 0)
        """
        lines = []
        inputs_str = " ".join(self.graph.inputs)
        lines.append(f"(input {inputs_str})")
        lines.append("")

        # Build outgoing edge map
        outgoing_edges: Dict[str, List[Tuple[str, int]]] = {}
        for child_id, child_node in self.graph.nodes.items():
            for parent_id, slot in child_node.dependencies:
                if parent_id not in outgoing_edges:
                    outgoing_edges[parent_id] = []
                outgoing_edges[parent_id].append((child_id, slot))

        for entry in self.schedule:
            action_type, node_id = entry
            if action_type != "issue":
                continue  # skip spill/reload
            compute_node = self.graph.nodes[node_id]
            opcode = NodeType.to_opcode(compute_node.type)
            is_switch = compute_node.type == NodeType.SWITCH

            lines.append(f"(node {node_id} {opcode}")

            # Wait clauses
            for parent_id, slot in compute_node.dependencies:
                lines.append(f"    (wait {parent_id} {slot})")

            if is_switch:
                # Find merge connected to this Switch via outgoing data edges
                merge_target = None
                for child_id, slot in outgoing_edges.get(node_id, []):
                    child_node = self.graph.nodes.get(child_id)
                    if child_node and child_node.type == NodeType.MERGE:
                        merge_target = (child_id, slot)
                        break

                # Find the fp_mul that feeds the SAME Merge as this Switch
                fp_mul_target = None
                if merge_target:
                    merge_id = merge_target[0]
                    merge_node = self.graph.nodes.get(merge_id)
                    if merge_node:
                        for dep_id, dep_slot in merge_node.dependencies:
                            if dep_id != node_id and dep_id.startswith("fp_mul"):
                                fp_mul_target = (dep_id, 0)
                                break
                if fp_mul_target is None:
                    for nid in self.topo_order:
                        if nid.startswith("fp_mul"):
                            fp_mul_target = (nid, 0)
                            break

                if merge_target:
                    tgt, slot = merge_target
                    lines.append(f"    (send_true {tgt} {slot})")
                if fp_mul_target:
                    tgt, slot = fp_mul_target
                    lines.append(f"    (send_false {tgt} {slot})")

                merge_name = merge_target[0] if merge_target else None
                for child_id, slot in outgoing_edges.get(node_id, []):
                    if child_id != merge_name:
                        lines.append(f"    (send {child_id} {slot})")
            else:
                # Non-Switch: auto-generate send for all outgoing data edges
                emitted = set()
                for child_id, slot in outgoing_edges.get(node_id, []):
                    if (child_id, slot) not in emitted:
                        lines.append(f"    (send {child_id} {slot})")
                        emitted.add((child_id, slot))

            lines.append(")")
            lines.append("")

        outputs_str = " ".join(self.graph.outputs)
        lines.append(f"(output {outputs_str})")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Step 5: Target Computation Graphs
# ═══════════════════════════════════════════════════════════════

def create_dot_product_graph() -> ComputeGraph:
    """A.B + C.D -- a simple 3-operation graph for NN Compiler v0 training."""
    g = ComputeGraph()
    g.add_node("a", NodeType.INPUT)
    g.add_node("b", NodeType.INPUT)
    g.add_node("c", NodeType.INPUT)
    g.add_node("d", NodeType.INPUT)
    g.add_node("mul0", NodeType.MUL)
    g.add_node("mul1", NodeType.MUL)
    g.add_node("add0", NodeType.ADD)
    g.add_node("out0", NodeType.OUTPUT)
    g.add_edge("a", "mul0", 0)
    g.add_edge("b", "mul0", 1)
    g.add_edge("c", "mul1", 0)
    g.add_edge("d", "mul1", 1)
    g.add_edge("mul0", "add0", 0)
    g.add_edge("mul1", "add0", 1)
    g.add_edge("add0", "out0", 0)
    return g


def create_matmul_2x2_graph() -> ComputeGraph:
    """2x2 Matrix Multiply: C = A x B"""
    g = ComputeGraph()
    g.add_node("a00", NodeType.INPUT)
    g.add_node("a01", NodeType.INPUT)
    g.add_node("a10", NodeType.INPUT)
    g.add_node("a11", NodeType.INPUT)
    g.add_node("b00", NodeType.INPUT)
    g.add_node("b01", NodeType.INPUT)
    g.add_node("b10", NodeType.INPUT)
    g.add_node("b11", NodeType.INPUT)

    g.add_node("mul_00_00", NodeType.MUL)
    g.add_node("mul_00_01", NodeType.MUL)
    g.add_node("mul_01_10", NodeType.MUL)
    g.add_node("mul_01_11", NodeType.MUL)
    g.add_node("mul_10_00", NodeType.MUL)
    g.add_node("mul_10_01", NodeType.MUL)
    g.add_node("mul_11_10", NodeType.MUL)
    g.add_node("mul_11_11", NodeType.MUL)

    g.add_node("add_00", NodeType.ADD)
    g.add_node("add_01", NodeType.ADD)
    g.add_node("add_10", NodeType.ADD)
    g.add_node("add_11", NodeType.ADD)

    g.add_node("c00", NodeType.OUTPUT)
    g.add_node("c01", NodeType.OUTPUT)
    g.add_node("c10", NodeType.OUTPUT)
    g.add_node("c11", NodeType.OUTPUT)

    g.add_edge("a00", "mul_00_00", 0); g.add_edge("b00", "mul_00_00", 1)
    g.add_edge("a01", "mul_01_10", 0); g.add_edge("b10", "mul_01_10", 1)
    g.add_edge("a00", "mul_00_01", 0); g.add_edge("b01", "mul_00_01", 1)
    g.add_edge("a01", "mul_01_11", 0); g.add_edge("b11", "mul_01_11", 1)
    g.add_edge("a10", "mul_10_00", 0); g.add_edge("b00", "mul_10_00", 1)
    g.add_edge("a11", "mul_11_10", 0); g.add_edge("b10", "mul_11_10", 1)
    g.add_edge("a10", "mul_10_01", 0); g.add_edge("b01", "mul_10_01", 1)
    g.add_edge("a11", "mul_11_11", 0); g.add_edge("b11", "mul_11_11", 1)

    g.add_edge("mul_00_00", "add_00", 0); g.add_edge("mul_01_10", "add_00", 1)
    g.add_edge("mul_00_01", "add_01", 0); g.add_edge("mul_01_11", "add_01", 1)
    g.add_edge("mul_10_00", "add_10", 0); g.add_edge("mul_11_10", "add_10", 1)
    g.add_edge("mul_10_01", "add_11", 0); g.add_edge("mul_11_11", "add_11", 1)

    g.add_edge("add_00", "c00", 0); g.add_edge("add_01", "c01", 0)
    g.add_edge("add_10", "c10", 0); g.add_edge("add_11", "c11", 0)
    return g


def create_relu_graph() -> ComputeGraph:
    """
    ReLU activation function: f(x) = max(0, x)

    Architecture:
      Input x -> CmpGeZ (x >= 0 -> 1, else -> 0)
      Input x -> Switch slot 0 (data)
      CmpGeZ   -> Switch slot 1 (condition)
      Switch True  -> Merge slot 0 (identity: x when x >= 0)
      Switch False -> Mul(FalsePath, ZeroConst) -> 0.0 -> Merge slot 1

    The Switch routes x to the identity path (via Merge) if x >= 0,
    or to a zeroing multiplication (x * 0 = 0) if x < 0.
    Merge fires on whichever token arrives first (inputs_required=1).
    """
    g = ComputeGraph()

    g.add_node("x", NodeType.INPUT)          # Input value
    g.add_node("zero", NodeType.INPUT)        # Constant 0.0

    g.add_node("cmp", NodeType.CMPGEZ)        # x >= 0? -> 1.0 else 0.0
    g.add_node("sw", NodeType.SWITCH)         # Route based on condition
    g.add_node("zero_path", NodeType.MUL)     # x * 0 = 0 when x < 0
    g.add_node("merge", NodeType.MERGE)       # Merge identity or zero

    g.add_node("out", NodeType.OUTPUT)        # Final output

    # Wiring
    # cmp: CmpGeZ(x) -> 1.0 if x >= 0, 0.0 if x < 0
    g.add_edge("x", "cmp", 0)

    # sw: Switch receives x (slot 0) and condition from cmp (slot 1)
    g.add_edge("x", "sw", 0)
    g.add_edge("cmp", "sw", 1)

    # zero_path: Mul(x, 0.0) -> 0.0. Slot 0 gets x from sw's IfFalse routing.
    g.add_edge("zero", "zero_path", 1)

    # merge: receives from sw (IfTrue) or zero_path (Always)
    g.add_edge("sw", "merge", 0)              # merge slot 0 receives from sw (IfTrue)
    g.add_edge("zero_path", "merge", 1)       # merge slot 1 receives from zero_path

    # Output
    g.add_edge("merge", "out", 0)

    return g


# ═══════════════════════════════════════════════════════════════
#  Main: Baseline Verification
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    vm_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "target", "release", "dfasm",
    )
    if os.name == "nt":
        vm_path += ".exe"

    if not os.path.exists(vm_path):
        print(f"VM binary not found at {vm_path}. Build with: cargo build --release")
        sys.exit(1)

    bridge = VMBridge(vm_path)
    g = create_relu_graph()

    # ── Batch distribution test suite ──────────────
    test_suite = [
        ({"x": 5.0, "zero": 0.0}, {"out": 5.0}),    # positive
        ({"x": -5.0, "zero": 0.0}, {"out": 0.0}),   # negative
        ({"x": 0.0, "zero": 0.0}, {"out": 0.0}),    # boundary
        ({"x": -0.1, "zero": 0.0}, {"out": 0.0}),   # small negative
        ({"x": 0.1, "zero": 0.0}, {"out": 0.1}),    # small positive
    ]

    # ── Test the correct optimal wiring ──────────────
    print("=== Correct ReLU wiring (batch evaluation) ===")
    env = DataflowGymEnv(g, bridge, test_suite)
    env.step({"target_destinations": [("sw", 1, "send")]})
    env.step({"target_destinations": [
        ("merge", 0, "send_true"),
        ("zero_path", 0, "send_false"),
    ]})
    env.step({"target_destinations": [("merge", 1, "send")]})
    _, r, _, info = env.step({"target_destinations": []})
    print(f"  Batch reward: {r}")
    for case in info.get("cases", []):
        print(f"    x={case['inputs']['x']:5.1f} -> reward={case['reward']:.1f}  {case['info']}")

    # ── Test the CHEATING wiring (identity bypass, no zero_path) ──
    print("\n=== Cheating wiring (skip zero_path, all to merge) ===")
    env2 = DataflowGymEnv(g, bridge, test_suite)
    env2.step({"target_destinations": [("sw", 1, "send")]})
    # Both IfTrue and IfFalse go to merge -> the cheating exploit
    env2.step({"target_destinations": [
        ("merge", 0, "send_true"),
        ("merge", 0, "send_false"),  # bypass zero_path entirely
    ]})
    env2.step({"target_destinations": [("merge", 1, "send")]})
    _, r2, _, info2 = env2.step({"target_destinations": []})
    print(f"  Batch reward: {r2}")
    for case in info2.get("cases", []):
        print(f"    x={case['inputs']['x']:5.1f} -> reward={case['reward']:.1f}  {case['info']}")

    # Show generated DF-ASM
    print("\n=== Correct DF-ASM ===")
    print(env._generate_dfasm())
