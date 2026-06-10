"""
graph_generator.py -- Procedural DAG generator and evaluator for
curriculum-based meta-compiler training.

Generates random ComputeGraphs with structural invariants, evaluates
them to produce expected outputs for correctness validation, and
supports three curriculum phases:

  Phase 1 (Linear Pipes): 3-4 ops, Add/Mul only, chain topologies
  Phase 2 (Branching):    4-6 ops, Add/Mul with fan-out
  Phase 3 (Conditionals): 6-8 ops, full ISA with Switch/Merge/CmpGeZ
"""

import random
from typing import Dict, List, Tuple, Optional

from .compiler_env import (
    ComputeGraph,
    NodeType,
)


# ═══════════════════════════════════════════════════════════════
#  Phase Configuration
# ═══════════════════════════════════════════════════════════════

PHASE_CONFIG = {
    1: {  # Linear Pipes
        "num_inputs": (2, 4),
        "num_ops": (3, 4),
        "op_pool": [NodeType.ADD, NodeType.MUL],
        "max_fanout": 1,
        "allow_conditional": False,
        "allow_skip": False,  # every op must depend on at least one earlier op
    },
    2: {  # Branching
        "num_inputs": (3, 5),
        "num_ops": (4, 6),
        "op_pool": [NodeType.ADD, NodeType.MUL],
        "max_fanout": 3,
        "allow_conditional": False,
        "allow_skip": True,
    },
    3: {  # Conditionals
        "num_inputs": (3, 5),
        "num_ops": (5, 7),  # (5-7 main ops + up to 1 inline Merge = 6-8 total)
        "op_pool": [
            NodeType.ADD, NodeType.MUL,
            NodeType.SWITCH, NodeType.MERGE, NodeType.CMPGEZ,
        ],
        "max_fanout": 3,
        "allow_conditional": True,
        "allow_skip": True,
    },
    4: {  # Parallel Lanes (scheduling curriculum)
        "num_inputs": (3, 6),
        "num_lanes": (3, 5),
        "ops_per_lane": (2, 4),
        "op_pool": [NodeType.ADD, NodeType.MUL],
        "max_fanout": 5,
        "allow_conditional": False,
        "allow_skip": False,
        "terminal_op": [NodeType.ADD, NodeType.MUL],  # how lanes recombine
    },
    5: {  # Wide Parallel Lanes (100+ op scaling curriculum)
        "num_inputs": (4, 10),
        "num_lanes": (8, 20),
        "ops_per_lane": (3, 8),
        "op_pool": [NodeType.ADD, NodeType.MUL],
        "max_fanout": 5,
        "allow_conditional": False,
        "allow_skip": False,
        "terminal_op": [NodeType.ADD, NodeType.MUL],
    },
}


# ═══════════════════════════════════════════════════════════════
#  Graph Evaluator (computes expected outputs)
# ═══════════════════════════════════════════════════════════════

def evaluate_graph(
    graph: ComputeGraph, inputs: Dict[str, float]
) -> Dict[str, float]:
    """
    Evaluate a ComputeGraph for given input values.

    Implements a simplified dataflow interpreter:
    - Nodes fire in topological order when all inputs are available.
    - Switch routes its data value; the condition determines which
      downstream path is active.
    - Merge fires on token from the active path (inputs_required=1).

    Returns: {output_name: computed_value}
    """
    # Values known so far (inputs + computed nodes)
    values: Dict[str, float] = dict(inputs)
    topo = graph.get_topological_order()

    # Build incoming slot map per node
    incoming: Dict[str, Dict[int, str]] = {}
    for child_name, child_node in graph.nodes.items():
        slot_map: Dict[int, str] = {}
        for parent_id, slot in child_node.dependencies:
            slot_map[slot] = parent_id
        incoming[child_name] = slot_map

    # Track conditional state for Switch → Merge tracking
    # Maps node_id -> bool (is the True path active?)
    switch_condition: Dict[str, bool] = {}

    for node_id in topo:
        node = graph.nodes[node_id]
        slots = incoming.get(node_id, {})

        if node.type == NodeType.INPUT:
            continue  # already in values

        elif node.type == NodeType.ADD:
            val0 = values.get(slots.get(0, ""), 0.0)
            val1 = values.get(slots.get(1, ""), 0.0)
            values[node_id] = val0 + val1

        elif node.type == NodeType.MUL:
            val0 = values.get(slots.get(0, ""), 0.0)
            val1 = values.get(slots.get(1, ""), 0.0)
            values[node_id] = val0 * val1

        elif node.type == NodeType.CMPGEZ:
            arg = values.get(slots.get(0, ""), 0.0)
            values[node_id] = 1.0 if arg >= 0.0 else 0.0

        elif node.type == NodeType.SWITCH:
            data = values.get(slots.get(0, ""), 0.0)
            cond = values.get(slots.get(1, ""), 0.0)
            values[node_id] = data
            switch_condition[node_id] = (cond != 0.0)

        elif node.type == NodeType.MERGE:
            # Merge fires on whichever token arrives first (inputs_required=1).
            # The correct token value depends on which Switch path activated.
            # We find it by checking each incoming edge's source.
            candidate = None
            for slot in sorted(slots.keys()):
                src = slots[slot]
                if src in values:
                    # Check if this source is gated by a Switch condition
                    src_node = graph.nodes.get(src)
                    if src_node and src_node.type == NodeType.SWITCH:
                        # Only take this value if the Switch's active path
                        # would send to this Merge slot
                        cond = switch_condition.get(src, True)
                        # Figure out which slot on the Merge this SW sends to
                        # IfTrue is for cond=True, IfFalse for cond=False
                        # The Merge slot determines which one we're reading
                        if (cond and slot == 0) or (not cond and slot == 1):
                            candidate = values[src]
                            break
                        # Otherwise, this SW wouldn't send here
                    else:
                        candidate = values[src]
                        break
            if candidate is not None:
                values[node_id] = candidate
            else:
                # Fallback: first available
                for slot in sorted(slots.keys()):
                    src = slots[slot]
                    if src in values:
                        values[node_id] = values[src]
                        break
                if node_id not in values:
                    values[node_id] = 0.0

    # Process output nodes (they are excluded from topological order)
    for out_name in graph.outputs:
        out_slots = incoming.get(out_name, {})
        for slot in sorted(out_slots.keys()):
            src = out_slots[slot]
            if src in values:
                values[out_name] = values[src]
                break
        if out_name not in values:
            values[out_name] = 0.0

    return {name: values.get(name, float("nan")) for name in graph.outputs}


# ═══════════════════════════════════════════════════════════════
#  Procedural Graph Generator
# ═══════════════════════════════════════════════════════════════

class ProceduralGraphGenerator:
    """
    Generates random ComputeGraphs with structural invariants enforced.

    Usage:
        gen = ProceduralGraphGenerator(seed=42)
        graph = gen.generate(phase=1)  # Phase 1: arithmetic chain
        inputs = gen.generate_inputs(graph)
        expected = gen.compute_expected(graph, inputs)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(self, phase: int = 1) -> ComputeGraph:
        """Generate a random ComputeGraph for the given curriculum phase."""
        config = PHASE_CONFIG.get(phase)
        if config is None:
            raise ValueError(f"Unknown phase: {phase}")

        g = ComputeGraph()
        op_counter: Dict[str, int] = {}

        def _node_name(base: str) -> str:
            op_counter[base] = op_counter.get(base, 0) + 1
            return f"{base}{op_counter[base] - 1}"

        # ── Phase 4/5: parallel lane topology ────────────
        # Phase 4/5 creates all inputs internally; skip outer input creation.
        if phase in (4, 5):
            return self._generate_phase4(config, _node_name)

        # ── Create input nodes ──────────────────────────
        num_inputs = self.rng.randint(*config["num_inputs"])
        input_names = [f"in{i}" for i in range(num_inputs)]
        for name in input_names:
            g.add_node(name, NodeType.INPUT)

        # ── Create operation nodes ──────────────────────
        # Phase 3 config (5,7) yields 5-7 main-loop ops = 6-8 total
        # with inline Merges. When no Switch is generated in the main
        # loop, _force_terminal_conditional adds 2-3 more nodes,
        # bringing the total to 7-10 — still well within reason.
        num_ops = self.rng.randint(*config["num_ops"])
        op_nodes: List[str] = []
        op_types: Dict[str, str] = {}

        # Track which nodes are available as wiring sources
        available_sources: List[str] = list(input_names)

        # ── Build operations ──────────────────────────
        # Phase 1: chain topology (each op depends on op_nodes[-1])
        # Phase 2: chain + branch points for fan-out
        # Phase 3: chain + Switch-Merge conditional clusters
        #
        # Critical: the LAST op always uses op_nodes[-1] as its chain source
        # to guarantee full connectivity. Branching (fan-out) is restricted
        # to non-terminal ops.

        pending_switch = False  # next op must be Switch
        branch_points: List[str] = []  # nodes available for fan-out (Phase 2)

        for i in range(num_ops):
            # ── Pick opcode based on pending flags ──────
            if pending_switch:
                op_type = NodeType.SWITCH
                pending_switch = False
            elif config["allow_conditional"] and len(op_nodes) >= 2:
                # Only allow Switch if a CmpGeZ exists for its condition.
                # The pending_switch mechanism already guarantees proper
                # CmpGeZ → Switch → Merge sequences; this filter prevents
                # random Switch selection when no condition source is available.
                pool = list(config["op_pool"])
                has_cmp = any(op_types.get(n) == NodeType.CMPGEZ for n in op_nodes)
                if not has_cmp:
                    pool = [t for t in pool if t != NodeType.SWITCH]
                op_type = self.rng.choice(pool)
            else:
                op_type = self.rng.choice(
                    [t for t in config["op_pool"]
                     if t not in (NodeType.SWITCH, NodeType.MERGE, NodeType.CMPGEZ)]
                )

            node_id = _node_name(op_type.lower())
            is_terminal = (i == num_ops - 1)
            non_op_sources = [n for n in available_sources if n in input_names]

            # ── Determine chain source (slot 0) ────────
            # Slot 0 is ALWAYS the immediate predecessor (op_nodes[-1]) when
            # available. This guarantees full DFS connectivity: the output is
            # wired to the last op, which depends on the second-to-last, etc.
            if i > 0 and op_nodes:
                chain_src = op_nodes[-1]
            else:
                chain_src = self.rng.choice(available_sources)

            # ── Determine slot 1 source ─────────────────
            # For Phase 2+, slot 1 can be a branch point (fan-out) or a
            # random input. For Phase 1 and non-branch cases, it's an input.
            is_branch = (
                phase >= 2
                and len(branch_points) > 0
                and self.rng.random() < 0.3
                and op_type not in (NodeType.SWITCH, NodeType.MERGE, NodeType.CMPGEZ)
            )
            if is_branch:
                slot1_src = self.rng.choice(branch_points)  # fan-out!
            elif non_op_sources and self.rng.random() < 0.7:
                # Ensure slot1 != chain_src — the VM's matching store
                # delivers only ONE token per source node per target node.
                # If both slots reference the same source, the second
                # slot never receives a value.
                non_op_candidates = [
                    s for s in non_op_sources if s != chain_src
                ]
                slot1_src = (
                    self.rng.choice(non_op_candidates)
                    if non_op_candidates else chain_src
                )
            else:
                others = [s for s in available_sources if s != chain_src]
                slot1_src = self.rng.choice(others) if others else chain_src

            # ── CmpGeZ: needs 1 input ──────────────────
            if op_type == NodeType.CMPGEZ:
                g.add_node(node_id, NodeType.CMPGEZ)
                g.add_edge(chain_src, node_id, 0)
                if config["allow_conditional"] and self.rng.random() < 0.4:
                    pending_switch = True

            # ── Switch: needs 2 inputs (data + condition) ──
            elif op_type == NodeType.SWITCH:
                cmp_nodes = [n for n in op_nodes if op_types.get(n) == NodeType.CMPGEZ]
                cond_src = self.rng.choice(cmp_nodes) if cmp_nodes else chain_src
                g.add_node(node_id, NodeType.SWITCH)
                g.add_edge(chain_src, node_id, 0)  # data
                g.add_edge(cond_src, node_id, 1)   # condition

                # Always create the false-path ghost track.
                # Every Switch needs both IfTrue (→Merge) and IfFalse (→fp_mul)
                # routing targets. Without the ghost track, non-terminal Switches
                # have no fp_mul for send_false, leaving 20% of Phase 3 graphs
                # with incomplete conditional routing space.
                fp_zero = _node_name("fp_zero")
                g.add_node(fp_zero, NodeType.INPUT)
                input_names.append(fp_zero)

                fp_mul = _node_name("fp_mul")
                g.add_node(fp_mul, NodeType.MUL)
                g.add_edge(fp_zero, fp_mul, 1)  # zero → Mul slot 1

                mg_name = _node_name("merge")
                g.add_node(mg_name, NodeType.MERGE)
                g.add_edge(node_id, mg_name, 0)  # Switch IfTrue → Merge slot 0
                g.add_edge(fp_mul, mg_name, 1)   # fp_mul → Merge slot 1

                op_types[mg_name] = NodeType.MERGE
                op_types[fp_mul] = NodeType.MUL
                available_sources.extend([fp_mul, mg_name])

                # Merge replaces Switch as the terminal node for chain continuity.
                # Subsequent ops depend on the Merge's output (which aggregates
                # both IfTrue and IfFalse paths).
                node_id = mg_name

            # ── Merge: needs 2 inputs ────────────────────
            elif op_type == NodeType.MERGE:
                g.add_node(node_id, NodeType.MERGE)
                g.add_edge(chain_src, node_id, 0)
                g.add_edge(slot1_src, node_id, 1)

            # ── Add / Mul: needs 2 inputs ───────────────
            else:
                g.add_node(node_id, op_type)
                g.add_edge(chain_src, node_id, 0)
                g.add_edge(slot1_src, node_id, 1)

            op_nodes.append(node_id)
            op_types[node_id] = op_type
            available_sources.append(node_id)

            # Record branch points for Phase 2 fan-out
            if op_type in (NodeType.ADD, NodeType.MUL) and phase >= 2 and not is_terminal:
                if self.rng.random() < 0.4:
                    branch_points.append(node_id)

        # ── Force terminal conditional for Phase 3 ──────
        # If no Switch-Merge pair was generated in the main loop,
        # force-add a terminal CmpGeZ + Switch with inline Merge.
        # This guarantees every Phase 3 graph has conditional
        # routing, giving the policy a training signal in every
        # single episode.
        if phase >= 3:
            has_switch = any(
                op_types.get(n) == NodeType.SWITCH for n in op_nodes
            )
            if not has_switch:
                self._force_terminal_conditional(
                    g, op_nodes, op_types,
                    available_sources, input_names,
                )

        # ── Create output node ─────────────────────────
        # Wire the last operation node to the output
        if op_nodes:
            output_name = "out"
            g.add_node(output_name, NodeType.OUTPUT)
            g.add_edge(op_nodes[-1], output_name, 0)

        return g

    def _force_terminal_conditional(
        self, g: ComputeGraph, op_nodes: List[str],
        op_types: Dict[str, str],
        available_sources: List[str],
        input_names: List[str],
    ) -> None:
        """
        After the main generation loop for Phase 3, if no Switch-Merge
        pair was created, force-add a terminal CmpGeZ + Switch with
        inline Merge (ghost track). This guarantees every Phase 3 graph
        has conditional routing, giving the policy a training signal in
        every single episode.

        Modifies op_nodes, op_types, available_sources, input_names in place.
        The existing output edge is removed (it was wired to op_nodes[-1])
        and re-wired to the new Merge node.
        """
        def _node_name(base: str) -> str:
            op_counter: Dict[str, int] = {}
            for n in op_nodes + g.inputs + g.outputs:
                for k in ["cmpgez", "switch", "fp_zero", "fp_mul", "merge"]:
                    if n.startswith(k):
                        try:
                            idx = int(n[len(k):])
                            op_counter[k] = max(op_counter.get(k, 0), idx + 1)
                        except ValueError:
                            pass
            op_counter[base] = op_counter.get(base, 0) + 1
            return f"{base}{op_counter[base] - 1}"

        # Remove the existing output edge (it points to op_nodes[-1])
        for out_name in g.outputs:
            g.nodes[out_name].dependencies = []

        chain_src = op_nodes[-1] if op_nodes else available_sources[0]

        # ── CmpGeZ ────────────────────────────────────
        cmp_name = _node_name("cmpgez")
        g.add_node(cmp_name, NodeType.CMPGEZ)
        g.add_edge(chain_src, cmp_name, 0)
        op_nodes.append(cmp_name)
        op_types[cmp_name] = NodeType.CMPGEZ
        available_sources.append(cmp_name)

        # ── Switch with inline Merge (ghost track) ────
        sw_name = _node_name("switch")
        g.add_node(sw_name, NodeType.SWITCH)
        g.add_edge(chain_src, sw_name, 0)  # data from chain
        g.add_edge(cmp_name, sw_name, 1)   # condition from CmpGeZ

        # False-path zero constant
        fp_zero = _node_name("fp_zero")
        g.add_node(fp_zero, NodeType.INPUT)
        input_names.append(fp_zero)

        # False-path Mul(zero, switch_data) → 0.0
        fp_mul = _node_name("fp_mul")
        g.add_node(fp_mul, NodeType.MUL)
        g.add_edge(fp_zero, fp_mul, 1)  # zero → Mul slot 1

        # Merge: Switch IfTrue → slot 0, fp_mul → slot 1
        mg_name = _node_name("merge")
        g.add_node(mg_name, NodeType.MERGE)
        g.add_edge(sw_name, mg_name, 0)   # Switch → Merge slot 0
        g.add_edge(fp_mul, mg_name, 1)    # fp_mul → Merge slot 1

        op_types[mg_name] = NodeType.MERGE
        op_types[fp_mul] = NodeType.MUL
        op_types[sw_name] = NodeType.SWITCH
        available_sources.extend([fp_mul, mg_name])
        op_nodes.append(mg_name)  # Merge is the new terminal

        # Re-wire output to the Merge
        for out_name in g.outputs:
            g.add_edge(mg_name, out_name, 0)

    def _generate_phase4(
        self, config: dict,
        _node_name,
    ) -> ComputeGraph:
        """
        Generate a Phase 4 graph with parallel execution lanes.

        Structure:
          [Broadcast Input] ─┬── Lane 0: op0 → op1 → ... → last
                             ├── Lane 1: op0 → op1 → ... → last
                             └── Lane 2: op0 → op1 → ... → last
                                        │
                             [Right-fold Reduction]
                                        │
                                     [Output]

        At step 0, the first op of every lane is ready simultaneously,
        giving a ready-set width of num_lanes (3-5). The policy must
        learn which lane interleaving minimizes total cycles.
        """
        g = ComputeGraph()
        input_names: List[str] = []
        op_counter: Dict[str, int] = {}

        def _name(base: str) -> str:
            op_counter[base] = op_counter.get(base, 0) + 1
            return f"{base}{op_counter[base] - 1}"

        # ── Create all input nodes ────────────────────
        # One broadcast input feeds all lanes; remaining inputs feed
        # individual lane ops (slot 1 of the first op in each lane).
        broadcast_name = "in_bcast"
        g.add_node(broadcast_name, NodeType.INPUT)
        input_names.append(broadcast_name)

        num_lanes = self.rng.randint(*config["num_lanes"])
        # Extra inputs for lane slot-1 sources (at least 1 per lane)
        num_extra = max(0, config["num_inputs"][0] - 1)
        num_extra = self.rng.randint(num_extra, config["num_inputs"][1] - 1)
        for i in range(num_extra):
            name = f"in{i}"
            g.add_node(name, NodeType.INPUT)
            input_names.append(name)

        all_inputs = list(input_names)
        lane_sources = [s for s in all_inputs if s != broadcast_name]

        # ── Build parallel lanes ───────────────────────
        lane_last: List[str] = []  # last node of each lane for terminal reduction
        all_op_nodes: List[str] = []
        op_types: Dict[str, str] = {}

        for lane_idx in range(num_lanes):
            ops_this_lane = self.rng.randint(*config["ops_per_lane"])
            lane_nodes: List[str] = []

            for op_idx in range(ops_this_lane):
                op_type = self.rng.choice(config["op_pool"])
                node_id = _name(f"L{lane_idx}_{op_type.lower()}")
                g.add_node(node_id, op_type)
                all_op_nodes.append(node_id)
                op_types[node_id] = op_type

                # Slot 0: first op gets broadcast input; rest get previous
                if op_idx == 0:
                    g.add_edge(broadcast_name, node_id, 0)
                else:
                    g.add_edge(lane_nodes[-1], node_id, 0)

                # Slot 1: pick from lane sources, avoiding slot-0 source.
                # The VM's matching store delivers only ONE token per
                # (source, target) pair, so slot 1 must differ from slot 0.
                slot0_src = broadcast_name if op_idx == 0 else lane_nodes[-1]
                non_colliding = [
                    s for s in (lane_sources + [broadcast_name])
                    if s != slot0_src
                ]
                if non_colliding:
                    src = self.rng.choice(non_colliding)
                else:
                    src = broadcast_name  # fallback (shouldn't happen)
                g.add_edge(src, node_id, 1)

                lane_nodes.append(node_id)

            if lane_nodes:
                lane_last.append(lane_nodes[-1])

        # ── Terminal right-fold reduction ──────────────
        # (lane0 + lane1) → tmp1 → (tmp1 + lane2) → tmp2 → ...
        # This right-skewed structure creates asymmetric output-hop
        # distances per lane: later lanes are shallower to the output.
        # This is a feature — it forces the policy to balance short
        # and long latency paths.
        terminal_type = self.rng.choice(config["terminal_op"])
        terminal_prefix = "reduce" if terminal_type == NodeType.ADD else "combine"

        if len(lane_last) >= 2:
            current = lane_last[0]
            for i, next_lane in enumerate(lane_last[1:], start=1):
                node_id = _name(f"{terminal_prefix}_{terminal_type.lower()}")
                g.add_node(node_id, terminal_type)
                g.add_edge(current, node_id, 0)
                g.add_edge(next_lane, node_id, 1)
                all_op_nodes.append(node_id)
                op_types[node_id] = terminal_type
                current = node_id
            lane_last = [current]

        # ── Output ──────────────────────────────────────
        output_name = "out"
        g.add_node(output_name, NodeType.OUTPUT)
        g.add_edge(lane_last[-1], output_name, 0)

        return g

    def generate_inputs(
        self, graph: ComputeGraph,
        batch_size: int = 3,
    ) -> List[Dict[str, float]]:
        """Generate random input values for a graph.

        Dedicated fp_zero inputs (false-path zero constants) are always 0.0.
        """
        batches = []
        for _ in range(batch_size):
            inputs: Dict[str, float] = {}
            for name in graph.inputs:
                if name.startswith("fp_zero"):
                    inputs[name] = 0.0  # false-path constant zero
                else:
                    inputs[name] = self.rng.uniform(-5.0, 5.0)
            batches.append(inputs)
        return batches

    def compute_expected(
        self, graph: ComputeGraph, inputs: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute expected output values for given inputs."""
        return evaluate_graph(graph, inputs)

    def generate_test_suite(
        self, graph: ComputeGraph, batch_size: int = 3
    ) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        """Generate a full test suite: (inputs, expected_outputs) pairs."""
        suite = []
        for inputs in self.generate_inputs(graph, batch_size):
            expected = self.compute_expected(graph, inputs)
            suite.append((inputs, expected))
        return suite


# ═══════════════════════════════════════════════════════════════
#  Main: Verification
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys

    # Try to load the VM bridge for end-to-end verification
    vm_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "target", "release", "dfasm",
    )
    if os.name == "nt":
        vm_path += ".exe"

    bridge = None
    if os.path.exists(vm_path):
        from .compiler_env import VMBridge, DataflowGymEnv
        bridge = VMBridge(vm_path)

    gen = ProceduralGraphGenerator(seed=42)

    for phase in [1, 2, 3]:
        print(f"\n{'=' * 60}")
        print(f"Phase {phase}: {PHASE_CONFIG[phase]['op_pool']}")
        print(f"{'=' * 60}")

        for trial in range(5):
            g = gen.generate(phase=phase)
            topo = g.get_topological_order()
            suite = gen.generate_test_suite(g, batch_size=3)

            print(f"\n  Trial {trial + 1}: {len(g.inputs)} inputs, "
                  f"{len(topo)} ops, {len(g.outputs)} outputs")
            print(f"    Topo: {topo}")
            print(f"    Types: {[g.nodes[n].type for n in topo]}")

            # Show first test case
            inps, exps = suite[0]
            inps_str = ", ".join(f"{k}={v:.1f}" for k, v in inps.items())
            exps_str = ", ".join(f"{k}={v:.1f}" for k, v in exps.items())
            print(f"    Inputs:  {inps_str}")
            print(f"    Outputs: {exps_str}")

            # End-to-end test via VM if available
            if bridge is not None and phase <= 2:
                # For Phase 1 and 2 (no conditionals), test baseline routing
                env = DataflowGymEnv(g, bridge, suite)
                # Take no actions (auto-wiring handles arithmetic only)
                for _ in range(len(topo)):
                    _, _, done, _ = env.step({"target_destinations": []})
                    if done:
                        break
