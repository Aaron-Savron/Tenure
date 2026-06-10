"""
dl_subgraphs.py -- Real deep learning subgraph definitions for OOD generalization testing.

Defines three topology families that real DL compilers encounter daily:
  1. MLP Block:      MatMul -> BiasAdd -> ReLU (deep linear chain)
  2. Residual Add:   Fork-join diamond with parallel lanes (tenure dilation)
  3. LayerNorm:      Wide fan-in reduction (high-pressure convergence)
"""
from typing import List, Tuple
from .compiler_env import ComputeGraph, NodeType


def create_mlp_block(
    num_hidden: int = 2,
    layer_width: int = 4,
    use_bias: bool = False,
    use_relu: bool = False,
) -> ComputeGraph:
    """
    MLP block: Linear repeated num_hidden times.

    Architecture (simplified, no ReLU to avoid Switch/Merge deadlock risk):
        in0, in1, ..., in_{W-1}          (W = layer_width, input activations)
        w0_0, w0_1, ..., w_{H-1}_{W-1}  (weight inputs for each layer)
        b0_0, b0_1, ..., b_{H-1}_{W-1}  (bias inputs, if use_bias)

        Layer 0: mul(in[i], w0[i]) -> (bias add) -> activations
        Layer 1: mul(activation[i], w1[i]) -> (bias add) -> activations
        ...
        Layer H-1: ... -> out[i]

    The chain is purely linear (no reuse, no branches) — the simplest
    test of the model's ability to sprint through deep chains. Default
    is no bias and no ReLU to create clean linear chains that are
    guaranteed deadlock-free.
    """
    g = ComputeGraph()

    # Input activations (shared across all layers' slot 0)
    input_activations = [f"in{i}" for i in range(layer_width)]
    for name in input_activations:
        g.add_node(name, NodeType.INPUT)

    # Weight inputs (separate for each layer's slot 1)
    weight_inputs = []
    for layer_idx in range(num_hidden):
        layer_weights = []
        for i in range(layer_width):
            wname = f"w_{layer_idx}_{i}"
            g.add_node(wname, NodeType.INPUT)
            layer_weights.append(wname)
        weight_inputs.append(layer_weights)

    # Bias inputs (if enabled)
    bias_inputs = []
    if use_bias:
        for layer_idx in range(num_hidden):
            for i in range(layer_width):
                bname = f"b_{layer_idx}_{i}"
                g.add_node(bname, NodeType.INPUT)
                bias_inputs.append(bname)

    # Build layers
    prev_sources = list(input_activations)  # W elements for layer 0

    for layer_idx in range(num_hidden):
        next_sources = []

        # Mul: activation * weight (slot 0 = activation, slot 1 = weight)
        for i in range(layer_width):
            mul_name = f"L{layer_idx}_mul{i}"
            g.add_node(mul_name, NodeType.MUL)
            g.add_edge(prev_sources[i], mul_name, 0)
            g.add_edge(weight_inputs[layer_idx][i], mul_name, 1)
            next_sources.append(mul_name)

        # Bias add (if enabled)
        if use_bias:
            biased = []
            for i in range(layer_width):
                add_name = f"L{layer_idx}_bias{i}"
                g.add_node(add_name, NodeType.ADD)
                g.add_edge(next_sources[i], add_name, 0)
                g.add_edge(bias_inputs[layer_idx * layer_width + i], add_name, 1)
                biased.append(add_name)
            next_sources = biased

        # Note: ReLU is intentionally omitted to avoid Switch/Merge
        # deadlock risk in the simulator. The linear chain (Mul + bias)
        # is sufficient to test the model's deep-chain sprint behavior.

        prev_sources = next_sources

    # Output
    for i in range(layer_width):
        out_name = f"out{i}"
        g.add_node(out_name, NodeType.OUTPUT)
        g.add_edge(prev_sources[i], out_name, 0)

    return g


def create_residual_add(
    num_lanes: int = 3,
    lane_depth: int = 2,
) -> ComputeGraph:
    """
    Residual Add fork-join diamond.

    Architecture:
        in_shared (broadcast)
          |
        fork_mul  (x * weight, produces value that stays live across lanes)
          |
        +-- lane0: op0 -> op1 -> ... -> lane_end0 --+
        |-- lane1: op0 -> op1 -> ... -> lane_end1 --+-> join_add -> out
        |-- lane2: op0 -> op1 -> ... -> lane_end2 --+
        |
        fork_output (the fork_mul result, kept alive across all lane execution)

    The fork_mul's result must remain in a register while all parallel lanes
    execute. At regs=3, this creates tenure dilation — the model must decide
    whether to stall lanes or spill.

    Args:
        num_lanes: number of parallel lanes (default 3, giving 3 concurrent
                   ready nodes + the fork output = 4 live regs at convergence)
        lane_depth: ops per lane (default 2, giving 2-6c of tenure per lane)
    """
    g = ComputeGraph()
    op_counter: dict = {}

    def _name(base: str) -> str:
        op_counter[base] = op_counter.get(base, 0) + 1
        return f"{base}{op_counter[base] - 1}"

    # Shared input
    g.add_node("in_shared", NodeType.INPUT)

    # Weight input for fork Mul
    g.add_node("in_weight", NodeType.INPUT)

    # Lane inputs (each lane gets its own slot-1 source)
    lane_inputs = []
    for i in range(num_lanes):
        name = f"in_lane{i}"
        g.add_node(name, NodeType.INPUT)
        lane_inputs.append(name)

    # Fork: Mul(shared, weight) → result stays live across all lanes
    g.add_node("fork_mul", NodeType.MUL)
    g.add_edge("in_shared", "fork_mul", 0)
    g.add_edge("in_weight", "fork_mul", 1)

    # Build lanes
    lane_ends = []
    for lane_idx in range(num_lanes):
        prev = f"fork_mul"  # each lane's first op consumes fork_mul's output
        for depth_idx in range(lane_depth):
            op_type = NodeType.MUL if depth_idx == 0 else NodeType.ADD
            node_id = f"L{lane_idx}_d{depth_idx}_{op_type.lower()}"
            g.add_node(node_id, op_type)
            g.add_edge(prev, node_id, 0)  # chain from fork or previous in lane

            # Slot 1: lane-specific input for first op, fork_mul for deeper ops
            if depth_idx == 0:
                g.add_edge(lane_inputs[lane_idx], node_id, 1)
            else:
                g.add_edge("fork_mul", node_id, 1)  # re-read fork_mul to extend tenure

            prev = node_id
        lane_ends.append(prev)

    # Join: add all lane outputs + the original fork_mul output
    # This is a right-skewed reduction tree: ((lane0 + lane1) + lane2) + ...
    # The fork_mul value is also added to ensure the model must keep it.
    current = lane_ends[0]
    join_start = 1 + 1  # lane_ends[1:] + fork_mul
    all_join_inputs = lane_ends[1:] + ["fork_mul"]

    for i, src in enumerate(all_join_inputs):
        add_name = f"join_add{i}"
        g.add_node(add_name, NodeType.ADD)
        g.add_edge(current, add_name, 0)
        g.add_edge(src, add_name, 1)
        current = add_name

    # Output
    g.add_node("out", NodeType.OUTPUT)
    g.add_edge(current, "out", 0)

    return g


def create_layer_norm(
    num_channels: int = 6,
) -> ComputeGraph:
    """
    LayerNorm reduction: wide fan-in → mean → variance → normalize.

    Architecture (simplified — computes only the mean + subtract for
    the normalization, which captures the core fan-in register behavior):

        in0  in1  in2  ...  in_{N-1}   (N = num_channels)
         |    |    |          |
         +----+----+----...---+-> add_tree (pairwise reduction)
         |    |    |          |
         +----+----+----...---+-> add_tree (pairwise reduction, separate instance)
              ...             |
                             div(1/N)  ->  mean
                              |
         in0 - mean -> out0
         in1 - mean -> out1
         ...

    The wide fan-in creates a massive register pressure spike as N-1 inputs
    must be live simultaneously during the reduction tree.

    Args:
        num_channels: number of input channels (default 6, giving peak
                      live regs of ~10+ before reduction starts)
    """
    g = ComputeGraph()
    op_counter: dict = {}

    def _name(base: str) -> str:
        op_counter[base] = op_counter.get(base, 0) + 1
        return f"{base}{op_counter[base] - 1}"

    # Create inputs
    input_names = [f"in{i}" for i in range(num_channels)]
    for name in input_names:
        g.add_node(name, NodeType.INPUT)

    # Build pairwise reduction tree for sum
    # Each level halves the number of values
    def build_add_tree(values: List[str], prefix: str) -> str:
        current = list(values)
        level = 0
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    add_name = _name(f"{prefix}_add_l{level}")
                    g.add_node(add_name, NodeType.ADD)
                    g.add_edge(current[i], add_name, 0)
                    g.add_edge(current[i + 1], add_name, 1)
                    next_level.append(add_name)
                else:
                    next_level.append(current[i])
            current = next_level
            level += 1
        return current[0]

    # First reduction: sum of all inputs
    sum_node = build_add_tree(input_names, "mean")

    # Div(1/N) -> mean. Use Mul(sum, 1/N) since we have no Div op.
    inv_n = 1.0 / num_channels
    # Create a constant input for the multiplier
    const_name = "norm_const"
    g.add_node(const_name, NodeType.INPUT)  # feed 1/N at runtime

    mean_node = _name("mean_mul")
    g.add_node(mean_node, NodeType.MUL)
    g.add_edge(sum_node, mean_node, 0)
    g.add_edge(const_name, mean_node, 1)

    # Subtract mean from each input (normalize)
    for i in range(num_channels):
        sub_name = f"norm_out{i}"
        g.add_node(sub_name, NodeType.SUB)
        g.add_edge(input_names[i], sub_name, 0)
        g.add_edge(mean_node, sub_name, 1)
        out_name = f"out{i}"
        g.add_node(out_name, NodeType.OUTPUT)
        g.add_edge(sub_name, out_name, 0)

    return g


# ── Test generators ─────────────────────────────────────────────

def generate_dl_test_suite(graph, generator):
    """
    Generate a simple test suite for DL subgraphs.
    Uses deterministic inputs for reproducibility.
    """
    from .graph_generator import evaluate_graph
    import random
    rng = random.Random(42)

    inputs = {}
    for name in graph.inputs:
        if name.startswith("fp_zero"):
            inputs[name] = 0.0
        elif name.startswith("norm_const"):
            inputs[name] = 1.0 / max(1, len([n for n in graph.inputs if n.startswith("in")]))
        else:
            inputs[name] = rng.uniform(-2.0, 2.0)

    # Single-case suite for now (OOD evaluation doesn't need batch validation)
    # We just want cycle counts and register behavior
    return [(inputs, {})]


# ═══════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("DL Subgraph Generator Self-Test")
    print("=" * 60)

    for name, g_fn in [
        ("MLP Block (1 layer, W=3)", lambda: create_mlp_block(num_hidden=1, layer_width=3)),
        ("Residual Add (3 lanes, depth=2)", lambda: create_residual_add(num_lanes=3, lane_depth=2)),
        ("LayerNorm (6 channels)", lambda: create_layer_norm(num_channels=6)),
        ("Deep MLP (3 layers, W=2)", lambda: create_mlp_block(num_hidden=3, layer_width=2)),
        ("Wide Residual (5 lanes, depth=1)", lambda: create_residual_add(num_lanes=5, lane_depth=1)),
    ]:
        g = g_fn()
        topo = g.get_topological_order()
        print(f"\n{name}:")
        print(f"  Inputs: {len(g.inputs)}, Ops: {len(topo)}, Outputs: {len(g.outputs)}")
        print(f"  Types: {[g.nodes[n].type for n in topo]}")
