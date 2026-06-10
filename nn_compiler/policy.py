"""
policy.py -- GNN policy with expanded action space supporting routing types.

Action space: num_nodes * 2 slots * 3 routing types = num_nodes * 6
  Routing types: 0=send (Always), 1=send_true, 2=send_false
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATConv, GATv2Conv


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

NUM_SLOTS = 2          # slot 0, slot 1
NUM_ROUTING_TYPES = 3  # Always=0, IfTrue=1, IfFalse=2
ACTIONS_PER_NODE = NUM_SLOTS * NUM_ROUTING_TYPES  # 6


# ═══════════════════════════════════════════════════════════════
#  Observation Encoder
# ═══════════════════════════════════════════════════════════════

def encode_observation(env):
    """
    Convert the current environment state into GNN-ready tensors.

    Node features [num_nodes, 9]:
      [0..5] = one-hot opcode: Input, Mul, Add, Switch, CmpGeZ, Output
      [6]    = is this the current node being compiled?
      [7]    = slot 0 available (unclaimed)?
      [8]    = slot 1 available (unclaimed)?

    Valid mask: [num_nodes * 6] boolean for each (target, slot, routing_type) combo.
    """
    graph = env.graph
    topo = env.topo_order
    node_names = graph.inputs + topo + graph.outputs
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    num_nodes = len(node_names)

    # ── Edge index ──────────────────────────────────
    edge_list = []
    for child_name, child_node in graph.nodes.items():
        for parent_name, _ in child_node.dependencies:
            if parent_name in name_to_idx and child_name in name_to_idx:
                edge_list.append([name_to_idx[parent_name], name_to_idx[child_name]])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ── Claimed slots ───────────────────────────────
    claimed_slots = {}
    for routing_list in env.forward_routing.values():
        for entry in routing_list:
            target = entry[0]
            slot = entry[1]
            if target not in claimed_slots:
                claimed_slots[target] = set()
            claimed_slots[target].add(slot)

    # ── Node features ───────────────────────────────
    # Node types: Input=0, Mul=1, Add=2, Switch=3, CmpGeZ=4, Output=5,
    #             Merge=6, Sub=7, Div=8  (for future use we use 6 features)
    type_to_feat = {
        "Input": 0, "Mul": 1, "Add": 2, "Switch": 3,
        "CmpGeZ": 4, "Output": 5, "Merge": 6,
    }
    feat_dim = 9
    x = torch.zeros(num_nodes, feat_dim)

    current_name = topo[env.current_node_idx] if env.current_node_idx < len(topo) else None

    for i, name in enumerate(node_names):
        node_obj = graph.nodes.get(name)
        if node_obj is None:
            continue
        feat_idx = type_to_feat.get(node_obj.type, 0)
        x[i, feat_idx] = 1.0

        x[i, 6] = 1.0 if name == current_name else 0.0

        if name not in claimed_slots:
            x[i, 7] = 1.0  # slot 0 free
            x[i, 8] = 1.0  # slot 1 free
        else:
            x[i, 7] = 0.0 if 0 in claimed_slots[name] else 1.0
            x[i, 8] = 0.0 if 1 in claimed_slots[name] else 1.0

    current_idx = name_to_idx.get(current_name, 0) if current_name else 0

    # ── Valid action mask ───────────────────────────
    # For each (target_node, slot, routing_type): is it valid?
    # routing_type 0 (Always): valid for all non-Switch nodes on available slots
    # routing_type 1 (IfTrue): valid only if current node is Switch
    # routing_type 2 (IfFalse): valid only if current node is Switch
    #
    # For Switch nodes: ALL outgoing routing is handled by the agent
    # (no auto-wiring for Switch outputs). So all available slots are valid.
    #
    # For non-Switch nodes: slots served by outgoing data dependencies
    # are auto-generated in _generate_dfasm(), so the agent doesn't need
    # to route to them — they're excluded from the valid mask.
    valid_mask = torch.zeros(num_nodes * ACTIONS_PER_NODE, dtype=torch.bool)

    if current_name is not None and current_name in graph.nodes:
        current_node_obj = graph.nodes[current_name]
        is_switch = current_node_obj.type == "Switch"

        current_idx_in_topo = topo.index(current_name)
        later_nodes = topo[current_idx_in_topo + 1:]

        # Build set of valid target names
        target_candidates = set(later_nodes)
        for out_id in graph.outputs:
            out_node = graph.nodes[out_id]
            for src_id, _ in out_node.dependencies:
                if src_id == current_name:
                    target_candidates.add(out_id)

        # For non-Switch nodes, find (target, slot) pairs that will be
        # auto-generated as (send ...) in _generate_dfasm(). The agent
        # doesn't need to route to these — they're handled automatically.
        auto_wired_outgoing: set = set()
        if not is_switch:
            # Find all data edges where current node's output is consumed
            for child_name, child_node in graph.nodes.items():
                for parent_id, slot in child_node.dependencies:
                    if parent_id == current_name:
                        auto_wired_outgoing.add((child_name, slot))

        # Build list of Switch-cluster nodes that non-Switch nodes must
        # not route to. These slots are reserved for the Switch's conditional
        # routing. Auto-wired data dependencies into these nodes are handled
        # automatically by _generate_dfasm().
        #
        # Included:
        #   - Switch (only receives data via auto-wired deps)
        #   - Merge  (only receives from Switch's send_true/send_false)
        #   - fp_*   (false-path nodes: fp_zero, fp_mul; these have no Switch
        #             in their dependency chain, so the dep-chain check alone
        #             won't catch them)
        switch_cluster_nodes: set = set()
        for nid in later_nodes:
            nt = graph.nodes[nid].type
            if nt == "Switch" or nt == "Merge" or nid.startswith("fp_"):
                switch_cluster_nodes.add(nid)
        # Also catch any node that has a Switch as a direct dependency
        # (belt-and-suspenders for Merge nodes or unusual naming).
        for nid in list(target_candidates):
            node = graph.nodes.get(nid)
            if node is not None:
                for dep_id, _ in node.dependencies:
                    dep_node = graph.nodes.get(dep_id)
                    if dep_node is not None and dep_node.type == "Switch":
                        switch_cluster_nodes.add(nid)

        for target in target_candidates:
            # Non-Switch nodes must not route into Switch-cluster nodes.
            # Those slots are reserved for the Switch's conditional routing.
            # Auto-wired data dependencies into these nodes are handled
            # automatically by _generate_dfasm().
            if not is_switch and target in switch_cluster_nodes:
                continue

            if target in name_to_idx:
                tidx = name_to_idx[target]
                target_slots = claimed_slots.get(target, set())
                for slot in [0, 1]:
                    if slot in target_slots:
                        continue
                    # For non-Switch nodes, skip slots auto-generated by data deps
                    if not is_switch and (target, slot) in auto_wired_outgoing:
                        continue
                    base = tidx * ACTIONS_PER_NODE + slot * NUM_ROUTING_TYPES
                    # Always routing (type 0) valid for all nodes
                    valid_mask[base + 0] = True
                    # Conditional routing only for Switch
                    if is_switch:
                        valid_mask[base + 1] = True  # IfTrue
                        valid_mask[base + 2] = True  # IfFalse

    return {
        "x": x,
        "edge_index": edge_index,
        "current_idx": current_idx,
        "valid_mask": valid_mask,
        "node_names": node_names,
        "is_switch": current_name is not None and current_name in graph.nodes and graph.nodes[current_name].type == "Switch",
    }


# ═══════════════════════════════════════════════════════════════
#  Lane Topology Inference
# ═══════════════════════════════════════════════════════════════

def _infer_lane_topology(graph, topo_order, node_names):
    """
    Infer parallel-lane structure from a dataflow DAG by structural traversal.

    Returns five per-node dicts keyed by node name:
      lane_id:             int   — lane index (0-based), -1 for non-lane nodes
      lane_position:       int   — distance from lane root (0-based)
      is_reduction:        bool  — True if node is a cross-lane reduction point
      lane_depth:          int   — total number of ops in this node's lane
      reduction_position:  int   — position within reduction right-fold chain
                            (0-based), -1 for non-reduction nodes

    Algorithm:
      1. Find the broadcast input: the INPUT node with highest out-degree.
      2. Its op-node children are the lane roots (first op of each lane).
      3. Walk each chain forward: a node stays in the same lane as long as
         it has exactly ONE child that is an op node. When a node feeds into
         a reduction node (an op node with >= 2 op-node parents), the chain
         ends and the target is classified as a reduction node.
      4. The reduction right-fold chain is identified transitively.

    Works on any DAG — no naming-convention dependency. Non-lane nodes
    (INPUT, OUTPUT) get lane_id=-1 and zero-valued structural features.
    """
    op_set = set(topo_order)
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    # Build children map (parent -> [child names that are op nodes])
    children: dict = {nid: [] for nid in graph.nodes}
    for child_id, child_node in graph.nodes.items():
        for parent_id, _ in child_node.dependencies:
            if parent_id in children:
                children[parent_id].append(child_id)

    # Build op-node parent counts: for each node, how many parents are op nodes?
    op_parent_count: dict = {}
    for nid in graph.nodes:
        node = graph.nodes[nid]
        op_parents = [
            p for p, _ in node.dependencies
            if p in op_set
        ]
        op_parent_count[nid] = len(op_parents)

    # ── 1. Find broadcast input ─────────────────────
    broadcast_name = None
    max_out = 0
    for in_id in graph.inputs:
        out_deg = len(children.get(in_id, []))
        if out_deg > max_out:
            max_out = out_deg
            broadcast_name = in_id

    # ── 2. Lane roots = op-node children of broadcast input ─
    lane_roots = []
    if broadcast_name is not None:
        for child in children.get(broadcast_name, []):
            if child in op_set:
                lane_roots.append(child)

    # ── 3. Walk each lane chain ─────────────────────
    lane_id: dict = {}
    lane_position: dict = {}
    lane_depth: dict = {}
    is_reduction: dict = {}

    # Find reduction nodes: op nodes with >= 2 op-node parents.
    # These are the right-fold merge points.
    reduction_nodes: set = set()
    for nid in op_set:
        if op_parent_count.get(nid, 0) >= 2:
            reduction_nodes.add(nid)

    for root_idx, root in enumerate(lane_roots):
        # Walk forward from root
        chain = [root]
        current = root
        while True:
            # Get op-node children of current
            op_children = [c for c in children.get(current, []) if c in op_set]

            if len(op_children) == 1:
                nxt = op_children[0]
                # If the only child is a reduction node, stop —
                # the current node is the lane-last.
                if nxt in reduction_nodes:
                    break
                chain.append(nxt)
                current = nxt
            elif len(op_children) == 0:
                # Dead end (shouldn't happen in Phase 4/5, but handle gracefully)
                break
            else:
                # Multiple op children — this node is a fork point,
                # not a simple chain. Stop the lane here.
                break

        # Assign lane features
        depth = len(chain)
        for pos, nid in enumerate(chain):
            lane_id[nid] = root_idx
            lane_position[nid] = pos
            lane_depth[nid] = depth
            is_reduction[nid] = False

    # ── 4. Classify reduction nodes ──────────────────
    # Reduction nodes appear in topo_order; assign them is_reduction=True
    # and track position within the reduction right-fold chain.
    reduction_chain = []
    for nid in topo_order:
        if nid in reduction_nodes:
            reduction_chain.append(nid)

    reduction_position: dict = {}
    num_red = len(reduction_chain)
    for pos, nid in enumerate(reduction_chain):
        lane_id[nid] = -1
        lane_position[nid] = 0
        lane_depth[nid] = 0
        is_reduction[nid] = True
        reduction_position[nid] = pos

    # ── 5. Assign defaults for all unclassified nodes ─
    for nid in graph.nodes:
        if nid not in lane_id:
            lane_id[nid] = -1
            lane_position[nid] = 0
            lane_depth[nid] = 0
            is_reduction[nid] = False
        if nid not in reduction_position:
            reduction_position[nid] = -1

    return lane_id, lane_position, is_reduction, lane_depth, reduction_position


# ═══════════════════════════════════════════════════════════════
#  Scheduling Observation Encoder
# ═══════════════════════════════════════════════════════════════

def encode_scheduling_obs(env):
    """
    Convert the scheduling environment state into GNN-ready tensors.

    Node features [num_nodes, 20]:
      [0..6]  = one-hot opcode: Input, Mul, Add, Switch, CmpGeZ, Output, Merge
      [7]     = is this node ready to fire?
      [8]     = has this node been executed already?
      [9]     = node height (longest hop path to output, normalized)
      [10]    = downstream latency sum (total descendant work, normalized)
      [11]    = CPD: critical path distance (longest latency path to output,
                normalized by max CPD across all nodes)
      [12]    = register pressure: outstanding_values / max_registers (capped at 1.0),
                0.0 when max_registers is not set (unlimited)
      [13]    = Is_Spilled: 1.0 if node's output is currently in the stack pool
                (spilled out of register file), 0.0 otherwise.
      [14]    = Remaining_Reloads: count of yet-to-issue consumers for this
                spilled value (0 for non-spilled nodes).
      [15]    = lane_id: which parallel lane this node belongs to,
                normalized by (num_lanes - 1). 0.0 for non-lane nodes.
      [16]    = lane_position: distance from lane root, normalized by
                (lane_depth - 1). Higher = closer to reduction point.
      [17]    = is_reduction: 1.0 if cross-lane reduction node, 0.0 otherwise.
      [18]    = lane_depth: total ops in this lane, normalized by max lane depth.
      [19]    = reduction_position: position within reduction right-fold chain,
                normalized. 0.0 for non-reduction nodes.

    Action mask: [2 * num_op_nodes] boolean where:
      0..N-1: Issue (node unissued + ready + not spilled) or
              Reload (node spilled + not reloading)
      N..2N-1: Spill (node's output live in a register)
    """
    from .scheduler_baseline import critical_path_height
    from .compiler_env import DEFAULT_LATENCY

    graph = env.graph
    topo_order = env.topo_order
    node_names = graph.inputs + topo_order + graph.outputs
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    num_nodes = len(node_names)

    # Edge index
    edge_list = []
    for child_name, child_node in graph.nodes.items():
        for parent_name, _ in child_node.dependencies:
            if parent_name in name_to_idx and child_name in name_to_idx:
                edge_list.append([name_to_idx[parent_name], name_to_idx[child_name]])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ── Structural features ───────────────────────
    heights = critical_path_height(graph)
    max_height = max(heights.values()) if heights else 1

    # Downstream latency: sum of latencies of the node itself + descendants.
    # Compute in reverse topological order (children before parents).
    latency = DEFAULT_LATENCY
    downstream: dict = {}
    # Build children map for reverse traversal
    children: dict = {nid: [] for nid in graph.nodes}
    for child_id, child_node in graph.nodes.items():
        for parent_id, _ in child_node.dependencies:
            if parent_id in children:
                children[parent_id].append(child_id)

    for nid in reversed(topo_order):
        node = graph.nodes[nid]
        own = latency.get(node.type, 1)
        child_sum = sum(downstream.get(c, 0) for c in children.get(nid, []))
        downstream[nid] = own + child_sum
    max_downstream = max(downstream.values()) if downstream else 1

    # Critical Path Distance: max latency path to any output.
    # CPD(node) = Latency(node) + max(CPD(child)) for children in the DAG.
    # Unlike downstream (sum), this captures the longest single path.
    cpd: dict = {}
    for nid in reversed(topo_order):
        node = graph.nodes[nid]
        own = latency.get(node.type, 1)
        child_max = max((cpd.get(c, 0) for c in children.get(nid, [])), default=0)
        cpd[nid] = own + child_max
    max_cpd = max(cpd.values()) if cpd else 1

    # ── Lane topology inference ───────────────────
    lane_id, lane_position, is_reduction, lane_depth, reduction_position = _infer_lane_topology(
        graph, topo_order, node_names
    )
    num_lanes = max(1, max((v for v in lane_id.values() if v >= 0), default=0) + 1)
    max_lane_depth = max((v for v in lane_depth.values() if v > 0), default=1)
    num_red_nodes = sum(1 for v in is_reduction.values() if v)

    # ── Node features ──────────────────────────────
    type_to_feat = {
        "Input": 0, "Mul": 1, "Add": 2, "Switch": 3,
        "CmpGeZ": 4, "Output": 5, "Merge": 6,
    }
    feat_dim = 20
    x = torch.zeros(num_nodes, feat_dim)

    ready_set = set(env._ready_set())

    # Global register pressure (same for all nodes in a given step)
    if env.max_registers is not None:
        reg_pressure = len(getattr(env, 'outstanding', set())) / max(env.max_registers, 1)
        reg_pressure = min(1.0, reg_pressure)
    else:
        reg_pressure = 0.0

    for i, name in enumerate(node_names):
        node_obj = graph.nodes.get(name)
        if node_obj is None:
            continue
        feat_idx = type_to_feat.get(node_obj.type, 0)
        x[i, feat_idx] = 1.0

        # Feature 7: is this node ready?
        x[i, 7] = 1.0 if name in ready_set else 0.0

        # Feature 8: has this node been executed?
        x[i, 8] = 1.0 if name in env.executed else 0.0

        # Feature 9: node height (normalized)
        h = heights.get(name, 0)
        x[i, 9] = h / max(max_height, 1)

        # Feature 10: downstream latency sum (normalized)
        dl = downstream.get(name, 0)
        x[i, 10] = dl / max(max_downstream, 1)

        # Feature 11: CPD — critical path distance (normalized)
        c = cpd.get(name, 0)
        x[i, 11] = c / max(max_cpd, 1)

        # Feature 12: register pressure (global, same for all nodes)
        x[i, 12] = reg_pressure

        # Feature 13: Is_Spilled — 1.0 if node's output is in the stack pool
        x[i, 13] = 1.0 if name in env.spilled_nodes else 0.0

        # Feature 14: Remaining_Reloads — consumers still needing this spilled value
        if name in env.spilled_nodes and name in env.node_consumers:
            x[i, 14] = float(env.node_consumers[name])
        else:
            x[i, 14] = 0.0

        # Feature 15: lane_id (normalized by num_lanes)
        lid = lane_id.get(name, -1)
        x[i, 15] = lid / max(num_lanes - 1, 1) if lid >= 0 else 0.0

        # Feature 16: lane_position (normalized by lane_depth)
        lpos = lane_position.get(name, 0)
        ldepth = lane_depth.get(name, 1)
        x[i, 16] = lpos / max(ldepth - 1, 1) if ldepth > 1 else 0.0

        # Feature 17: is_reduction
        x[i, 17] = 1.0 if is_reduction.get(name, False) else 0.0

        # Feature 18: lane_depth (normalized by max_lane_depth)
        x[i, 18] = ldepth / max(max_lane_depth, 1)

        # Feature 19: reduction_position (normalized, 0.0 for non-reduction)
        rpos = reduction_position.get(name, -1)
        x[i, 19] = rpos / max(num_red_nodes - 1, 1) if rpos >= 0 else 0.0

    # ── N-length ready mask (for CPD tracking, backward compat) ──
    n_ready_mask = torch.zeros(len(node_names), dtype=torch.bool)
    for name in ready_set:
        if name in name_to_idx:
            n_ready_mask[name_to_idx[name]] = True

    # ── Action mask: 2*num_nodes-length for flat Issue/Reload + Spill space ──
    # Must cover ALL nodes (including Input/Output) to match the flattened
    # scores tensor [num_nodes, 2] -> [2 * num_nodes], but only op nodes
    # (in topo_order) can be valid actions.
    num_total = len(node_names)
    action_mask = torch.zeros(2 * num_total, dtype=torch.bool)
    for i, nid in enumerate(env.topo_order):
        nidx = name_to_idx[nid]  # position in node_names
        # Issue: unissued + ready + not spilled
        unissued_ready = (nid not in env.executed and
                          nid in ready_set and
                          nid not in env.spilled_nodes)
        # Reload: spilled + not already reloading
        can_reload = (nid in env.spilled_nodes and
                      nid not in env.reload_in_progress)
        action_mask[nidx] = unissued_ready or can_reload

        # Spill: output is live in a register (outstanding + not spilled)
        can_spill = (nid in env.outstanding and
                     nid not in env.spilled_nodes)
        action_mask[num_total + nidx] = can_spill

    return {
        "x": x,
        "edge_index": edge_index,
        "ready_mask": action_mask,  # 2*num_nodes-length for policy
        "n_ready_mask": n_ready_mask,  # node_names-length for CPD tracking
        "node_names": node_names,
        "action_mask": action_mask,
    }


# ═══════════════════════════════════════════════════════════════
#  Scheduling Policy Network
# ═══════════════════════════════════════════════════════════════

class SchedulingPolicy(nn.Module):
    """
    GNN policy for instruction scheduling with flat 2N action space.

    Action head outputs [num_nodes, 2] scores per node (Issue/Reload, Spill).
    Flattened to [2 * num_nodes] and masked with action_mask:
      0..N-1: Issue (if ready + unissued) or Reload (if spilled)
      N..2N-1: Spill (if node's output is live in a register)

    Default: 20-dim node features (15 structural + 5 lane topology),
    256-dim hidden, 4-layer 4-head GAT.
    """

    def __init__(self, node_feat_dim=20, hidden_dim=256, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.gat_convs = nn.ModuleList()
        self.gat_lin = nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
            lin = nn.Linear(hidden_dim, hidden_dim)
            self.gat_convs.append(conv)
            self.gat_lin.append(lin)

        # Priority head: two scores per node (Issue/Reload, Spill)
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def _encode(self, obs):
        h = self.input_proj(obs["x"])
        for conv, lin in zip(self.gat_convs, self.gat_lin):
            h_new = conv(h, obs["edge_index"])
            h_new = F.elu(h_new)
            h_new = lin(h_new)
            h = h + h_new
            h = F.elu(h)
        return h

    def _get_priority_distribution(self, obs):
        """Compute masked 2N action distribution."""
        h = self._encode(obs)
        scores = self.priority_head(h)  # [num_nodes, 2]
        flat_scores = scores.flatten()   # [2 * num_nodes]
        mask = obs.get("action_mask", obs.get("ready_mask"))
        masked = flat_scores.clone()
        masked[~mask] = -float("inf")
        if masked.max() == -float("inf"):
            # All masked — return uniform over any valid actions
            # (shouldn't happen in normal operation)
            valid = mask.float()
            if valid.sum() > 0:
                probs = valid / valid.sum()
            else:
                probs = torch.ones_like(flat_scores) / len(flat_scores)
        else:
            probs = F.softmax(masked, dim=0)
        return probs

    def sample_action(self, obs):
        """Sample a 2N action index."""
        probs = self._get_priority_distribution(obs)
        mask = obs.get("action_mask", obs.get("ready_mask"))

        if not mask.any() or probs.sum() == 0:
            return None, torch.tensor(0.0)

        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob

    def forward(self, obs):
        return self.sample_action(obs)

    def get_entropy(self, obs):
        probs = self._get_priority_distribution(obs)
        mask = obs.get("action_mask", obs.get("ready_mask"))
        if not mask.any() or probs.sum() == 0:
            return torch.tensor(0.0)
        dist = Categorical(probs)
        return dist.entropy()


# ═══════════════════════════════════════════════════════════════
#  Hierarchical Scheduling Policy (GATv2 + Local/Global Attention)
# ═══════════════════════════════════════════════════════════════

BOUNDARY_OUTDEGREE_THRESHOLD = 3  # nodes with >= this many children are "boundary"


class HierarchicalSchedulingPolicy(nn.Module):
    """
    GNN policy with GATv2 dynamic attention and hierarchical masking.

    Architecture:
      - Local branch: 3-layer GATv2 on full dataflow edge_index
        (captures local dependency patterns within 3-hop neighborhoods)
      - Global branch: 3-layer GATv2 on sparse boundary edge_index
        (all nodes <-> boundary nodes with out-degree >= 3, tracking
         global register pressure across subgraph boundaries)
      - Fusion: concat(local_embed, global_embed) -> Linear(2H -> H)
      - Priority head: [num_nodes, 2] scores (Issue/Reload, Spill)

    This mirrors SchedulingPolicy's API exactly: _encode(),
    _get_priority_distribution(), sample_action(), forward(),
    get_entropy(). The observation encoder (encode_scheduling_obs)
    is unchanged — the policy computes boundary edge_index internally
    from the dataflow edge_index.

    Default: 20-dim node features, 256-dim hidden, 3+3 GATv2 layers, 4 heads.
    """

    def __init__(
        self,
        node_feat_dim=20,
        hidden_dim=256,
        num_local_layers=3,
        num_global_layers=3,
        heads=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_local_layers = num_local_layers
        self.num_global_layers = num_global_layers
        head_dim = hidden_dim // heads

        # ── Shared input projection ────────────────────
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # ── Local branch: GATv2 on dataflow edges ─────
        self.local_convs = nn.ModuleList()
        self.local_lin = nn.ModuleList()
        for _ in range(num_local_layers):
            self.local_convs.append(
                GATv2Conv(hidden_dim, head_dim, heads=heads, concat=True)
            )
            self.local_lin.append(nn.Linear(hidden_dim, hidden_dim))

        # ── Global branch: GATv2 on boundary edges ────
        self.global_convs = nn.ModuleList()
        self.global_lin = nn.ModuleList()
        for _ in range(num_global_layers):
            self.global_convs.append(
                GATv2Conv(hidden_dim, head_dim, heads=heads, concat=True)
            )
            self.global_lin.append(nn.Linear(hidden_dim, hidden_dim))

        # ── Fusion: local || global -> hidden ─────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # ── Priority head: 2 scores per node ──────────
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def _compute_boundary_edge_index(self, edge_index, num_nodes):
        """
        Build the global bipartite edge index for boundary attention.

        Boundary nodes are those with out-degree >= BOUNDARY_OUTDEGREE_THRESHOLD
        (default 3). These are the "spill points" — nodes whose output
        feeds many consumers and creates register pressure.

        The bipartite edge_index connects:
          - All nodes -> boundary nodes  (boundary nodes aggregate global state)
          - Boundary nodes -> all nodes  (everyone receives global signal)

        Returns a [2, E] tensor. If no boundary nodes exist (tiny graph),
        falls back to the top-3 highest-out-degree nodes.

        Re-computation is O(N+E), negligible relative to the GATv2 forward pass.
        """
        # Compute out-degree from edge_index
        out_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        unique_src, counts = torch.unique(edge_index[0], return_counts=True)
        out_degree[unique_src] = counts

        # Boundary nodes: out-degree >= threshold
        boundary_mask = out_degree >= BOUNDARY_OUTDEGREE_THRESHOLD
        boundary_idx = torch.where(boundary_mask)[0]

        if len(boundary_idx) == 0:
            # Fallback: use top-3 highest out-degree nodes
            k = min(3, num_nodes)
            _, top_indices = torch.topk(out_degree, k)
            boundary_idx = top_indices

        # Build bipartite: all_nodes <-> boundary_nodes
        all_idx = torch.arange(num_nodes, device=edge_index.device)

        # Direction 1: all -> boundary (boundary aggregates)
        src_a2b = all_idx.repeat_interleave(len(boundary_idx))
        dst_a2b = boundary_idx.repeat(len(all_idx))

        # Direction 2: boundary -> all (distribute global signal)
        src_b2a = boundary_idx.repeat_interleave(len(all_idx))
        dst_b2a = all_idx.repeat(len(boundary_idx))

        return torch.stack([
            torch.cat([src_a2b, src_b2a]),
            torch.cat([dst_a2b, dst_b2a]),
        ], dim=0)

    def get_branch_norms(self, obs):
        """
        Compute L2 norms of local and global branch embeddings for logging.

        Returns dict with:
          - "local_norm": mean L2 norm of local branch node embeddings
          - "global_norm": mean L2 norm of global branch node embeddings
          - "norm_ratio": global_norm / (local_norm + 1e-8)

        The norm_ratio tracks whether the global attention heads are active
        relative to the local branch. A ratio near 0 means the global branch
        is dormant; a ratio >> 1 means it's dominating.
        """
        with torch.no_grad():
            _, h_local, h_global = self._encode(obs, return_branch_embeddings=True)
            local_norm = h_local.norm(dim=-1).mean().item()
            global_norm = h_global.norm(dim=-1).mean().item()

        return {
            "local_norm": local_norm,
            "global_norm": global_norm,
            "norm_ratio": global_norm / (local_norm + 1e-8),
        }

    def get_reduction_attention(self, obs):
        """
        Compute the global branch's embedding concentration on reduction nodes.

        Splits node embeddings into reduction (obs['x'][:, 17] == 1.0) vs
        non-reduction groups and returns the ratio of their mean L2 norms
        in the global branch output. A ratio > 1.0 means the global branch
        is concentrating on reduction/synchronization boundaries.

        Returns dict with:
          - "red_norm": mean global L2 norm for reduction nodes
          - "nonred_norm": mean global L2 norm for non-reduction nodes
          - "red_attn_ratio": red_norm / (nonred_norm + 1e-8)
        """
        with torch.no_grad():
            _, _, h_global = self._encode(obs, return_branch_embeddings=True)
            is_red = obs["x"][:, 17] > 0.5  # feature 17 = is_reduction
            norms = h_global.norm(dim=-1)
            if is_red.any():
                red_norm = norms[is_red].mean().item()
            else:
                red_norm = 0.0
            if (~is_red).any():
                nonred_norm = norms[~is_red].mean().item()
            else:
                nonred_norm = 0.0

        return {
            "red_global_norm": red_norm,
            "nonred_global_norm": nonred_norm,
            "red_attn_ratio": red_norm / (nonred_norm + 1e-8),
        }

    def _encode(self, obs, return_branch_embeddings=False):
        """
        Encode node features through parallel local + global GATv2 branches,
        then fuse via concatenation.

        When return_branch_embeddings=True, returns (h_fused, h_local, h_global)
        for per-mask-type logging. Default: returns just h_fused.
        """
        x = obs["x"]
        edge_index = obs["edge_index"]
        num_nodes = x.size(0)

        h = self.input_proj(x)

        # ── Local branch: operate on dataflow edges ───
        h_local = h
        for conv, lin in zip(self.local_convs, self.local_lin):
            h_new = conv(h_local, edge_index)
            h_new = F.elu(h_new)
            h_new = lin(h_new)
            h_local = h_local + h_new
            h_local = F.elu(h_local)

        # ── Global branch: operate on boundary edges ──
        global_edge_index = self._compute_boundary_edge_index(
            edge_index, num_nodes
        )
        h_global = h
        for conv, lin in zip(self.global_convs, self.global_lin):
            h_new = conv(h_global, global_edge_index)
            h_new = F.elu(h_new)
            h_new = lin(h_new)
            h_global = h_global + h_new
            h_global = F.elu(h_global)

        # ── Fuse: concat local + global -> project ──
        h_fused = self.fusion(torch.cat([h_local, h_global], dim=-1))

        if return_branch_embeddings:
            return h_fused, h_local, h_global
        return h_fused

    def _get_priority_distribution(self, obs):
        """Compute masked 2N action distribution (same API as SchedulingPolicy)."""
        h = self._encode(obs)
        scores = self.priority_head(h)  # [num_nodes, 2]
        flat_scores = scores.flatten()   # [2 * num_nodes]
        mask = obs.get("action_mask", obs.get("ready_mask"))
        masked = flat_scores.clone()
        masked[~mask] = -float("inf")
        if masked.max() == -float("inf"):
            valid = mask.float()
            if valid.sum() > 0:
                probs = valid / valid.sum()
            else:
                probs = torch.ones_like(flat_scores) / max(len(flat_scores), 1)
        else:
            probs = F.softmax(masked, dim=0)
        return probs

    def sample_action(self, obs):
        """Sample a 2N action index."""
        probs = self._get_priority_distribution(obs)
        mask = obs.get("action_mask", obs.get("ready_mask"))

        if not mask.any() or probs.sum() == 0:
            return None, torch.tensor(0.0)

        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob

    def forward(self, obs):
        return self.sample_action(obs)

    def get_entropy(self, obs):
        probs = self._get_priority_distribution(obs)
        mask = obs.get("action_mask", obs.get("ready_mask"))
        if not mask.any() or probs.sum() == 0:
            return torch.tensor(0.0)
        dist = Categorical(probs)
        return dist.entropy()


# ═══════════════════════════════════════════════════════════════
#  GNN Policy Network
# ═══════════════════════════════════════════════════════════════

class GNNPolicy(nn.Module):
    """
    GNN policy with expanded action space for routing types.

    Action head outputs [num_nodes, NUM_SLOTS * NUM_ROUTING_TYPES] = [num_nodes, 6]
    scores per node. Flattened to [num_nodes * 6] and masked.
    """

    def __init__(self, node_feat_dim=9, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.gat_convs = nn.ModuleList()
        self.gat_lin = nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
            lin = nn.Linear(hidden_dim, hidden_dim)
            self.gat_convs.append(conv)
            self.gat_lin.append(lin)

        # Action head: [num_nodes, 6] = scores for (slot0_Always, slot0_IfTrue,
        # slot0_IfFalse, slot1_Always, slot1_IfTrue, slot1_IfFalse)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_SLOTS * NUM_ROUTING_TYPES),
        )

    def _encode(self, obs):
        """Shared GAT encoding for forward and get_entropy."""
        h = self.input_proj(obs["x"])
        for conv, lin in zip(self.gat_convs, self.gat_lin):
            h_new = conv(h, obs["edge_index"])
            h_new = F.elu(h_new)
            h_new = lin(h_new)
            h = h + h_new
            h = F.elu(h)
        return h

    def _get_action_distribution(self, obs):
        """Compute masked action distribution from observation."""
        h = self._encode(obs)
        scores = self.action_head(h)  # [num_nodes, 6]
        flat_scores = scores.flatten()  # [num_nodes * 6]
        masked_scores = flat_scores.clone()
        masked_scores[~obs["valid_mask"]] = -float("inf")
        probs = F.softmax(masked_scores, dim=0)
        return probs

    def sample_action(self, obs):
        """Sample a single action from the distribution."""
        probs = self._get_action_distribution(obs)

        if not obs["valid_mask"].any() or probs.sum() == 0:
            return None, torch.tensor(0.0)

        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob

    def forward(self, obs):
        """
        Produce one action. For Switch nodes, this is called ONCE per routing type.
        The env handles calling forward() multiple times for Switch.
        """
        return self.sample_action(obs)

    def get_entropy(self, obs):
        """Entropy of the action distribution."""
        probs = self._get_action_distribution(obs)
        if not obs["valid_mask"].any() or probs.sum() == 0:
            return torch.tensor(0.0)
        dist = Categorical(probs)
        return dist.entropy()


# ═══════════════════════════════════════════════════════════════
#  Action Decoder
# ═══════════════════════════════════════════════════════════════

ROUTING_NAMES = ["send", "send_true", "send_false"]


def decode_action(action_idx, node_names, override_routing_type=None):
    """
    Convert flat action index into env action format.

    action_idx: target_node * 6 + slot * 3 + routing_type
    Returns:    {'target_destinations': [(target, slot, routing_str)]}

    If override_routing_type is set (e.g., for Switch nodes), force that routing type.
    """
    if action_idx is None:
        return {"target_destinations": []}

    num_nodes = len(node_names)
    target_node_idx = action_idx // ACTIONS_PER_NODE
    remaining = action_idx % ACTIONS_PER_NODE
    slot = remaining // NUM_ROUTING_TYPES
    rt_idx = remaining % NUM_ROUTING_TYPES

    if override_routing_type is not None:
        rt_idx = override_routing_type

    if target_node_idx >= num_nodes:
        return {"target_destinations": []}

    target_name = node_names[target_node_idx]
    rt_str = ROUTING_NAMES[rt_idx]
    return {"target_destinations": [(target_name, slot, rt_str)]}


def collect_switch_routings(obs, policy, device="cpu"):
    """
    For a Switch node, produce BOTH the IfTrue and IfFalse routings.
    Called by the training loop when env's current node is a Switch.

    IMPORTANT: The valid_mask is filtered per routing type BEFORE sampling,
    so the log_prob correctly reflects the probability of the action that
    was actually taken. Do NOT use override_routing_type here — that
    misaligns the gradient signal (the log_prob would reflect a different
    action than what the env receives).

    Returns an action dict with two target_destinations.
    """
    obs_on_device = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in obs.items()
    }

    node_names = obs["node_names"]
    full_mask = obs_on_device["valid_mask"]

    # ── IfTrue: mask to only routing_type 1 ──────────
    # Each action index = target * 6 + slot * 3 + routing_type
    # routing_type occupies the lowest 2 bits (indices 0,1,2 of each group of 3)
    mask_true = full_mask.clone()
    for i in range(len(mask_true)):
        if mask_true[i]:
            rt = i % NUM_ROUTING_TYPES
            if rt != 1:  # only keep routing_type 1 (send_true)
                mask_true[i] = False

    # ── IfFalse: mask to only routing_type 2 ─────────
    mask_false = full_mask.clone()
    for i in range(len(mask_false)):
        if mask_false[i]:
            rt = i % NUM_ROUTING_TYPES
            if rt != 2:  # only keep routing_type 2 (send_false)
                mask_false[i] = False

    # --- Sample IfTrue routing ---
    obs_true = dict(obs_on_device)
    obs_true["valid_mask"] = mask_true
    idx1, lp1 = policy.sample_action(obs_true)
    action = decode_action(idx1, node_names)

    # --- Sample IfFalse routing ---
    obs_false = dict(obs_on_device)
    obs_false["valid_mask"] = mask_false
    idx2, lp2 = policy.sample_action(obs_false)
    action2 = decode_action(idx2, node_names)

    # Combine both routings
    action["target_destinations"].extend(action2["target_destinations"])

    return action, (lp1 + lp2)
