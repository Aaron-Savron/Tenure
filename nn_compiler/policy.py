"""
policy.py -- GNN policy with expanded action space supporting routing types.

Action space: num_nodes * 2 slots * 3 routing types = num_nodes * 6
  Routing types: 0=send (Always), 1=send_true, 2=send_false
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATConv


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
#  Scheduling Observation Encoder
# ═══════════════════════════════════════════════════════════════

def encode_scheduling_obs(env):
    """
    Convert the scheduling environment state into GNN-ready tensors.

    Node features [num_nodes, 13]:
      [0..6] = one-hot opcode: Input, Mul, Add, Switch, CmpGeZ, Output, Merge
      [7]    = is this node ready to fire?
      [8]    = has this node been executed already?
      [9]    = node height (longest hop path to output, normalized)
      [10]   = downstream latency sum (total descendant work, normalized)
      [11]   = CPD: critical path distance (longest latency path to output,
               normalized by max CPD across all nodes)
      [12]   = register pressure: outstanding_values / max_registers (capped at 1.0),
               0.0 when max_registers is not set (unlimited)

    Ready mask: [num_nodes] boolean. Only op nodes (not Input/Output)
                that are ready and not yet executed are valid actions.
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

    # ── Node features ──────────────────────────────
    type_to_feat = {
        "Input": 0, "Mul": 1, "Add": 2, "Switch": 3,
        "CmpGeZ": 4, "Output": 5, "Merge": 6,
    }
    feat_dim = 13
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

    # Ready mask: only op nodes that are ready (not executed, all deps met)
    ready_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for name in ready_set:
        if name in name_to_idx:
            ready_mask[name_to_idx[name]] = True

    return {
        "x": x,
        "edge_index": edge_index,
        "ready_mask": ready_mask,
        "node_names": node_names,
    }


# ═══════════════════════════════════════════════════════════════
#  Scheduling Policy Network
# ═══════════════════════════════════════════════════════════════

class SchedulingPolicy(nn.Module):
    """
    GNN policy for instruction scheduling.

    Action head outputs [num_nodes, 1] priority scores per node.
    The ready mask is applied before softmax to restrict choices
    to nodes whose dependencies are satisfied.

    Default: 13-dim node features (12 structural + register pressure),
    256-dim hidden, 4-layer 4-head GAT.
    """

    def __init__(self, node_feat_dim=13, hidden_dim=256, num_layers=4):
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

        # Priority head: one scalar per node
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
        """Compute masked priority distribution over nodes."""
        h = self._encode(obs)
        scores = self.priority_head(h).squeeze(-1)  # [num_nodes]
        masked = scores.clone()
        masked[~obs["ready_mask"]] = -float("inf")
        probs = F.softmax(masked, dim=0)
        return probs

    def sample_action(self, obs):
        """Sample a node from the ready set."""
        probs = self._get_priority_distribution(obs)

        if not obs["ready_mask"].any() or probs.sum() == 0:
            return None, torch.tensor(0.0)

        dist = Categorical(probs)
        node_idx = dist.sample()
        log_prob = dist.log_prob(node_idx)
        return node_idx.item(), log_prob

    def forward(self, obs):
        return self.sample_action(obs)

    def get_entropy(self, obs):
        probs = self._get_priority_distribution(obs)
        if not obs["ready_mask"].any() or probs.sum() == 0:
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
