"""
train.py -- REINFORCE training loop for the GNN compiler policy.

Trains the GNN policy to route dataflow nodes optimally by maximizing
reward from the Rust VM (minimizing cycles and queue depth).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from .compiler_env import (
    create_dot_product_graph,
    create_relu_graph,
    DataflowGymEnv,
    SchedulingGymEnv,
    DEFAULT_LATENCY,
    VMBridge,
)
from .policy import (
    GNNPolicy,
    SchedulingPolicy,
    HierarchicalSchedulingPolicy,
    encode_observation,
    encode_scheduling_obs,
    decode_action,
    collect_switch_routings,
)
from typing import Dict
from .checkpoint_surgery import transplant_weights
from .scheduler_baseline import schedule_cpf


class RunningBaseline:
    """Exponential moving average baseline for variance reduction."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, reward):
        if not self.initialized:
            self.value = reward
            self.initialized = True
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * reward

    def get(self):
        return self.value


def collect_episode(env, policy, device="cpu"):
    """
    Run one episode, recording actions and log_probs.
    Returns the final reward, log_probs list, and entropies list.

    The reward is captured from the terminal step (single pass through
    the episode), eliminating the double-VM-call overhead.
    """
    env.reset()
    log_probs = []
    entropies = []
    actions = []
    final_reward = 0.0
    done = False

    while not done:
        obs = encode_observation(env)
        obs_on_device = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in obs.items()
        }

        # Switch nodes need dual routing (IfTrue + IfFalse)
        if obs.get("is_switch", False):
            action, log_prob = collect_switch_routings(obs, policy, device)
            entropy = policy.get_entropy(obs_on_device)  # avg entropy per sample
        else:
            action_idx, log_prob = policy(obs_on_device)
            action = decode_action(action_idx, obs["node_names"])
            entropy = policy.get_entropy(obs_on_device)

        actions.append(action)
        log_probs.append(log_prob)
        entropies.append(entropy)

        _, reward, done, info = env.step(action)
        final_reward = reward if done else 0.0  # only terminal step has real reward

    return final_reward, log_probs, entropies, info


def train_episode(env, policy, optimizer, baseline, entropy_coef=0.01,
                  reward_scale=1.0, device="cpu"):
    """Run one training episode: collect, compute loss, update."""

    reward, log_probs, entropies, info = collect_episode(env, policy, device)

    scaled_reward = reward * reward_scale
    advantage = scaled_reward - baseline.get()
    baseline.update(scaled_reward)

    policy_loss = -torch.stack(log_probs).sum() * advantage

    if len(entropies) > 0:
        entropy_bonus = -entropy_coef * torch.stack(entropies).mean()
    else:
        entropy_bonus = torch.tensor(0.0)

    loss = policy_loss + entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return reward, info


def collect_scheduling_episode(env, policy, device="cpu"):
    """
    Run one scheduling episode: at each step, pick a ready node by priority.
    Returns total cycles (terminal reward), log_probs list, and entropies list.
    """
    env.reset()
    log_probs = []
    entropies = []
    final_reward = 0.0
    final_info = {}

    while not env.done:
        obs = encode_scheduling_obs(env)
        obs_on_device = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in obs.items()
        }

        node_idx, log_prob = policy(obs_on_device)
        if node_idx is None:
            partial_credit = getattr(env, '_accumulated_partial_credit', 0.0)
            final_reward = -1000.0 + partial_credit
            final_info = {"error": "No valid actions (empty ready set)",
                          "partial_credit": partial_credit}

        node_id = obs["node_names"][node_idx]
        log_probs.append(log_prob)
        entropies.append(policy.get_entropy(obs_on_device))

        _, reward, done, info = env.step(node_id)
        if done:
            final_reward = reward
            final_info = info

    return final_reward, log_probs, entropies, final_info


def train_scheduling_episode(env, policy, optimizer, baseline,
                              entropy_coef=0.01, reward_scale=1.0, device="cpu"):
    """Run one scheduling training episode."""
    reward, log_probs, entropies, info = collect_scheduling_episode(
        env, policy, device
    )

    scaled_reward = reward * reward_scale
    advantage = scaled_reward - baseline.get()
    baseline.update(scaled_reward)

    if len(log_probs) == 0:
        return reward, info

    policy_loss = -torch.stack(log_probs).sum() * advantage

    if len(entropies) > 0:
        entropy_bonus = -entropy_coef * torch.stack(entropies).mean()
    else:
        entropy_bonus = torch.tensor(0.0)

    loss = policy_loss + entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return reward, info


def collect_scheduling_episode_ppo(env, policy, device="cpu"):
    """
    Run one scheduling episode, recording old_log_probs (detached) and
    observations so PPO can replay them for K epochs.

    Returns:
        final_reward, old_log_probs (list of detached tensors),
        observations (list of obs dicts), actions (list of node_idx ints),
        final_info (dict)
    """
    env.reset()
    old_log_probs = []
    observations = []
    actions = []
    final_reward = 0.0
    final_info = {}

    while not env.done:
        obs = encode_scheduling_obs(env)
        obs_on_device = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in obs.items()
        }

        node_idx, log_prob = policy(obs_on_device)
        if node_idx is None:
            partial_credit = getattr(env, '_accumulated_partial_credit', 0.0)
            final_reward = -1000.0 + partial_credit
            final_info = {"error": "No valid actions (empty ready set)",
                          "partial_credit": partial_credit}

        node_id = obs["node_names"][node_idx]
        old_log_probs.append(log_prob.detach())
        observations.append(obs_on_device)
        actions.append(node_idx)

        _, reward, done, info = env.step(node_id)
        if done:
            final_reward = reward
            final_info = info

    return final_reward, old_log_probs, observations, actions, final_info


def train_scheduling_episode_ppo(env, policy, optimizer, baseline,
                                  entropy_coef=0.01, reward_scale=1.0,
                                  clip_epsilon=0.2, ppo_epochs=3, device="cpu"):
    """
    Run one PPO scheduling training episode.

    Uses the clipped surrogate objective to bound policy updates:
        L = -min(r * A, clip(r, 1-ε, 1+ε) * A)
    where r = π_new(a|s) / π_old(a|s).

    The episode data is replayed for ppo_epochs gradient steps before
    moving to the next episode, preventing catastrophic distribution collapse.
    """
    reward, old_log_probs, observations, actions, info = \
        collect_scheduling_episode_ppo(env, policy, device)

    scaled_reward = reward * reward_scale
    advantage = scaled_reward - baseline.get()
    baseline.update(scaled_reward)

    n_steps = len(old_log_probs)
    if n_steps == 0:
        return reward, info

    # PPO: run K epochs on the collected episode data.
    # Each epoch recomputes log_probs under the evolving policy and clips
    # the ratio to prevent any single step from collapsing the distribution.
    for epoch in range(ppo_epochs):
        epoch_losses = []
        for obs, old_lp, action_idx in zip(
            observations, old_log_probs, actions
        ):
            # Recompute log_prob under CURRENT (updated) policy
            probs = policy._get_priority_distribution(obs)
            if not obs["ready_mask"].any() or probs.sum() == 0:
                continue
            dist = Categorical(probs)
            action_tensor = torch.tensor(action_idx, device=device)
            new_lp = dist.log_prob(action_tensor)

            # PPO clipped surrogate objective
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
            policy_loss = -torch.minimum(surr1, surr2)

            # Entropy bonus (uses the new distribution's entropy, not the old one)
            entropy = dist.entropy()
            entropy_bonus = -entropy_coef * entropy

            epoch_losses.append(policy_loss + entropy_bonus)

        if not epoch_losses:
            continue

        loss = torch.stack(epoch_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    return reward, info


def ppo_schedule_train(
    env_factory,
    policy,
    optimizer,
    num_episodes=500,
    entropy_coef=0.01,
    clip_epsilon=0.2,
    ppo_epochs=3,
    log_every=50,
    reward_scale=1.0,
    device="cpu",
):
    """
    Full PPO training loop for instruction scheduling.

    Args:
        env_factory: callable that returns a fresh SchedulingGymEnv.
                     Called each episode so the graph can rotate.
        policy: SchedulingPolicy instance.
        optimizer: torch optimizer.
        num_episodes: total episodes to train.
        entropy_coef: entropy bonus coefficient.
        clip_epsilon: PPO clip range (±ε around ratio 1.0).
        ppo_epochs: gradient steps per collected episode.
        log_every: print metrics every N episodes.
        reward_scale: multiplier applied to raw reward before advantage calc.
        device: "cpu" or "cuda".
    """
    policy.train()
    baseline = RunningBaseline(alpha=0.9)

    episode_rewards = []
    best_reward = -float("inf")
    deadlock_count = 0

    try:
        from tqdm import tqdm
        progress = tqdm(range(num_episodes))
    except ImportError:
        progress = range(num_episodes)

    for episode in progress:
        env = env_factory()
        reward, info = train_scheduling_episode_ppo(
            env, policy, optimizer, baseline,
            entropy_coef=entropy_coef,
            reward_scale=reward_scale,
            clip_epsilon=clip_epsilon,
            ppo_epochs=ppo_epochs,
            device=device,
        )

        episode_rewards.append(reward)

        if reward > best_reward:
            best_reward = reward
        if reward <= -900:
            deadlock_count += 1

        # Logging
        if (episode + 1) % log_every == 0:
            recent = episode_rewards[-log_every:]
            avg = sum(recent) / len(recent) if recent else 0.0
            recent_dead = sum(1 for r in recent if r <= -900)
            print(
                f"Ep {episode + 1:4d} | "
                f"avg: {avg:7.1f} | "
                f"best: {best_reward:7.1f} | "
                f"dead: {recent_dead:2d}/{log_every} | "
                f"bl: {baseline.get():.1f}"
            )
        elif hasattr(progress, "set_postfix"):
            recent = episode_rewards[-50:]
            avg = sum(recent) / len(recent) if recent else 0.0
            progress.set_postfix({
                "avg": f"{avg:.1f}",
                "best": f"{best_reward:.1f}",
                "bl": f"{baseline.get():.1f}",
            })

    return episode_rewards


def train(
    env,
    policy,
    optimizer,
    num_episodes=500,
    entropy_coef=0.05,
    log_every=50,
    reward_scale=1.0,
    device="cpu",
):
    """Full REINFORCE training loop."""
    policy.train()
    baseline = RunningBaseline(alpha=0.9)

    episode_rewards = []
    episode_steps = []
    best_reward = -float("inf")

    try:
        from tqdm import tqdm
        progress = tqdm(range(num_episodes))
    except ImportError:
        progress = range(num_episodes)

    for episode in progress:
        reward, info = train_episode(
            env, policy, optimizer, baseline,
            entropy_coef, reward_scale, device
        )

        episode_rewards.append(reward)
        episode_steps.append(len(env.topo_order))

        if reward > best_reward:
            best_reward = reward

        # Logging
        if isinstance(progress, range):
            if (episode + 1) % log_every == 0:
                recent = episode_rewards[-log_every:]
                avg = sum(recent) / len(recent)
                deadlocks = sum(1 for r in recent if r <= -900)
                print(
                    f"Ep {episode + 1:4d} | "
                    f"avg: {avg:7.1f} | "
                    f"best: {best_reward:7.1f} | "
                    f"deadlocks: {deadlocks:2d} | "
                    f"bl: {baseline.get():.1f}"
                )
        else:
            recent = episode_rewards[-50:]
            avg = sum(recent) / len(recent) if recent else 0.0
            progress.set_postfix({
                "avg": f"{avg:.1f}",
                "best": f"{best_reward:.1f}",
                "bl": f"{baseline.get():.1f}",
            })

    return episode_rewards, episode_steps


# ═══════════════════════════════════════════════════════════════
#  v2: Active-Spilling PPO Collection and Training
# ═══════════════════════════════════════════════════════════════

def collect_scheduling_episode_ppo_v2(env, policy, device="cpu"):
    """
    v2 PPO episode collection for the flat 2N action space.

    Unlike v1, the policy returns an action_idx in [0, 2N-1]
    that maps directly to env.step(action_idx). No node name
    lookup needed.

    Returns:
        final_reward, old_log_probs, observations, actions (int), final_info
    """
    env.reset()
    old_log_probs = []
    observations = []
    actions = []
    final_reward = 0.0
    final_info = {}

    while not env.done:
        obs = encode_scheduling_obs(env)
        obs_on_device = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in obs.items()
        }

        action_idx, log_prob = policy(obs_on_device)
        if action_idx is None:
            # Include accumulated partial credit so PPO sees gradient signal
            # even when the policy deadlocks mid-episode.
            partial_credit = getattr(env, '_accumulated_partial_credit', 0.0)
            final_reward = -1000.0 + partial_credit
            final_info = {"error": "No valid actions (empty action mask)",
                          "partial_credit": partial_credit}

        old_log_probs.append(log_prob.detach())
        observations.append(obs_on_device)
        actions.append(action_idx)

        _, reward, done, info = env.step(action_idx)
        if done:
            final_reward = reward
            final_info = info

    return final_reward, old_log_probs, observations, actions, final_info


def train_scheduling_episode_ppo_v2(env, policy, optimizer, baseline,
                                     entropy_coef=0.01, reward_scale=1.0,
                                     clip_epsilon=0.2, ppo_epochs=3,
                                     device="cpu", track_kl: bool = False):
    """
    v2 PPO training step for the flat 2N action space.

    Uses the clipped surrogate objective on the unified Issue/Reload/Spill
    distribution. The GAT body may be frozen (requires_grad=False) — this
    function works correctly either way since .backward() only flows through
    trainable parameters.

    When track_kl=True, computes approximate KL divergence between the
    old and new policy distributions after the final PPO epoch and returns
    it via info['approx_kl']. The KL is computed as:
        KL ≈ mean(exp(new_lp - old_lp) - 1 - (new_lp - old_lp))
    which is the second-order Taylor expansion of KL, numerically stable
    for small divergences.
    """
    reward, old_log_probs, observations, actions, info = \
        collect_scheduling_episode_ppo_v2(env, policy, device)

    scaled_reward = reward * reward_scale
    advantage = scaled_reward - baseline.get()
    baseline.update(scaled_reward)

    n_steps = len(old_log_probs)
    if n_steps == 0:
        return reward, info

    # Track KL across all steps in the final epoch
    final_epoch_kls: list = [] if track_kl else None

    for epoch in range(ppo_epochs):
        epoch_losses = []
        if track_kl:
            epoch_kls = []

        for obs, old_lp, act in zip(observations, old_log_probs, actions):
            probs = policy._get_priority_distribution(obs)
            mask = obs.get("action_mask", obs.get("ready_mask"))
            if not mask.any() or probs.sum() == 0:
                continue
            dist = Categorical(probs)
            action_tensor = torch.tensor(act, device=device)
            new_lp = dist.log_prob(action_tensor)

            # Approximate KL: π_old * log(π_old/π_new)
            # Use the numerically stable form: exp(d) - 1 - d where d = new_lp - old_lp
            if track_kl:
                log_ratio = new_lp - old_lp
                kl_step = torch.exp(log_ratio) - 1.0 - log_ratio
                epoch_kls.append(kl_step)

            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon,
                                1.0 + clip_epsilon) * advantage
            policy_loss = -torch.minimum(surr1, surr2)

            entropy = dist.entropy()
            entropy_bonus = -entropy_coef * entropy
            epoch_losses.append(policy_loss + entropy_bonus)

        if not epoch_losses:
            continue

        loss = torch.stack(epoch_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        if track_kl and epoch_kls:
            final_epoch_kls = epoch_kls  # keep last epoch's KL for logging

    if track_kl and final_epoch_kls:
        info["approx_kl"] = torch.stack(final_epoch_kls).mean().item()

    # ── Per-branch embedding norm logging (hierarchical policy only) ──
    if isinstance(policy, HierarchicalSchedulingPolicy) and observations:
        try:
            branch_norms = policy.get_branch_norms(observations[0])
            info["branch_local_norm"] = branch_norms["local_norm"]
            info["branch_global_norm"] = branch_norms["global_norm"]
            info["branch_norm_ratio"] = branch_norms["norm_ratio"]
            red_attn = policy.get_reduction_attention(observations[0])
            info["red_global_norm"] = red_attn["red_global_norm"]
            info["nonred_global_norm"] = red_attn["nonred_global_norm"]
            info["red_attn_ratio"] = red_attn["red_attn_ratio"]
        except Exception:
            pass  # best-effort; don't fail training over logging

    return reward, info


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Frozen Adaptation Training
# ═══════════════════════════════════════════════════════════════

def freeze_gat_body(policy):
    """
    Freeze GAT body parameters while keeping input_proj and
    priority_head trainable. Returns the list of trainable param names.
    """
    trainable = []
    frozen = 0
    for name, param in policy.named_parameters():
        if name.startswith("gat_convs.") or name.startswith("gat_lin."):
            param.requires_grad = False
            frozen += 1
        else:
            param.requires_grad = True
            trainable.append(name)
    print(f"  Frozen: {frozen} params (GAT body)")
    print(f"  Trainable: {len(trainable)} param groups: {trainable}")
    return trainable


def unfreeze_gat_body(policy):
    """Unfreeze all parameters (for Phase 2+)."""
    for param in policy.parameters():
        param.requires_grad = True
    print("  Unfrozen: all parameters are now trainable")





# ── Graph corpus for frozen adaptation training ──────────
def _build_frozen_corpus():
    """Build a list of (label, graph_fn) pairs for Phase 1 training.
    Import is deferred to avoid circular imports at module level."""
    from .dl_subgraphs import create_mlp_block, create_residual_add, create_layer_norm
    return [
        ("MLP (1L, W=3)", lambda: create_mlp_block(1, 3)),
        ("MLP (2L, W=2)", lambda: create_mlp_block(2, 2)),
        ("MLP (3L, W=2)", lambda: create_mlp_block(3, 2)),
        ("ResAdd (3L, d2)", lambda: create_residual_add(3, 2)),
        ("ResAdd (4L, d2)", lambda: create_residual_add(4, 2)),
        ("LayerNorm (4ch)", lambda: create_layer_norm(4)),
    ]


def frozen_adaptation_train(
    checkpoint_path: str = "checkpoints/production_v2_init.pt",
    num_episodes: int = 500,
    k: int = 2,
    max_registers: int = 3,
    register_penalty_alpha: float = 1.0,
    mem_latency: int = 3,
    learning_rate: float = 5e-4,
    entropy_coef: float = 0.05,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 3,
    log_every: int = 50,
    save_every: int = 500,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
    from_scratch: bool = False,
) -> dict:
    """
    Phase 1: Frozen Adaptation Training.

    When from_scratch=False (default): loads the surgically-transplanted
    v2_init checkpoint, freezes the GAT body, and trains ONLY input_proj
    and priority_head on a corpus of small DL subgraphs.

    When from_scratch=True: creates a random-initialized SchedulingPolicy
    (15-dim), freezes the GAT body, and trains from scratch. This avoids
    the v1 weight transfer entirely — the GAT starts with random attention
    maps and must learn spill-aware representations from the beginning.

    The model learns the syntax of the 2N action space (Issue vs Spill)
    without risking corruption of the pre-trained GAT attention maps.
    """
    import os
    from itertools import cycle

    print("=" * 66)
    if from_scratch:
        print("  PHASE 1 (FROM SCRATCH): FROZEN ADAPTATION")
        print("  Random-init GAT body -> train new 2N action head")
    else:
        print("  PHASE 1: FROZEN ADAPTATION (v1 transplant)")
        print("  Transplant v1 GAT body -> train new 2N action head")
    print("=" * 66)

    # Build corpus (deferred import to avoid circular deps)
    corpus = _build_frozen_corpus()

    # ── Build policy ────────────────────────────────
    policy = SchedulingPolicy(node_feat_dim=20).to(device)

    if from_scratch:
        # Random init — no weight surgery
        print(f"  Random-init SchedulingPolicy(15-dim)")
        print(f"  Policy params: {sum(p.numel() for p in policy.parameters())}")
    else:
        # Weight surgery from v1 checkpoint
        transplant_weights(
            "nn_compiler/production_v1.pt", policy,
            device=device,
            output_path=checkpoint_path,
        )
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Policy params: {sum(p.numel() for p in policy.parameters())}")

    # ── Freeze GAT body ─────────────────────────────
    trainable = freeze_gat_body(policy)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    print(f"  Optimizer: {len(optimizer.param_groups[0]['params'])} trainable params")
    print(f"  mem_latency={mem_latency}, K={k}, regs={max_registers}, "
          f"alpha={register_penalty_alpha}")
    print(f"  entropy={entropy_coef}, clip={clip_epsilon}, PPO epochs={ppo_epochs}")
    print(f"  Corpus: {len(corpus)} graph types")
    print(f"  Init: {'from scratch (random)' if from_scratch else 'v1 transplant'}")
    print()

    # ── Training state ──────────────────────────────
    # Per-graph baselines for clean advantage signals
    baselines: Dict[str, RunningBaseline] = {}
    episode_rewards = []
    episode_spills = []
    episode_reloads = []
    best_reward = -float("inf")
    deadlock_count = 0
    n_cpf_beaten = 0

    graph_cycle = cycle(corpus)

    try:
        from tqdm import tqdm
        progress = tqdm(range(num_episodes), desc="Frozen Adaptation")
    except ImportError:
        progress = range(num_episodes)

    for episode in progress:
        label, g_fn = next(graph_cycle)
        graph = g_fn()
        topo = graph.get_topological_order()
        node_names = graph.inputs + topo + graph.outputs

        # ── CPF baseline ────────────────────────────
        cpf_sched = schedule_cpf(graph)
        cpf_env = SchedulingGymEnv(
            graph, None, [],
            max_exec_units=k, latency=DEFAULT_LATENCY,
            max_registers=max_registers,
            register_penalty_alpha=register_penalty_alpha,
            mem_latency=mem_latency,
        )
        cpf_env.reset()
        for nid in cpf_sched:
            cpf_idx = node_names.index(nid)
            _, _, cpf_done, cpf_info = cpf_env.step(cpf_idx)
            if cpf_done:
                break
        cpf_reward = cpf_info.get("spill_penalty", 0) + cpf_info.get("cycles", 0)
        cpf_reward = -cpf_reward

        # ── v2 training env ──────────────────────────
        env = SchedulingGymEnv(
            graph, None, [],
            max_exec_units=k, latency=DEFAULT_LATENCY,
            max_registers=max_registers,
            register_penalty_alpha=register_penalty_alpha,
            mem_latency=mem_latency,
        )

        # ── Per-label baseline ────────────────────────
        if label not in baselines:
            baselines[label] = RunningBaseline(alpha=0.9)
        graph_baseline = baselines[label]

        reward, info = train_scheduling_episode_ppo_v2(
            env, policy, optimizer, graph_baseline,
            entropy_coef=entropy_coef,
            reward_scale=1.0,
            clip_epsilon=clip_epsilon,
            ppo_epochs=ppo_epochs,
            device=device,
        )

        # Extract spill/reload counts from env.schedule (not info dict)
        n_spills = sum(1 for e in env.schedule
                       if isinstance(e, tuple) and e[0] == "spill")
        n_reloads = sum(1 for e in env.schedule
                        if isinstance(e, tuple) and e[0] == "reload")

        episode_rewards.append(reward)
        episode_spills.append(n_spills)
        episode_reloads.append(n_reloads)

        if reward > best_reward:
            best_reward = reward
        if reward <= -900:
            deadlock_count += 1

        # Track @CPF performance
        if reward >= cpf_reward:
            n_cpf_beaten += 1

        # Logging
        if (episode + 1) % log_every == 0 or episode < 5:
            recent_r = episode_rewards[-min(log_every, len(episode_rewards)):]
            avg_r = sum(recent_r) / len(recent_r)
            avg_s = (sum(episode_spills[-min(log_every, len(episode_spills)):])
                     / min(log_every, len(episode_spills)))

            if isinstance(progress, range):
                cpf_mark = " @CPF" if n_cpf_beaten > 0 else ""
                print(
                    f"  Ep {episode + 1:4d}/{num_episodes} | "
                    f"reward={avg_r:7.1f} | "
                    f"spills={avg_s:.1f}/ep | "
                    f"best={best_reward:7.1f} | "
                    f"dead={deadlock_count:3d} | "
                    f"{label}{cpf_mark}"
                )
            else:
                recent_r = episode_rewards[-50:]
                avg_r = sum(recent_r) / len(recent_r) if recent_r else 0.0
                progress.set_postfix({
                    "avg": f"{avg_r:.1f}",
                    "best": f"{best_reward:.1f}",
                    "spills": f"{n_spills}",
                    "dead": f"{deadlock_count}",
                })

        # Save checkpoint
        if (episode + 1) % save_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"phase1_ep{episode + 1}.pt",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "phase": 1,
                "episode": episode + 1,
                "rewards": episode_rewards,
                "best_reward": best_reward,
                "config": {
                    "k": k,
                    "max_registers": max_registers,
                    "register_penalty_alpha": register_penalty_alpha,
                    "mem_latency": mem_latency,
                    "entropy_coef": entropy_coef,
                    "clip_epsilon": clip_epsilon,
                    "ppo_epochs": ppo_epochs,
                },
            }, ckpt_path)
            print(f"  [Checkpoint] {ckpt_path}")

    # ── Summary ──────────────────────────────────────
    last_100 = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    avg_last = sum(last_100) / len(last_100)
    total_spills = sum(episode_spills)
    total_reloads = sum(episode_reloads)

    print()
    print(f"  PHASE 1 COMPLETE")
    print(f"  {'=' * 40}")
    print(f"  Best reward:     {best_reward:7.1f}")
    print(f"  Avg last 100:    {avg_last:7.1f}")
    print(f"  Deadlocks:       {deadlock_count:4d}/{num_episodes}")
    print(f"  @CPF beaten:     {n_cpf_beaten:4d}/{num_episodes}")
    print(f"  Total spills:    {total_spills:4d}")
    print(f"  Total reloads:   {total_reloads:4d}")

    suffix = "_scratch" if from_scratch else ""
    final_path = os.path.join(checkpoint_dir, f"phase1_final{suffix}.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "phase": 1,
        "episode": num_episodes,
        "rewards": episode_rewards,
        "best_reward": best_reward,
        "config": {"from_scratch": from_scratch},
    }, final_path)
    print(f"  Final checkpoint: {final_path}")

    return {
        "rewards": episode_rewards,
        "spill_counts": episode_spills,
        "reload_counts": episode_reloads,
        "best_reward": best_reward,
        "deadlock_count": deadlock_count,
        "n_cpf_beaten": n_cpf_beaten,
        "final_checkpoint": final_path,
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Unfrozen Escalation Training
# ═══════════════════════════════════════════════════════════════

# ── Latency escalation: start higher (5c) for sharper gradients, step to 10c ──
LATENCY_MILESTONES = {
    0: 5,
    500: 8,
    1000: 10,
}

def _get_latency(episode: int) -> int:
    """Determine mem_latency for a given episode based on curriculum milestones."""
    lat = LATENCY_MILESTONES[0]
    for ep_milestone, lat_val in sorted(LATENCY_MILESTONES.items()):
        if episode >= ep_milestone:
            lat = lat_val
    return lat


def configure_stratified_optimizer(policy, base_lr: float = 5e-4):
    """
    Construct an Adam optimizer with geometrically decaying learning rates
    from the output priority heads down to the foundational input layer.

    Supports both SchedulingPolicy (4-layer GATv1) and
    HierarchicalSchedulingPolicy (3+3 layer GATv2 with local/global branches).

    For SchedulingPolicy (4-layer GATv1):
        Heads (input_proj + priority_head): 1.0x -> 5e-4
        Layer 3 (closest to head):          0.5x -> 2.5e-4
        Layer 2:                            0.25x -> 1.25e-4
        Layer 1:                            0.1x -> 5e-5
        Layer 0 (closest to input):         0.05x -> 2.5e-5

    For HierarchicalSchedulingPolicy (3+3 layer GATv2):
        Heads (input_proj + priority_head): 1.0x
        fusion:                             0.8x
        local layers 2->1->0:             0.6x, 0.4x, 0.2x
        global layers 2->1->0:            0.6x, 0.4x, 0.2x
    """
    # Ensure all parameters are trainable
    for param in policy.parameters():
        param.requires_grad = True

    param_groups = []

    if isinstance(policy, HierarchicalSchedulingPolicy):
        # ── Hierarchical GATv2 optimizer ──────────────
        # Heads: full velocity
        head_params = (list(policy.input_proj.parameters()) +
                       list(policy.priority_head.parameters()))
        param_groups.append({"params": head_params, "lr": base_lr * 1.0})

        # Fusion layer
        param_groups.append({"params": list(policy.fusion.parameters()),
                             "lr": base_lr * 0.8})

        # Local branch layers (3 layers, layer 2 closest to fusion)
        local_scales = [0.6, 0.4, 0.2]  # layer 2, 1, 0
        for layer_idx in range(policy.num_local_layers):
            layer_params = []
            layer_params.extend(list(policy.local_convs[layer_idx].parameters()))
            layer_params.extend(list(policy.local_lin[layer_idx].parameters()))
            scale = local_scales[layer_idx] if layer_idx < len(local_scales) else 0.1
            param_groups.append({
                "params": layer_params,
                "lr": base_lr * scale,
            })

        # Global branch layers (3 layers, layer 2 closest to fusion)
        global_scales = [0.6, 0.4, 0.2]  # layer 2, 1, 0
        for layer_idx in range(policy.num_global_layers):
            layer_params = []
            layer_params.extend(list(policy.global_convs[layer_idx].parameters()))
            layer_params.extend(list(policy.global_lin[layer_idx].parameters()))
            scale = global_scales[layer_idx] if layer_idx < len(global_scales) else 0.1
            param_groups.append({
                "params": layer_params,
                "lr": base_lr * scale,
            })
    else:
        # ── SchedulingPolicy (GATv1) optimizer ────────
        # Group 1: Heads (input_proj + priority_head) — full velocity
        head_params = (list(policy.input_proj.parameters()) +
                       list(policy.priority_head.parameters()))
        param_groups.append({"params": head_params, "lr": base_lr * 1.0})

        # Groups 2-5: GAT layers with geometrically decaying LR
        lr_scales = {3: 0.5, 2: 0.25, 1: 0.1, 0: 0.05}
        for layer_idx in [3, 2, 1, 0]:
            layer_params = []
            layer_params.extend(list(policy.gat_convs[layer_idx].parameters()))
            layer_params.extend(list(policy.gat_lin[layer_idx].parameters()))
            param_groups.append({
                "params": layer_params,
                "lr": base_lr * lr_scales[layer_idx],
            })

    optimizer = torch.optim.Adam(param_groups)

    n_total = sum(p.numel() for p in policy.parameters())
    policy_name = type(policy).__name__
    print(f"  Stratified optimizer ({policy_name}): {len(param_groups)} param groups")
    for i, pg in enumerate(param_groups):
        n_p = sum(p.numel() for p in pg["params"])
        print(f"    Group {i}: {n_p:>7,} params @ lr={pg['lr']:.0e}")
    print(f"  Total: {n_total:,} params (all trainable)")
    return optimizer


def _build_escalation_corpus():
    """
    Build a graded corpus of DL subgraphs for Phase 2 training.
    Returns a list of (label, graph_fn, difficulty_rank) tuples.
    Import is deferred to avoid circular imports.
    """
    from .dl_subgraphs import create_mlp_block, create_residual_add, create_layer_norm
    return [
        # Tier 1 — easiest (low register pressure, short tenure)
        ("MLP (1L, W=3)",  lambda: create_mlp_block(1, 3), 1),
        ("ResAdd (3L, d2)", lambda: create_residual_add(3, 2), 1),
        # Tier 2 — moderate (deeper chains, more live regs)
        ("MLP (3L, W=2)",  lambda: create_mlp_block(3, 2), 2),
        ("ResAdd (4L, d2)", lambda: create_residual_add(4, 2), 2),
        # Tier 3 — hardest (wide fan-in, long tenure)
        ("ResAdd (5L, d1)", lambda: create_residual_add(5, 1), 3),
        ("LayerNorm (6ch)", lambda: create_layer_norm(6), 3),
        ("MLP (5L, W=2)",  lambda: create_mlp_block(5, 2), 3),
    ]


def unfrozen_escalation_train(
    checkpoint_path: str = "checkpoints/phase1_final.pt",
    num_episodes: int = 2000,
    k: int = 2,
    max_registers: int = 3,
    register_penalty_alpha: float = 1.0,
    learning_rate: float = 5e-4,
    entropy_coef: float = 0.05,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 3,
    log_every: int = 50,
    save_every: int = 500,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
) -> dict:
    """
    Phase 2: Unfrozen Escalation with Stratified Learning Rates.

    Loads the Phase 1 checkpoint and trains all 598K parameters with
    geometrically decaying learning rates across GAT layers. The base LR
    (5e-4) applies to the priority head and input projection. Each GAT
    layer below learns at a fraction: Layer 3 at 0.5x (2.5e-4),
    Layer 2 at 0.25x (1.25e-4), Layer 1 at 0.1x (5e-5),
    Layer 0 at 0.05x (2.5e-5).

    Memory latency escalates through three milestones (5c -> 8c -> 10c)
    to create progressively stronger gradient contrast around spill/reload
    decisions. The progressive pool manager rotates graph types by
    @CPF mastery, graduating graphs that consistently beat the CPF
    heuristic.

    This replaces the binary freeze/thaw progressive unfreeze with a
    smooth velocity gradient across the network depth.

    Curriculum milestones:
        Ep 0-499:   mem_latency=5  (moderate penalty, crisp gradients)
        Ep 500-999:  mem_latency=8  (high pressure)
        Ep 1000+:    mem_latency=10 (production-level cache miss)

    LR multiplier schedule:
        Heads:      1.0x base_lr (5e-4, full velocity for action syntax)
        Layer 3:    0.5x base_lr (2.5e-4)
        Layer 2:    0.25x base_lr (1.25e-4)
        Layer 1:    0.1x base_lr (5e-5)
        Layer 0:    0.05x base_lr (2.5e-5, protected floor)

    Args:
        checkpoint_path: Path to Phase 1 checkpoint (.pt with 'policy_state_dict').
        num_episodes: Total training episodes.
        k: Issue width for the scheduling simulator.
        max_registers: Physical register limit (3 forces early spilling).
        register_penalty_alpha: Penalty multiplier for register pressure excess.
        learning_rate: Base LR for the priority head (other layers scaled down).
        entropy_coef: Entropy bonus coefficient.
        clip_epsilon: PPO clip range.
        ppo_epochs: Gradient steps per collected episode.
        log_every: Metrics logging interval.
        save_every: Checkpoint saving interval.
        checkpoint_dir: Directory for checkpoints.
        device: 'cpu' or 'cuda'.

    Returns:
        dict with keys:
          - "rewards": per-episode reward list
          - "spill_counts": per-episode spill count list
          - "reload_counts": per-episode reload count list
          - "best_reward": best reward seen
          - "deadlock_count": total deadlock episodes
          - "n_cpf_beaten": episodes where reward >= CPF baseline
          - "graduated": list of (ep, label, tier, mastery) tuples
          - "final_checkpoint": path to the saved final checkpoint
          - "latency_schedule": list of (ep, latency) milestones reached
    """
    import os
    import numpy as np
    from itertools import cycle

    print()
    print("=" * 66)
    print("  PHASE 2: STRATIFIED LR ESCALATION")
    print("  All layers trainable, geometrically decaying LRs, 5c->8c->10c")
    print("=" * 66)

    # ── Build graded corpus ──────────────────────────
    corpus = _build_escalation_corpus()
    print(f"  Corpus: {len(corpus)} graph types ({len([c for c in corpus if c[2]==1])} easy, "
          f"{len([c for c in corpus if c[2]==2])} medium, "
          f"{len([c for c in corpus if c[2]==3])} hard)")

    # ── Load Phase 1 checkpoint ─────────────────────
    print(f"  Loading Phase 1 checkpoint: {checkpoint_path}")
    policy = SchedulingPolicy(node_feat_dim=20).to(device)
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    ckpt_sd = ckpt_data["policy_state_dict"]
    policy.load_state_dict(ckpt_sd)
    print(f"  Policy params: {sum(p.numel() for p in policy.parameters())}")

    # ── Stratified optimizer (replaces freeze/thaw infrastructure) ──
    optimizer = configure_stratified_optimizer(policy, base_lr=learning_rate)
    print(f"  entropy={entropy_coef}, clip={clip_epsilon}, PPO epochs={ppo_epochs}")
    print()

    # ── Progressive pool manager ────────────────────
    # Active pool: graphs currently in training rotation.
    # Waiting list: harder graphs that promote when easier ones graduate.
    # Graduation: a graph graduates when @CPF mastery >= 0.8 over last 30 eps.
    GRAD_THRESHOLD = 0.8
    GRAD_WINDOW = 30
    TIER_POOL_SIZE = {1: 2, 2: 2, 3: 3}  # how many slots per tier (matches corpus)

    # Build pool structure
    active_pool = {}  # item_index -> {label, g_fn, tier, cpf_hits: [], eps_on: int}
    waiting_list = {1: [], 2: [], 3: []}  # tier -> list of (label, g_fn) awaiting promotion

    for label, g_fn, tier in corpus:
        waiting_list[tier].append((label, g_fn))

    _next_pool_id = 0  # monotonically increasing ID to prevent key collisions

    def _promote_from_tier(tier):
        """Pop one graph from the waiting list for a given tier and add to active pool.
        Uses a monotonically increasing ID to prevent key collisions when graphs graduate."""
        nonlocal _next_pool_id
        if waiting_list[tier]:
            label, g_fn = waiting_list[tier].pop(0)
            idx = _next_pool_id
            _next_pool_id += 1
            active_pool[idx] = {
                "label": label,
                "g_fn": g_fn,
                "tier": tier,
                "cpf_hits": [],
                "eps_on": 0,
            }
            print(f"  [Pool] Promoted '{label}' (tier {tier}) to pool (id={idx})")
            return idx
        return None

    # Initialize active pool — fill up to TIER_POOL_SIZE per tier
    for tier in [1, 2, 3]:
        for _ in range(TIER_POOL_SIZE[tier]):
            _promote_from_tier(tier)

    print(f"  Active pool: {len(active_pool)} graphs")
    for idx, info in active_pool.items():
        print(f"    {idx}: {info['label']} (tier {info['tier']})")
    print(f"  Waiting list: {sum(len(v) for v in waiting_list.values())} graphs remaining")
    print()

    graduated = []  # (ep, label, tier, mastery_at_graduation)

    # ── Training state ──────────────────────────────
    # Per-graph baselines: each topology label gets its own EMA.
    # This isolates the advantage signal to that specific graph's reward
    # distribution, eliminating cross-topology "baseline whiplash."
    baselines: Dict[str, RunningBaseline] = {}
    episode_rewards = []
    episode_spills = []
    episode_reloads = []
    best_reward = -float("inf")
    deadlock_count = 0
    n_cpf_beaten = 0
    latency_schedule = []  # (ep, latency) at milestone transitions

    # ── KL divergence tracking ────────────────────
    # Compute approximate KL(π_old || π_new) after each PPO epoch.
    # This is computed from old_log_probs and new_log_probs already available
    # in train_scheduling_episode_ppo_v2. We log the mean per episode.
    episode_kl = []  # mean KL per episode

    # Per-graph CPF cache (avoid recomputing every episode)
    cpf_cache: Dict[str, tuple] = {}  # label -> (cpf_reward, cpf_cycles)

    # Per-episode snapshot of each topology's RunningBaseline value.
    # Logged every log_every episodes to track individual graph reward curves.
    baseline_snapshots: list = []

    # Track which graph and CPF baseline was used each episode
    episode_graphs = []
    episode_cpf_rewards = []  # parallel to episode_rewards for accurate summary

    try:
        from tqdm import tqdm
        progress = tqdm(range(num_episodes), desc="Unfrozen Escalation")
    except ImportError:
        progress = range(num_episodes)

    for episode in progress:
        # ── Determine current latency ────────────────
        current_lat = _get_latency(episode)

        # ── Performance-weighted graph sampling ──────
        # Compute selection weights: low-mastery graphs are more likely.
        if not active_pool:
            print(f"Ep {episode}: active pool empty — all graphs graduated.")
            break

        weights = {}
        for idx, info in active_pool.items():
            recent = info["cpf_hits"][-GRAD_WINDOW:]
            mastery = sum(recent) / len(recent) if recent else 0.0
            # Invert: low mastery -> high weight (prefer underperforming graphs)
            weights[idx] = max(0.05, 1.0 - mastery)

        total_w = sum(weights.values())
        probs = [weights[idx] / total_w for idx in active_pool]
        rng = np.random.RandomState(episode * 117 + 42)
        chosen_idx = rng.choice(list(active_pool.keys()), p=probs)

        chosen = active_pool[chosen_idx]
        chosen["eps_on"] += 1
        graph = chosen["g_fn"]()
        topo = graph.get_topological_order()
        node_names = graph.inputs + topo + graph.outputs
        episode_graphs.append(chosen["label"])

        # ── CPF baseline ────────────────────────────
        label = chosen["label"]
        # Invalidate CPF cache at latency transitions so baseline is accurate
        if episode in (0, 500, 1000):
            cpf_cache.clear()

        if label not in cpf_cache or current_lat != cpf_cache.get(label, (None, None))[2]:
            cpf_sched = schedule_cpf(graph)
            cpf_env = SchedulingGymEnv(
                graph, None, [],
                max_exec_units=k, latency=DEFAULT_LATENCY,
                max_registers=max_registers,
                register_penalty_alpha=register_penalty_alpha,
                mem_latency=current_lat,
            )
            cpf_env.reset()
            for nid in cpf_sched:
                cpf_idx = node_names.index(nid)
                _, _, cpf_done, cpf_info = cpf_env.step(cpf_idx)
                if cpf_done:
                    break
            cpf_r = cpf_info.get("spill_penalty", 0) + cpf_info.get("cycles", 0)
            cpf_cyc = cpf_info.get("cycles", 0)
            cpf_cache[label] = (-cpf_r, cpf_cyc, current_lat)

        cpf_reward, cpf_cycles, _ = cpf_cache[label]

        # ── Training env with current latency ────────
        env = SchedulingGymEnv(
            graph, None, [],
            max_exec_units=k, latency=DEFAULT_LATENCY,
            max_registers=max_registers,
            register_penalty_alpha=register_penalty_alpha,
            mem_latency=current_lat,
        )

        # ── Per-label baseline lookup ────────────────
        # Each graph label gets its own EMA, preventing the shared-baseline
        # whiplash where a hard LayerNorm episode's reward poisons the
        # advantage signal for the next easy MLP episode.
        if label not in baselines:
            baselines[label] = RunningBaseline(alpha=0.9)
        graph_baseline = baselines[label]

        reward, info = train_scheduling_episode_ppo_v2(
            env, policy, optimizer, graph_baseline,
            entropy_coef=entropy_coef,
            reward_scale=1.0,
            clip_epsilon=clip_epsilon,
            ppo_epochs=ppo_epochs,
            device=device,
            track_kl=True,
        )

        # ── KL logging ──────────────────────────────
        if "approx_kl" in info:
            episode_kl.append(info["approx_kl"])

        # Extract spill/reload counts
        n_spills = sum(1 for e in env.schedule
                       if isinstance(e, tuple) and e[0] == "spill")
        n_reloads = sum(1 for e in env.schedule
                        if isinstance(e, tuple) and e[0] == "reload")

        episode_rewards.append(reward)
        episode_spills.append(n_spills)
        episode_reloads.append(n_reloads)

        if reward > best_reward:
            best_reward = reward
        if reward <= -900:
            deadlock_count += 1

        # Track @CPF performance for this graph
        episode_cpf_rewards.append(cpf_reward)
        if reward >= cpf_reward:
            n_cpf_beaten += 1
            chosen["cpf_hits"].append(1)
        else:
            chosen["cpf_hits"].append(0)

        # ── Graduation check ────────────────────────
        recent_hits = chosen["cpf_hits"][-GRAD_WINDOW:]
        if len(recent_hits) >= GRAD_WINDOW:
            mastery = sum(recent_hits) / GRAD_WINDOW
            if mastery >= GRAD_THRESHOLD and chosen["eps_on"] >= GRAD_WINDOW:
                # Graduate this graph
                graduated.append((episode, chosen["label"], chosen["tier"], mastery))
                del active_pool[chosen_idx]
                print(
                    f"  Ep {episode:4d}: GRADUATED '{chosen['label']}' "
                    f"(tier {chosen['tier']}, mastery={mastery:.0%})"
                )
                # Promote new graph from same tier or next tier
                promoted = _promote_from_tier(chosen["tier"])
                if promoted is not None:
                    pass
                # Clear stale CPF cache entry so it's recomputed next time
                if chosen["label"] in cpf_cache:
                    del cpf_cache[chosen["label"]]

        # ── Latency milestone logging ────────────────
        if episode == 0 or (
            episode in LATENCY_MILESTONES and current_lat != _get_latency(episode - 1)
        ):
            latency_schedule.append((episode, current_lat))
            print(f"  Ep {episode:4d}: Latency step-up -> mem_latency={current_lat}")

        # ── Logging ──────────────────────────────────
        if (episode + 1) % log_every == 0 or episode < 5:
            recent_r = episode_rewards[-min(log_every, len(episode_rewards)):]
            avg_r = sum(recent_r) / len(recent_r)
            avg_s = (sum(episode_spills[-min(log_every, len(episode_spills)):])
                     / min(log_every, len(episode_spills)))
            avg_kl = sum(episode_kl[-min(log_every, len(episode_kl)):]) / min(log_every, len(episode_kl)) if episode_kl else 0.0

            pool_mastery = {
                info["label"]: (
                    sum(info["cpf_hits"][-GRAD_WINDOW:]) / len(info["cpf_hits"][-GRAD_WINDOW:])
                    if len(info["cpf_hits"]) >= GRAD_WINDOW else 0.0
                )
                for info in active_pool.values()
            }

            # ── Baseline snapshot (every log_every) ──
            if (episode + 1) % log_every == 0 and baselines:
                snapshot = {
                    "ep": episode + 1,
                    "mem_lat": current_lat,                        "layers": "all",
                    "baselines": {
                        lbl: bl.get() for lbl, bl in sorted(baselines.items())
                    },
                }
                baseline_snapshots.append(snapshot)

            if isinstance(progress, range):
                print(
                    f"  Ep {episode + 1:4d}/{num_episodes} | "
                    f"reward={avg_r:7.1f} | "
                    f"lat={current_lat:2d} | "
                    f"kl={avg_kl:.4f} | "
                    f"spills={avg_s:.1f}/ep | "
                    f"pool={len(active_pool)}g | "
                    f"grad={len(graduated)} | "
                    f"dead={deadlock_count:3d} | "
                    f"{chosen['label']}"
                )
                if pool_mastery and (episode + 1) % (log_every * 2) == 0:
                    mastery_str = " | ".join(
                        f"{k}={v:.0%}" for k, v in sorted(pool_mastery.items())
                    )
                    print(f"         mastery: {mastery_str}")
                    # Show per-topology baseline values alongside mastery
                    bl_str = " | ".join(
                        f"bl:{k}={baselines[k].get():.0f}"
                        for k in sorted(pool_mastery.keys())
                        if k in baselines
                    )
                    if bl_str:
                        print(f"         {bl_str}")
            else:
                progress.set_postfix({
                    "avg": f"{avg_r:.1f}",
                    "lat": current_lat,
                    "kl": f"{avg_kl:.4f}",
                    "pool": len(active_pool),
                    "grad": len(graduated),
                    "dead": deadlock_count,
                })

        # ── Save checkpoint ──────────────────────────
        if (episode + 1) % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"phase2_ep{episode + 1}.pt",
            )
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "phase": 2,
                "episode": episode + 1,
                "rewards": episode_rewards,
                "spill_counts": episode_spills,
                "reload_counts": episode_reloads,
                "best_reward": best_reward,
                "graduated": graduated,
                "baseline_snapshots": baseline_snapshots,
                "latency_schedule": latency_schedule,
                "config": {
                    "k": k,
                    "max_registers": max_registers,
                    "register_penalty_alpha": register_penalty_alpha,
                    "learning_rate": learning_rate,
                    "entropy_coef": entropy_coef,
                    "clip_epsilon": clip_epsilon,
                    "ppo_epochs": ppo_epochs,
                },
            }, ckpt_path)
            print(f"  [Checkpoint] {ckpt_path}")

    # ── Final summary ────────────────────────────────
    last_200 = episode_rewards[-200:] if len(episode_rewards) >= 200 else episode_rewards
    avg_last = sum(last_200) / len(last_200)
    total_spills = sum(episode_spills)
    total_reloads = sum(episode_reloads)

    # Track CPF-beaten rate per latency regime (matches new milestones: 5c→8c→10c)
    regime_0_499 = episode_cpf_rewards[:500]
    regime_500_999 = episode_cpf_rewards[500:1000]
    regime_1000_plus = episode_cpf_rewards[1000:]

    def _regime_cpf_rate(segment, rewards_seg):
        """CPF-beaten rate using parallel reward and CPF lists."""
        if not segment or not rewards_seg:
            return 0.0
        n = min(len(segment), len(rewards_seg))
        hits = sum(1 for i in range(n) if rewards_seg[i] >= segment[i])
        return hits / n

    print()
    print(f"  PHASE 2 COMPLETE")
    print(f"  {'=' * 40}")
    print(f"  Best reward:     {best_reward:7.1f}")
    print(f"  Avg last 200:    {avg_last:7.1f}")
    print(f"  Deadlocks:       {deadlock_count:4d}/{num_episodes}")
    print(f"  @CPF beaten:     {n_cpf_beaten:4d}/{num_episodes}")
    print(f"  Graphs graduated:{len(graduated):4d}")
    print(f"  Total spills:    {total_spills:4d}")
    print(f"  Total reloads:   {total_reloads:4d}")
    print(f"  Latency schedule:"
          f"  mem_lat=5: {_regime_cpf_rate(regime_0_499, episode_rewards[:500])*100:.0f}% @CPF, "
          f"mem_lat=8: {_regime_cpf_rate(regime_500_999, episode_rewards[500:1000])*100:.0f}% @CPF, "
          f"mem_lat=10: {_regime_cpf_rate(regime_1000_plus, episode_rewards[1000:])*100:.0f}% @CPF")
    print()

    if graduated:
        print(f"  Graduated graphs:")
        for ep, lbl, tier, mas in graduated:
            print(f"    Ep {ep:4d}: {lbl:20s} (tier {tier}, mastery={mas:.0%})")

    # ── Save final model ─────────────────────────────
    final_path = os.path.join(checkpoint_dir, "phase2_final.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "phase": 2,
        "episode": num_episodes,
        "rewards": episode_rewards,
        "spill_counts": episode_spills,
        "reload_counts": episode_reloads,
        "best_reward": best_reward,
        "graduated": graduated,
        "baseline_snapshots": baseline_snapshots,
        "latency_schedule": latency_schedule,
        "config": {},
    }, final_path)
    print(f"  Final checkpoint: {final_path}")

    return {
        "rewards": episode_rewards,
        "spill_counts": episode_spills,
        "reload_counts": episode_reloads,
        "best_reward": best_reward,
        "deadlock_count": deadlock_count,
        "n_cpf_beaten": n_cpf_beaten,
        "graduated": graduated,
        "baseline_snapshots": baseline_snapshots,
        "latency_schedule": latency_schedule,
        "final_checkpoint": final_path,
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 3: Continuous Curricular Annealing
# ═══════════════════════════════════════════════════════════════

# ── Continuous annealing: performance-gated difficulty interpolation ──
# Replaces the discrete CHAOS_PHASES regime structure with a smooth
# `difficulty_factor ∈ [0.0, 1.0]` driven by the rolling @CPF rate.
# The environment interpolates continually — no regime transition shocks,
# no baseline resets, no wasted re-stabilization episodes.
#
# As the policy improves (higher @CPF rate), the difficulty automatically
# escalates: latency rises 5c→10c, more port constraints activate, and
# stochastic jitter widens. An episode-based floor ensures full chaos
# by episode 500 even if @CPF stays low.
#
# Entropy scales with difficulty: `effective_entropy = 0.05 * (1.0 + df)`
# giving the policy more exploration slack exactly when it needs it most.


BASE_LATENCIES = {"Mul": 3, "Add": 1, "Sub": 1, "Div": 3, "CmpGeZ": 1}


def _interpolate_chaos(difficulty_factor: float) -> dict:
    """
    Return the chaos config interpolated at a given difficulty_factor [0.0, 1.0].

    All parameters are continuous functions of difficulty_factor:
    - mem_latency:  linear 5c -> 10c
    - unit_limit:   step thresholds at df=0.2 (Mul=1), df=0.5 (Add=2), df=0.7 (Div=1)
    - jitter width: linear 0 -> 2 (widening distribution around base latency)
    """
    df = max(0.0, min(1.0, difficulty_factor))

    mem_latency = int(5 + df * 5)  # 5c -> 10c

    # Port limits activate at difficulty thresholds
    unit_limit = {}
    if df > 0.2:
        unit_limit["Mul"] = 1
    if df > 0.5:
        unit_limit["Add"] = 2
    if df > 0.7:
        unit_limit["Div"] = 1

    # Latency jitter widens linearly with difficulty
    jitter_radius = int(df * 2)  # 0 at df=0.0, 2 at df=1.0
    latency_distribution = {}
    if jitter_radius > 0:
        for op, base in BASE_LATENCIES.items():
            if base > 1 or df > 0.6:  # delay jitter for fast ops until mid-chaos
                low = max(1, base - jitter_radius)
                high = base + jitter_radius
                latency_distribution[op] = (low, high)

    return {
        "mem_latency": mem_latency,
        "unit_limit": unit_limit,
        "latency_distribution": latency_distribution,
    }


def chaos_annealing_train(
    checkpoint_path: str = "checkpoints/phase2_final.pt",
    num_episodes: int = 2000,
    k: int = 2,
    max_registers: int = 3,
    register_penalty_alpha: float = 1.0,
    learning_rate: float = 5e-4,
    entropy_coef: float = 0.05,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 3,
    log_every: int = 50,
    save_every: int = 500,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
) -> dict:
    """
    Phase 3: Continuous Curricular Annealing.

    Loads the Phase 2 checkpoint and activates the dormant port hazard
    and stochastic latency infrastructure, but unlike the discrete-regime
    Phase 3, the difficulty interpolates continuously based on the rolling
    @CPF performance rate.

    The difficulty_factor ∈ [0.0, 1.0] is computed as:
        roll_cpf_rate  = rolling @CPF over last 100 episodes
        df = max(min(1.0, roll_cpf_rate * 2.0), min(1.0, episode / 500))

    The performance-driven component (roll_cpf_rate * 2.0) ensures the
    environment ramps at the rate of the policy's own improvement.
    The episode-based floor (episode / 500) guarantees full chaos by
    episode 500 even if @CPF stays near zero.

    This eliminates regime transition shock entirely:
    - No baseline clears (EMAs track gradual drift naturally)
    - No CPF cache invalidations (env changes slowly)
    - No KL spikes from sudden distribution shifts
    - Entropy scales dynamically: effective_entropy = entropy_coef * (1.0 + df)

    At df=0.0: No port hazards, deterministic latencies, 5c latency
    At df=1.0: Mul=1 Add=2 Div=1 ports, all ops stochastic with ±2c jitter, 10c latency

    Args:
        checkpoint_path: Path to Phase 2 checkpoint.
        num_episodes: Total training episodes.
        k: Issue width.
        max_registers: Physical register limit.
        register_penalty_alpha: Register pressure penalty multiplier.
        learning_rate: Base LR for stratified optimizer.
        entropy_coef: Base entropy coefficient (scaled by 1.0 + difficulty_factor).
        clip_epsilon: PPO clip range.
        ppo_epochs: Gradient steps per collected episode.
        log_every: Metrics logging interval.
        save_every: Checkpoint saving interval.
        checkpoint_dir: Directory for checkpoints.
        device: 'cpu' or 'cuda'.

    Returns:
        dict with full training telemetry.
    """
    import os
    import numpy as np
    from itertools import cycle

    print()
    print("=" * 66)
    print("  PHASE 3: CONTINUOUS CURRICULAR ANNEALING")
    print("  Performance-gated difficulty, no regime transitions")
    print("=" * 66)

    # ── Load Phase 2 checkpoint ─────────────────────
    corpus = _build_escalation_corpus()
    print(f"  Corpus: {len(corpus)} graph types ({len([c for c in corpus if c[2]==1])} easy, "
          f"{len([c for c in corpus if c[2]==2])} medium, "
          f"{len([c for c in corpus if c[2]==3])} hard)")

    print(f"  Loading Phase 2 checkpoint: {checkpoint_path}")
    policy = SchedulingPolicy(node_feat_dim=20).to(device)
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    ckpt_sd = ckpt_data["policy_state_dict"]
    policy.load_state_dict(ckpt_sd)
    print(f"  Policy params: {sum(p.numel() for p in policy.parameters())}")

    # ── Stratified optimizer (fixed base_lr throughout) ──
    optimizer = configure_stratified_optimizer(policy, base_lr=learning_rate)
    print(f"  base_entropy={entropy_coef}, clip={clip_epsilon}, PPO epochs={ppo_epochs}")
    print()

    # ── Progressive pool manager (same as Phase 2) ──
    GRAD_THRESHOLD = 0.8
    GRAD_WINDOW = 30
    TIER_POOL_SIZE = {1: 2, 2: 2, 3: 3}

    active_pool = {}
    waiting_list = {1: [], 2: [], 3: []}

    for label, g_fn, tier in corpus:
        waiting_list[tier].append((label, g_fn))

    _next_pool_id = 0

    def _promote_from_tier(tier):
        nonlocal _next_pool_id
        if waiting_list[tier]:
            label, g_fn = waiting_list[tier].pop(0)
            idx = _next_pool_id
            _next_pool_id += 1
            active_pool[idx] = {
                "label": label,
                "g_fn": g_fn,
                "tier": tier,
                "cpf_hits": [],
                "eps_on": 0,
            }
            print(f"  [Pool] Promoted '{label}' (tier {tier}) to pool (id={idx})")
            return idx
        return None

    for tier in [1, 2, 3]:
        for _ in range(TIER_POOL_SIZE[tier]):
            _promote_from_tier(tier)

    print(f"  Active pool: {len(active_pool)} graphs")
    print(f"  Waiting list: {sum(len(v) for v in waiting_list.values())} graphs remaining")
    print()

    graduated = []

    # ── Training state ──────────────────────────────
    baselines: Dict[str, RunningBaseline] = {}
    episode_rewards = []
    episode_spills = []
    episode_reloads = []
    best_reward = -float("inf")
    deadlock_count = 0
    n_cpf_beaten = 0

    episode_kl = []
    cpf_cache: Dict[str, tuple] = {}
    baseline_snapshots: list = []
    episode_cpf_rewards = []

    # ── Rolling @CPF buffer for performance gating ──
    CPF_ROLLING_WINDOW = 100
    _cpf_hits_buffer = []  # rolling window of per-episode @CPF booleans

    # ── Difficulty history ──────────────────────────
    difficulty_history = []  # (episode, difficulty_factor) pairs

    # ── Entropy tracking ────────────────────────────
    current_entropy_coef = entropy_coef

    # ── CPF cache invalidated only when env changes measurably ──
    _last_cpf_env_key = None  # tuple of (lat, frozenset(unit_limit), frozenset(lat_dist))

    # ── KL-Triggered Adaptive Bump ──────────────────
    # When KL flatlines below 0.001 for 30+ episodes, the policy has
    # exhausted its current strategy. A discrete difficulty bump (+0.15)
    # forces re-exploration, followed by a transient entropy boost.
    KLLATLINE_THRESHOLD = 0.001
    KLLATLINE_WINDOW = 30
    BUMP_STEP = 0.15
    _kl_window = []           # rolling window of per-episode KL values
    _kl_flatline_counter = 0  # consecutive episodes with avg KL < threshold
    _adaptive_bump = 0.0      # cumulative difficulty bump from KL triggers
    _entropy_boost_remaining = 0  # episodes of boosted entropy after bump

    try:
        from tqdm import tqdm
        progress = tqdm(range(num_episodes), desc="Annealing")
    except ImportError:
        progress = range(num_episodes)

    for episode in progress:
        # ═══════════════════════════════════════════════
        #  Continuous Difficulty Computation
        # ═══════════════════════════════════════════════

        # Compute rolling @CPF rate
        roll_cpf = 0.0
        if _cpf_hits_buffer:
            roll_cpf = sum(_cpf_hits_buffer) / len(_cpf_hits_buffer)

        # Performance-gated difficulty: 0% @CPF -> df=0, 50% @CPF -> df=1
        perf_df = min(1.0, roll_cpf * 2.0)

        # Episode-based floor: guarantees full chaos by episode 500
        ep_floor = min(1.0, episode / 500)

        # Final difficulty factor
        difficulty_factor = max(perf_df, ep_floor)

        # Interpolate chaos config at this difficulty
        config = _interpolate_chaos(difficulty_factor)
        current_lat = config["mem_latency"]
        current_unit_limit = config["unit_limit"]
        current_lat_dist = config["latency_distribution"]

        # Apply adaptive bump from KL triggers
        bumped_df = difficulty_factor + _adaptive_bump
        effective_df = min(1.0, bumped_df)

        # Re-interpolate at effective difficulty (includes bump)
        config = _interpolate_chaos(effective_df)
        current_lat = config["mem_latency"]
        current_unit_limit = config["unit_limit"]
        current_lat_dist = config["latency_distribution"]
        difficulty_factor = effective_df  # use bumped df for logging

        # Dynamic entropy: higher entropy at higher difficulty
        entropy_boost = 0.1 if _entropy_boost_remaining > 0 else 0.0
        current_entropy_coef = entropy_coef * (1.0 + difficulty_factor + entropy_boost)
        if _entropy_boost_remaining > 0:
            _entropy_boost_remaining -= 1

        # Track difficulty for telemetry
        if episode == 0 or (episode + 1) % log_every == 0:
            difficulty_history.append((episode, difficulty_factor, _adaptive_bump))

        # ── Performance-weighted graph sampling ──────
        if not active_pool:
            print(f"Ep {episode}: active pool empty — all graphs graduated.")
            break

        weights = {}
        for idx, info in active_pool.items():
            recent = info["cpf_hits"][-GRAD_WINDOW:]
            mastery = sum(recent) / len(recent) if recent else 0.0
            weights[idx] = max(0.05, 1.0 - mastery)

        total_w = sum(weights.values())
        probs = [weights[idx] / total_w for idx in active_pool]
        rng = np.random.RandomState(episode * 117 + 42)
        chosen_idx = rng.choice(list(active_pool.keys()), p=probs)

        chosen = active_pool[chosen_idx]
        chosen["eps_on"] += 1
        graph = chosen["g_fn"]()
        topo = graph.get_topological_order()
        node_names = graph.inputs + topo + graph.outputs
        label = chosen["label"]

        # ═══════════════════════════════════════════════
        #  CPF Baseline (continuous recomputation)
        # ═══════════════════════════════════════════════

        # Build env key to detect meaningful config changes
        env_key = (
            current_lat,
            frozenset(current_unit_limit.items()),
            frozenset((k, v[0], v[1]) for k, v in sorted(current_lat_dist.items()))
        )

        if label not in cpf_cache or env_key != _last_cpf_env_key:
            cpf_sched = schedule_cpf(graph)
            cpf_env = SchedulingGymEnv(
                graph, None, [],
                max_exec_units=k, latency=DEFAULT_LATENCY,
                max_registers=max_registers,
                register_penalty_alpha=register_penalty_alpha,
                mem_latency=current_lat,
                unit_limit=current_unit_limit,
                latency_distribution=current_lat_dist,
                partial_credit_per_issue=partial_credit_per_issue,
            )
            cpf_env.reset()
            for nid in cpf_sched:
                cpf_idx = node_names.index(nid)
                _, _, cpf_done, cpf_info = cpf_env.step(cpf_idx)
                if cpf_done:
                    break
            cpf_r = cpf_info.get("spill_penalty", 0) + cpf_info.get("cycles", 0)
            cpf_cyc = cpf_info.get("cycles", 0)
            cpf_cache[label] = (-cpf_r, cpf_cyc)
            _last_cpf_env_key = env_key

        cpf_reward, cpf_cycles = cpf_cache[label]

        # ═══════════════════════════════════════════════
        #  Training Environment
        # ═══════════════════════════════════════════════

        env = SchedulingGymEnv(
            graph, None, [],
            max_exec_units=k, latency=DEFAULT_LATENCY,
            max_registers=max_registers,
            register_penalty_alpha=register_penalty_alpha,
            mem_latency=current_lat,
            unit_limit=current_unit_limit,
            latency_distribution=current_lat_dist,
            partial_credit_per_issue=partial_credit_per_issue,
        )

        # ═══════════════════════════════════════════════
        #  Per-Label Baseline (never cleared — gradual drift)
        # ═══════════════════════════════════════════════

        if label not in baselines:
            baselines[label] = RunningBaseline(alpha=0.9)
        graph_baseline = baselines[label]

        reward, info = train_scheduling_episode_ppo_v2(
            env, policy, optimizer, graph_baseline,
            entropy_coef=current_entropy_coef,
            reward_scale=1.0,
            clip_epsilon=clip_epsilon,
            ppo_epochs=ppo_epochs,
            device=device,
            track_kl=True,
        )

        # ═══════════════════════════════════════════════
        #  Post-Episode Accounting
        # ═══════════════════════════════════════════════

        # KL logging + flatline detection
        if "approx_kl" in info:
            kl_val = info["approx_kl"]
            episode_kl.append(kl_val)
            # Rolling KL window
            _kl_window.append(kl_val)
            if len(_kl_window) > KLLATLINE_WINDOW:
                _kl_window.pop(0)
            # Flatline check (need at least 5 samples to avoid noise)
            if len(_kl_window) >= 5 and not _entropy_boost_remaining > 0:
                avg_kl_window = sum(_kl_window) / len(_kl_window)
                if avg_kl_window < KLLATLINE_THRESHOLD:
                    _kl_flatline_counter += 1
                else:
                    _kl_flatline_counter = 0
                # Trigger bump when flatlined for KLLATLINE_WINDOW episodes
                if _kl_flatline_counter >= KLLATLINE_WINDOW:
                    _adaptive_bump += BUMP_STEP
                    _kl_flatline_counter = 0
                    _entropy_boost_remaining = 10  # 10 episodes of boosted entropy
                    print(f"  ** KL TRIGGER ep {episode:4d}: bump +{BUMP_STEP:.2f} "
                          f"(total={_adaptive_bump:.2f}, entropy_boost=10eps)")

        # Extract spill/reload counts
        n_spills = sum(1 for e in env.schedule
                       if isinstance(e, tuple) and e[0] == "spill")
        n_reloads = sum(1 for e in env.schedule
                        if isinstance(e, tuple) and e[0] == "reload")

        episode_rewards.append(reward)
        episode_spills.append(n_spills)
        episode_reloads.append(n_reloads)

        if reward > best_reward:
            best_reward = reward
        if reward <= -900:
            deadlock_count += 1

        episode_cpf_rewards.append(cpf_reward)
        if reward >= cpf_reward:
            n_cpf_beaten += 1
            chosen["cpf_hits"].append(1)
            _cpf_hits_buffer.append(1)
        else:
            chosen["cpf_hits"].append(0)
            _cpf_hits_buffer.append(0)

        # Maintain rolling buffer size
        if len(_cpf_hits_buffer) > CPF_ROLLING_WINDOW:
            _cpf_hits_buffer.pop(0)

        # ── Graduation check ────────────────────────
        recent_hits = chosen["cpf_hits"][-GRAD_WINDOW:]
        if len(recent_hits) >= GRAD_WINDOW:
            mastery = sum(recent_hits) / GRAD_WINDOW
            if mastery >= GRAD_THRESHOLD and chosen["eps_on"] >= GRAD_WINDOW:
                graduated.append((episode, chosen["label"], chosen["tier"], mastery))
                del active_pool[chosen_idx]
                print(
                    f"  Ep {episode:4d}: GRADUATED '{chosen['label']}' "
                    f"(tier {chosen['tier']}, mastery={mastery:.0%})"
                )
                _promote_from_tier(chosen["tier"])
                if chosen["label"] in cpf_cache:
                    del cpf_cache[chosen["label"]]

        # ═══════════════════════════════════════════════
        #  Logging
        # ═══════════════════════════════════════════════

        if (episode + 1) % log_every == 0 or episode < 5:
            recent_r = episode_rewards[-min(log_every, len(episode_rewards)):]
            avg_r = sum(recent_r) / len(recent_r)
            avg_s = (sum(episode_spills[-min(log_every, len(episode_spills)):])
                     / min(log_every, len(episode_spills)))
            avg_kl = sum(episode_kl[-min(log_every, len(episode_kl)):]) / min(log_every, len(episode_kl)) if episode_kl else 0.0

            pool_mastery = {
                info["label"]: (
                    sum(info["cpf_hits"][-GRAD_WINDOW:]) / len(info["cpf_hits"][-GRAD_WINDOW:])
                    if len(info["cpf_hits"]) >= GRAD_WINDOW else 0.0
                )
                for info in active_pool.values()
            }

            # ── Baseline snapshot ──
            if (episode + 1) % log_every == 0 and baselines:
                snapshot = {
                    "ep": episode + 1,
                    "mem_lat": current_lat,
                    "difficulty": round(difficulty_factor, 3),
                    "bump": round(_adaptive_bump, 3),
                    "kl_flatline": _kl_flatline_counter,
                    "entropy": current_entropy_coef,
                    "roll_cpf": round(roll_cpf, 3),
                    "unit_limit": dict(current_unit_limit),
                    "lat_dist": {
                        k: v for k, v in sorted(current_lat_dist.items())
                    },
                    "baselines": {
                        lbl: bl.get() for lbl, bl in sorted(baselines.items())
                    },
                }
                baseline_snapshots.append(snapshot)

            if isinstance(progress, range):
                bolt_indicator = " ⚡" if _entropy_boost_remaining > 0 else "   "
                print(
                    f"  Ep {episode + 1:4d}/{num_episodes} | "
                    f"reward={avg_r:7.1f} | "
                    f"lat={current_lat:2d}c | "
                    f"df={difficulty_factor:.2f}{bolt_indicator} | "
                    f"kl={avg_kl:.4f} | "
                    f"spills={avg_s:.1f}/ep | "
                    f"pool={len(active_pool)}g | "
                    f"grad={len(graduated)} | "
                    f"dead={deadlock_count:3d} | "
                    f"{chosen['label']}"
                )
                if pool_mastery and (episode + 1) % (log_every * 2) == 0:
                    mastery_str = " | ".join(
                        f"{k}={v:.0%}" for k, v in sorted(pool_mastery.items())
                    )
                    print(f"         mastery: {mastery_str}")
                    bl_str = " | ".join(
                        f"bl:{k}={baselines[k].get():.0f}"
                        for k in sorted(pool_mastery.keys())
                        if k in baselines
                    )
                    if bl_str:
                        print(f"         {bl_str}")
            else:
                progress.set_postfix({
                    "avg": f"{avg_r:.1f}",
                    "lat": current_lat,
                    "df": f"{difficulty_factor:.2f}",
                    "kl": f"{avg_kl:.4f}",
                    "pool": len(active_pool),
                    "grad": len(graduated),
                    "dead": deadlock_count,
                })

        # ── Save checkpoint ──────────────────────────
        if (episode + 1) % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"phase3_ep{episode + 1}.pt",
            )
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "phase": 3,
                "episode": episode + 1,
                "rewards": episode_rewards,
                "spill_counts": episode_spills,
                "reload_counts": episode_reloads,
                "best_reward": best_reward,
                "graduated": graduated,
                "baseline_snapshots": baseline_snapshots,
                "difficulty_history": difficulty_history,
                "config": {
                    "k": k,
                    "max_registers": max_registers,
                    "register_penalty_alpha": register_penalty_alpha,
                    "learning_rate": learning_rate,
                    "entropy_coef": entropy_coef,
                    "clip_epsilon": clip_epsilon,
                    "ppo_epochs": ppo_epochs,
                },
            }, ckpt_path)
            print(f"  [Checkpoint] {ckpt_path}")

    # ── Final summary ────────────────────────────────
    last_200 = episode_rewards[-200:] if len(episode_rewards) >= 200 else episode_rewards
    avg_last = sum(last_200) / len(last_200)
    total_spills = sum(episode_spills)
    total_reloads = sum(episode_reloads)

    print()
    print(f"  PHASE 3 (ANNEALING) COMPLETE")
    print(f"  {'=' * 40}")
    print(f"  Best reward:     {best_reward:7.1f}")
    print(f"  Avg last 200:    {avg_last:7.1f}")
    print(f"  Deadlocks:       {deadlock_count:4d}/{num_episodes}")
    print(f"  @CPF beaten:     {n_cpf_beaten:4d}/{num_episodes}")
    print(f"  Graphs graduated:{len(graduated):4d}")
    print(f"  Total spills:    {total_spills:4d}")
    print(f"  Total reloads:   {total_reloads:4d}")

    if difficulty_history:
        df_at_start = difficulty_history[0][1]
        df_at_end = difficulty_history[-1][1]
        print(f"  Difficulty:      {df_at_start:.2f} -> {df_at_end:.2f} "
              f"(target={roll_cpf:.3f} roll @CPF)")

    if graduated:
        print(f"  Graduated graphs:")
        for ep, lbl, tier, mas in graduated:
            print(f"    Ep {ep:4d}: {lbl:20s} (tier {tier}, mastery={mas:.0%})")

    # ── Save final model ─────────────────────────────
    final_path = os.path.join(checkpoint_dir, "phase3_final.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "phase": 3,
        "episode": num_episodes,
        "rewards": episode_rewards,
        "spill_counts": episode_spills,
        "reload_counts": episode_reloads,
        "best_reward": best_reward,
        "graduated": graduated,
        "baseline_snapshots": baseline_snapshots,
        "difficulty_history": difficulty_history,
        "config": {},
    }, final_path)
    print(f"  Final checkpoint: {final_path}")

    return {
        "rewards": episode_rewards,
        "spill_counts": episode_spills,
        "reload_counts": episode_reloads,
        "best_reward": best_reward,
        "deadlock_count": deadlock_count,
        "n_cpf_beaten": n_cpf_beaten,
        "graduated": graduated,
        "baseline_snapshots": baseline_snapshots,
        "difficulty_history": difficulty_history,
        "final_checkpoint": final_path,
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 4: Hierarchical GATv2 Training (Burn-in + Scaled Chaos)
# ═══════════════════════════════════════════════════════════════


def _build_hierarchical_burnin_corpus():
    """
    Build a burn-in corpus for HierarchicalSchedulingPolicy.
    Uses small Phase 4 graphs (3-5 lanes x 2-3 ops = 6-15 ops) to
    stabilize the local/global branch norms before scaling up.
    """
    from .graph_generator import ProceduralGraphGenerator
    gen = ProceduralGraphGenerator(seed=42)
    return [
        (f"Phase4_burnin_{i}", lambda i=i: gen.generate(phase=4))
        for i in range(10)
    ]


def _build_hierarchical_scaled_corpus():
    """
    Build a scaled-up corpus for HierarchicalSchedulingPolicy.
    Uses Phase 5 graphs (8-20 lanes x 3-8 ops = 50-160 ops) with
    the same parallel lane topology as Phase 4, scaled 10x.
    """
    from .graph_generator import ProceduralGraphGenerator
    gen = ProceduralGraphGenerator(seed=117)
    return [
        (f"Phase5_scaled_{i}", lambda i=i: gen.generate(phase=5))
        for i in range(15)
    ]


def hierarchical_phase4_train(
    num_episodes: int = 2000,
    burnin_episodes: int = 100,
    k: int = 2,
    max_registers: int = 3,
    register_penalty_alpha: float = 1.0,
    learning_rate: float = 5e-4,
    entropy_coef: float = 0.05,
    burnin_entropy_coef: float = 0.08,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 3,
    log_every: int = 25,
    save_every: int = 250,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
    dry_run: bool = False,
    partial_credit_per_issue: float = 0.1,
) -> dict:
    """
    Phase 4: Hierarchical GATv2 Training with Burn-in + Scaled Chaos.

    Initializes HierarchicalSchedulingPolicy from scratch and runs a
    100-episode deterministic burn-in on small Phase 4 graphs to
    stabilize the local/global branch norm ratio. Once the ratio is
    stable in [0.5, 2.0], promotes to scaled Phase 5 graphs (50-160 ops)
    with continuous curricular annealing + adaptive solenoid.

    Burn-in phase (eps 0-99):
      - Small Phase 4 graphs (3-5 lanes x 2-3 ops = 6-15 ops)
      - mem_latency = 3c (deterministic)
      - No port hazards, no jitter
      - Higher entropy (0.08 for eps 0-49, 0.05 for eps 50-99)
      - Track branch_norm_ratio every log interval

    Promotion check (at ep 100):
      - If branch_norm_ratio in [0.5, 2.0] for last 30 episodes -> promote
      - Otherwise -> extend burn-in by 50 episodes, re-check

    Scaled phase (eps 100+):
      - Phase 5 graphs (8-20 lanes x 3-8 ops = 50-160 ops)
      - Continuous annealing via _interpolate_chaos()
      - Adaptive solenoid (KL-triggered bumps)
      - Dynamic entropy scaling

    Args:
        num_episodes: Total training episodes.
        burnin_episodes: Episodes in deterministic burn-in phase.
        k: Issue width.
        max_registers: Physical register limit.
        register_penalty_alpha: Register pressure penalty multiplier.
        learning_rate: Base LR for stratified optimizer.
        entropy_coef: Base entropy coefficient (scaled chaos phase).
        burnin_entropy_coef: Entropy coefficient during burn-in.
        clip_epsilon: PPO clip range.
        ppo_epochs: Gradient steps per collected episode.
        log_every: Metrics logging interval.
        save_every: Checkpoint saving interval.
        checkpoint_dir: Directory for checkpoints.
        device: 'cpu' or 'cuda'.

    Returns:
        dict with full training telemetry including branch norms.
    """
    import os
    import numpy as np

    print()
    print("=" * 66)
    print("  PHASE 4: HIERARCHICAL GATv2 TRAINING")
    print("  100-ep burn-in -> scaled 50-160 op chaos")
    print("=" * 66)

    # ── Build corpora ──────────────────────────────
    burnin_corpus = _build_hierarchical_burnin_corpus()
    scaled_corpus = _build_hierarchical_scaled_corpus()
    print(f"  Burn-in corpus: {len(burnin_corpus)} graph types (6-15 ops each)")
    print(f"  Scaled corpus:  {len(scaled_corpus)} graph types (50-160 ops each)")

    # ── Initialize policy from scratch ──────────────
    policy = HierarchicalSchedulingPolicy(node_feat_dim=20).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: HierarchicalSchedulingPolicy ({n_params:,} params)")

    # ── Stratified optimizer ────────────────────────
    optimizer = configure_stratified_optimizer(policy, base_lr=learning_rate)
    print(f"  base_lr={learning_rate}, burnin_entropy={burnin_entropy_coef}, "
          f"entropy={entropy_coef}")
    print(f"  clip={clip_epsilon}, PPO epochs={ppo_epochs}")
    if dry_run:
        print(f"  DRY RUN: 5 episodes (3 burn-in + 2 chaos) to validate memory footprint")
    print()

    # ── Dry run: override to 5 episodes ────────────
    if dry_run:
        num_episodes = 5
        save_every = 99999  # effectively disable checkpoint saving

    # ── Training state ──────────────────────────────
    baselines: Dict[str, RunningBaseline] = {}
    episode_rewards = []
    episode_spills = []
    episode_reloads = []
    best_reward = -float("inf")
    deadlock_count = 0
    n_cpf_beaten = 0

    episode_kl = []
    episode_cpf_rewards = []
    branch_norm_history = []  # (ep, local_norm, global_norm, ratio)
    reduction_attn_history = []  # (ep, red_global_norm, nonred_global_norm, ratio)

    # ── Burn-in phase: deterministic 3c, small graphs ──
    in_burnin = True
    promoted_ep = None
    burnin_extension = 0

    # ── Scaled phase: adaptive solenoid state ────────
    _cpf_hits_buffer = []
    difficulty_history = []
    current_entropy_coef = entropy_coef
    _last_cpf_env_key = None
    cpf_cache: Dict[str, tuple] = {}

    KLLATLINE_THRESHOLD = 0.001
    KLLATLINE_WINDOW = 30
    BUMP_STEP = 0.15
    _kl_window = []
    _kl_flatline_counter = 0
    _adaptive_bump = 0.0
    _entropy_boost_remaining = 0

    # ── Graph rotation: deterministic sequence ──────
    burnin_idx = 0
    scaled_idx = 0

    try:
        from tqdm import tqdm
        progress = tqdm(range(num_episodes), desc="Phase 4")
    except ImportError:
        progress = range(num_episodes)

    for episode in progress:
        # ═══════════════════════════════════════════════
        #  Phase Detection
        # ═══════════════════════════════════════════════

        if in_burnin and episode >= burnin_episodes + burnin_extension:
            # Check promotion condition
            recent_ratios = [
                r for _, _, _, r in branch_norm_history[-30:]
            ] if len(branch_norm_history) >= 30 else []
            if recent_ratios:
                min_r, max_r = min(recent_ratios), max(recent_ratios)
                if 0.5 <= min_r and max_r <= 2.0:
                    in_burnin = False
                    promoted_ep = episode
                    print(f"\n  === PROMOTED at ep {episode}: "
                          f"branch_norm_ratio [{min_r:.3f}, {max_r:.3f}] -> scaled chaos ===\n")
                else:
                    burnin_extension += 50
                    print(f"  [Burn-in] Extended by 50 eps: "
                          f"ratio [{min_r:.3f}, {max_r:.3f}] not in [0.5, 2.0]")

        # ═══════════════════════════════════════════════
        #  Environment Config
        # ═══════════════════════════════════════════════

        if in_burnin:
            # Deterministic burn-in
            current_lat = 3
            current_unit_limit = {}
            current_lat_dist = {}
            label, g_fn = burnin_corpus[burnin_idx % len(burnin_corpus)]
            burnin_idx += 1

            # Higher entropy for first 50 burn-in episodes
            if episode < 50:
                current_entropy_coef = burnin_entropy_coef
            else:
                current_entropy_coef = entropy_coef
        else:
            # Scaled chaos phase: continuous annealing
            roll_cpf = 0.0
            if _cpf_hits_buffer:
                roll_cpf = sum(_cpf_hits_buffer) / len(_cpf_hits_buffer)

            perf_df = min(1.0, roll_cpf * 2.0)
            ep_floor = min(1.0, (episode - promoted_ep) / 500)
            difficulty_factor = max(perf_df, ep_floor)

            config = _interpolate_chaos(difficulty_factor)
            current_lat = config["mem_latency"]
            current_unit_limit = config["unit_limit"]
            current_lat_dist = config["latency_distribution"]

            # Apply adaptive bump
            bumped_df = difficulty_factor + _adaptive_bump
            effective_df = min(1.0, bumped_df)
            config = _interpolate_chaos(effective_df)
            current_lat = config["mem_latency"]
            current_unit_limit = config["unit_limit"]
            current_lat_dist = config["latency_distribution"]
            difficulty_factor = effective_df

            entropy_boost = 0.1 if _entropy_boost_remaining > 0 else 0.0
            current_entropy_coef = entropy_coef * (1.0 + difficulty_factor + entropy_boost)
            if _entropy_boost_remaining > 0:
                _entropy_boost_remaining -= 1

            # Scaled graph sampling
            label, g_fn = scaled_corpus[scaled_idx % len(scaled_corpus)]
            scaled_idx += 1

            if episode == promoted_ep or (episode + 1 - promoted_ep) % log_every == 0:
                difficulty_history.append(
                    (episode, difficulty_factor, _adaptive_bump)
                )

        # ═══════════════════════════════════════════════
        #  Graph + CPF
        # ═══════════════════════════════════════════════

        graph = g_fn()
        topo = graph.get_topological_order()
        node_names = graph.inputs + topo + graph.outputs

        # CPF baseline
        cpf_key = f"{label}_{current_lat}_{current_unit_limit}_{sorted(current_lat_dist.items())}"
        if cpf_key not in cpf_cache or (not in_burnin and cpf_key != _last_cpf_env_key):
            cpf_sched = schedule_cpf(graph)
            cpf_env = SchedulingGymEnv(
                graph, None, [],
                max_exec_units=k, latency=DEFAULT_LATENCY,
                max_registers=max_registers,
                register_penalty_alpha=register_penalty_alpha,
                mem_latency=current_lat,
                unit_limit=current_unit_limit,
                latency_distribution=current_lat_dist,
                partial_credit_per_issue=partial_credit_per_issue,
            )
            cpf_env.reset()
            for nid in cpf_sched:
                cpf_idx = node_names.index(nid)
                _, _, cpf_done, cpf_info = cpf_env.step(cpf_idx)
                if cpf_done:
                    break
            cpf_r = cpf_info.get("spill_penalty", 0) + cpf_info.get("cycles", 0)
            cpf_cache[cpf_key] = -cpf_r
            _last_cpf_env_key = cpf_key

        cpf_reward = cpf_cache[cpf_key]

        # ═══════════════════════════════════════════════
        #  Training Environment
        # ═══════════════════════════════════════════════

        env = SchedulingGymEnv(
            graph, None, [],
            max_exec_units=k, latency=DEFAULT_LATENCY,
            max_registers=max_registers,
            register_penalty_alpha=register_penalty_alpha,
            mem_latency=current_lat,
            unit_limit=current_unit_limit,
            latency_distribution=current_lat_dist,
            partial_credit_per_issue=partial_credit_per_issue,
        )

        # ═══════════════════════════════════════════════
        #  Per-Label Baseline
        # ═══════════════════════════════════════════════

        if label not in baselines:
            baselines[label] = RunningBaseline(alpha=0.9)
        graph_baseline = baselines[label]

        reward, info = train_scheduling_episode_ppo_v2(
            env, policy, optimizer, graph_baseline,
            entropy_coef=current_entropy_coef,
            reward_scale=1.0,
            clip_epsilon=clip_epsilon,
            ppo_epochs=ppo_epochs,
            device=device,
            track_kl=True,
        )

        # ═══════════════════════════════════════════════
        #  Post-Episode Accounting
        # ═══════════════════════════════════════════════

        if "approx_kl" in info:
            kl_val = info["approx_kl"]
            episode_kl.append(kl_val)
            if not in_burnin:
                _kl_window.append(kl_val)
                if len(_kl_window) > KLLATLINE_WINDOW:
                    _kl_window.pop(0)
                if len(_kl_window) >= 5 and _entropy_boost_remaining <= 0:
                    avg_kl_window = sum(_kl_window) / len(_kl_window)
                    if avg_kl_window < KLLATLINE_THRESHOLD:
                        _kl_flatline_counter += 1
                    else:
                        _kl_flatline_counter = 0
                    if _kl_flatline_counter >= KLLATLINE_WINDOW:
                        _adaptive_bump += BUMP_STEP
                        _kl_flatline_counter = 0
                        _entropy_boost_remaining = 10
                        print(f"  ** KL TRIGGER ep {episode:4d}: bump +{BUMP_STEP:.2f} "
                              f"(total={_adaptive_bump:.2f}, entropy_boost=10eps)")

        # Branch norm tracking (hierarchical policy always provides this)
        if "branch_norm_ratio" in info:
            branch_norm_history.append((
                episode,
                info.get("branch_local_norm", 0),
                info.get("branch_global_norm", 0),
                info["branch_norm_ratio"],
            ))
        # Reduction-node attention tracking
        if "red_attn_ratio" in info:
            reduction_attn_history.append((
                episode,
                info.get("red_global_norm", 0),
                info.get("nonred_global_norm", 0),
                info["red_attn_ratio"],
            ))

        n_spills = sum(1 for e in env.schedule
                       if isinstance(e, tuple) and e[0] == "spill")
        n_reloads = sum(1 for e in env.schedule
                        if isinstance(e, tuple) and e[0] == "reload")

        episode_rewards.append(reward)
        episode_spills.append(n_spills)
        episode_reloads.append(n_reloads)

        if reward > best_reward:
            best_reward = reward
        if reward <= -900:
            deadlock_count += 1

        episode_cpf_rewards.append(cpf_reward)
        if reward >= cpf_reward:
            n_cpf_beaten += 1
            if not in_burnin:
                _cpf_hits_buffer.append(1)
                if len(_cpf_hits_buffer) > 100:
                    _cpf_hits_buffer.pop(0)
            # else: not tracked during burn-in (deterministic)
        elif not in_burnin:
            _cpf_hits_buffer.append(0)
            if len(_cpf_hits_buffer) > 100:
                _cpf_hits_buffer.pop(0)

        # ═══════════════════════════════════════════════
        #  Logging
        # ═══════════════════════════════════════════════

        if (episode + 1) % log_every == 0 or episode < 5:
            recent_r = episode_rewards[-min(log_every, len(episode_rewards)):]
            avg_r = sum(recent_r) / len(recent_r)
            avg_s = (sum(episode_spills[-min(log_every, len(episode_spills)):])
                     / min(log_every, len(episode_spills)))
            avg_kl = (sum(episode_kl[-min(log_every, len(episode_kl)):])
                      / min(log_every, len(episode_kl))) if episode_kl else 0.0

            # Branch norm ratio
            ratio_str = ""
            if branch_norm_history:
                last_ratio = branch_norm_history[-1][3]
                ratio_str = f" | br={last_ratio:.3f}"

            phase_tag = "BURNIN" if in_burnin else "CHAOS"
            ops_tag = "~10" if in_burnin else "~100"

            if isinstance(progress, range):
                bolt_indicator = " ⚡" if _entropy_boost_remaining > 0 else "   "
                df_str = "" if in_burnin else f"df={difficulty_factor:.2f}{bolt_indicator}"
                print(
                    f"  Ep {episode + 1:4d}/{num_episodes} "
                    f"[{phase_tag:6s}] | "
                    f"reward={avg_r:7.1f} | "
                    f"ops={ops_tag:>4s} | "
                    f"kl={avg_kl:.4f}{ratio_str} | "
                    f"spills={avg_s:.1f}/ep | "
                    f"dead={deadlock_count:3d} | "
                    f"{label}"
                )
                if df_str:
                    print(f"         {df_str}")
            else:
                progress.set_postfix({
                    "phase": phase_tag,
                    "avg": f"{avg_r:.1f}",
                    "ratio": f"{last_ratio:.3f}" if branch_norm_history else "-",
                    "dead": deadlock_count,
                })

        # ═══════════════════════════════════════════════
        #  Save Checkpoint
        # ═══════════════════════════════════════════════

        if (episode + 1) % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"phase4_ep{episode + 1}.pt",
            )
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "phase": 4,
                "episode": episode + 1,
                "in_burnin": in_burnin,
                "promoted_ep": promoted_ep,
                "rewards": episode_rewards,
                "spill_counts": episode_spills,
                "reload_counts": episode_reloads,
                "best_reward": best_reward,
                "branch_norm_history": branch_norm_history,
                "reduction_attn_history": reduction_attn_history,
                "difficulty_history": difficulty_history,
                "config": {
                    "k": k,
                    "max_registers": max_registers,
                    "register_penalty_alpha": register_penalty_alpha,
                    "learning_rate": learning_rate,
                    "entropy_coef": entropy_coef,
                    "burnin_entropy_coef": burnin_entropy_coef,
                    "clip_epsilon": clip_epsilon,
                    "ppo_epochs": ppo_epochs,
                },
            }, ckpt_path)
            print(f"  [Checkpoint] {ckpt_path}")

    # ── Final summary ────────────────────────────────
    last_200 = episode_rewards[-200:] if len(episode_rewards) >= 200 else episode_rewards
    avg_last = sum(last_200) / len(last_200)
    total_spills = sum(episode_spills)
    total_reloads = sum(episode_reloads)

    print()
    print(f"  PHASE 4 COMPLETE")
    print(f"  {'=' * 40}")
    print(f"  Best reward:       {best_reward:7.1f}")
    print(f"  Avg last 200:      {avg_last:7.1f}")
    print(f"  Deadlocks:         {deadlock_count:4d}/{num_episodes}")
    print(f"  @CPF beaten:       {n_cpf_beaten:4d}/{num_episodes}")
    print(f"  Total spills:      {total_spills:4d}")
    print(f"  Total reloads:     {total_reloads:4d}")
    print(f"  Promoted at ep:    {promoted_ep if promoted_ep else 'NEVER'}")

    if branch_norm_history:
        first_ratio = branch_norm_history[0][3]
        last_ratio = branch_norm_history[-1][3]
        print(f"  Branch norm ratio: {first_ratio:.3f} -> {last_ratio:.3f}")

    if difficulty_history:
        df_start = difficulty_history[0][1]
        df_end = difficulty_history[-1][1]
        roll_cpf = (sum(_cpf_hits_buffer) / len(_cpf_hits_buffer)
                     if _cpf_hits_buffer else 0.0)
        print(f"  Difficulty:        {df_start:.2f} -> {df_end:.2f} "
              f"(roll @CPF={roll_cpf:.3f})")

    # ── Save final model ─────────────────────────────
    final_path = os.path.join(checkpoint_dir, "phase4_final.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "phase": 4,
        "episode": num_episodes,
        "promoted_ep": promoted_ep,
        "rewards": episode_rewards,
        "spill_counts": episode_spills,
        "reload_counts": episode_reloads,
        "best_reward": best_reward,
        "branch_norm_history": branch_norm_history,
        "difficulty_history": difficulty_history,
        "config": {"policy_type": "HierarchicalSchedulingPolicy"},
    }, final_path)
    print(f"  Final checkpoint: {final_path}")

    return {
        "rewards": episode_rewards,
        "spill_counts": episode_spills,
        "reload_counts": episode_reloads,
        "best_reward": best_reward,
        "deadlock_count": deadlock_count,
        "n_cpf_beaten": n_cpf_beaten,
        "promoted_ep": promoted_ep,
        "branch_norm_history": branch_norm_history,
        "difficulty_history": difficulty_history,
        "final_checkpoint": final_path,
    }


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════╗")
    print("║  NN Compiler v0: Batch-Distribution Training ║")
    print("╚═══════════════════════════════════════════════╝")

    # ── ReLU batch-distribution setup ──────────────
    test_suite = [
        ({"x": 5.0, "zero": 0.0}, {"out": 5.0}),    # positive
        ({"x": -5.0, "zero": 0.0}, {"out": 0.0}),   # negative
        ({"x": 0.0, "zero": 0.0}, {"out": 0.0}),    # boundary
        ({"x": -0.1, "zero": 0.0}, {"out": 0.0}),   # small negative
        ({"x": 0.1, "zero": 0.0}, {"out": 0.1}),    # small positive
    ]

    # VM binary
    vm_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "target", "release", "dfasm",
    )
    if os.name == "nt":
        vm_path += ".exe"

    if not os.path.exists(vm_path):
        print(f"ERROR: VM binary not found at {vm_path}")
        sys.exit(1)

    bridge = VMBridge(vm_path)

    # ── Graph ───────────────────────────────────────
    graph = create_relu_graph()
    topo_order = graph.get_topological_order()
    print(f"Graph: {len(graph.inputs)} inputs, "
          f"{len(topo_order)} ops, {len(graph.outputs)} outputs")
    print(f"Topo order: {topo_order}")
    print(f"Test suite: {len(test_suite)} cases (positive, negative, boundary)")

    # ── Environment ─────────────────────────────────
    env = DataflowGymEnv(graph, bridge, test_suite)

    # ── Policy ──────────────────────────────────────
    policy = GNNPolicy(node_feat_dim=9, hidden_dim=64, num_layers=2)
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")

    # ── Optimal baseline (correct ReLU wiring) ──────
    env.reset()
    env.step({"target_destinations": [("sw", 1, "send")]})
    env.step({"target_destinations": [
        ("merge", 0, "send_true"),
        ("zero_path", 0, "send_false"),
    ]})
    env.step({"target_destinations": [("merge", 1, "send")]})
    _, optimal_reward, _, optimal_info = env.step({"target_destinations": []})

    # Show breakdown of optimal baseline
    print(f"\nOptimal batch reward: {optimal_reward:.1f}")
    for case in optimal_info.get("cases", []):
        c = case["info"]
        print(f"  x={case['inputs']['x']:5.1f} -> "
              f"reward={case['reward']:.1f} "
              f"(cycles={c.get('cycles','?')}, queue={c.get('max_queue','?')})")

    # ── Train ───────────────────────────────────────
    print(f"\nTraining 500 episodes...\n")
    rewards, steps = train(
        env, policy, optimizer,
        num_episodes=500, log_every=50, entropy_coef=0.1,
    )

    # ── Results ─────────────────────────────────────
    print(f"\n{'=' * 50}")
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    avg = sum(last_100) / len(last_100)
    deadlocks = sum(1 for r in last_100 if r <= -900)

    print(f"Best:     {max(rewards):.1f}")
    print(f"Optimal:  {optimal_reward:.1f}")
    print(f"Avg-100:  {avg:.1f}")
    print(f"Deadlock: {deadlocks}/{len(last_100)}")
