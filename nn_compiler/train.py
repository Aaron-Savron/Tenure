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
    encode_observation,
    encode_scheduling_obs,
    decode_action,
    collect_switch_routings,
)
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
            final_reward = -1000.0
            final_info = {"error": "No valid actions (empty ready set)"}
            break

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
            final_reward = -1000.0
            final_info = {"error": "No valid actions (empty ready set)"}
            break

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
            final_reward = -1000.0
            final_info = {"error": "No valid actions (empty action mask)"}
            break

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
                                     device="cpu"):
    """
    v2 PPO training step for the flat 2N action space.

    Uses the clipped surrogate objective on the unified Issue/Reload/Spill
    distribution. The GAT body may be frozen (requires_grad=False) — this
    function works correctly either way since .backward() only flows through
    trainable parameters.
    """
    reward, old_log_probs, observations, actions, info = \
        collect_scheduling_episode_ppo_v2(env, policy, device)

    scaled_reward = reward * reward_scale
    advantage = scaled_reward - baseline.get()
    baseline.update(scaled_reward)

    n_steps = len(old_log_probs)
    if n_steps == 0:
        return reward, info

    for epoch in range(ppo_epochs):
        epoch_losses = []
        for obs, old_lp, act in zip(observations, old_log_probs, actions):
            probs = policy._get_priority_distribution(obs)
            mask = obs.get("action_mask", obs.get("ready_mask"))
            if not mask.any() or probs.sum() == 0:
                continue
            dist = Categorical(probs)
            action_tensor = torch.tensor(act, device=device)
            new_lp = dist.log_prob(action_tensor)

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
) -> dict:
    """
    Phase 1: Frozen Adaptation Training.

    Loads the surgically-transplanted v2_init checkpoint, freezes the
    GAT body, and trains ONLY input_proj and priority_head on a corpus
    of small DL subgraphs with low memory latency (mem_latency=3).

    The model learns the syntax of the 2N action space (Issue vs Spill)
    without risking corruption of the pre-trained GAT attention maps.
    """
    import os
    from itertools import cycle

    print("=" * 66)
    print("  PHASE 1: FROZEN ADAPTATION")
    print("  Transplant v1 GAT body -> train new 2N action head")
    print("=" * 66)

    # Build corpus (deferred import to avoid circular deps)
    corpus = _build_frozen_corpus()

    # ── Build policy ────────────────────────────────
    policy = SchedulingPolicy(node_feat_dim=15).to(device)
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
    print()

    # ── Training state ──────────────────────────────
    baseline = RunningBaseline(alpha=0.9)
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

        reward, info = train_scheduling_episode_ppo_v2(
            env, policy, optimizer, baseline,
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

    final_path = os.path.join(checkpoint_dir, "phase1_final.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "phase": 1,
        "episode": num_episodes,
        "rewards": episode_rewards,
        "best_reward": best_reward,
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
