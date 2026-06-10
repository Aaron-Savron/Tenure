"""
curriculum_train.py -- Curriculum training loop for the NN Compiler.

Scales the policy through three phases, sampling a fresh random graph
from ProceduralGraphGenerator at each episode:

  Phase 1 (Linear Pipes):  3-4 ops, Add/Mul chain
  Phase 2 (Branching):     4-6 ops, Add/Mul with fan-out
  Phase 3 (Conditionals):  6-8 ops, full ISA with Switch/Merge/CmpGeZ

Because the graph topology changes every episode, the policy cannot
memorize a fixed routing — it must learn the *general* rules of
dataflow compilation. Reward convergence across unseen topologies
is the measure of generalization.
"""

import os
import sys
import hashlib
import torch
import torch.optim as optim
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from itertools import count
import numpy as np

from .compiler_env import VMBridge, DataflowGymEnv, ComputeGraph, SchedulingGymEnv, DEFAULT_LATENCY
from .graph_generator import ProceduralGraphGenerator, PHASE_CONFIG
from .policy import GNNPolicy, SchedulingPolicy, encode_scheduling_obs
from .scheduler_baseline import schedule_cpf
from .train import RunningBaseline, train_episode, train_scheduling_episode_ppo


# ═══════════════════════════════════════════════════════════════
#  Graph hashing (per-MDP baseline isolation)
# ═══════════════════════════════════════════════════════════════

def graph_to_hash(graph: ComputeGraph) -> str:
    """
    Produce a deterministic hash that uniquely identifies a graph's topology.

    The hash is based on node types and their dependency edges (sorted for
    determinism). This is used as a key in the per-graph baseline dict so
    that each distinct MDP gets its own EMA baseline, preventing the
    destructive gradient interference from a shared scalar baseline.
    """
    parts = []
    for node_id in sorted(graph.nodes.keys()):
        node = graph.nodes[node_id]
        deps = sorted(node.dependencies, key=lambda x: (x[0], x[1]))
        deps_str = ";".join(f"{p}:{s}" for p, s in deps)
        parts.append(f"{node_id}({node.type})[{deps_str}]")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
#  Default configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "episodes_per_phase": {
        1: 300,    # Linear Pipes
        2: 500,    # Branching
        3: 1000,   # Conditionals
    },
    "batch_size": 3,          # test cases per graph
    "entropy_coef": 0.1,
    "reward_scale": 1.0,
    "log_every": 50,
    "learning_rate": 5e-4,
    "hidden_dim": 64,
    "num_layers": 2,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "save_every": 100,
}


# ═══════════════════════════════════════════════════════════════
#  VM binary discovery
# ═══════════════════════════════════════════════════════════════

def find_vm_binary() -> str:
    """Locate the Rust VM binary relative to the project root."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vm_path = os.path.join(base, "target", "release", "dfasm")
    if os.name == "nt":
        vm_path += ".exe"
    return vm_path


def verify_vm(vm_path: str) -> bool:
    """Check that the VM binary exists."""
    if not os.path.exists(vm_path):
        return False
    if os.name != "nt" and not os.access(vm_path, os.X_OK):
        return False
    return True


# ═══════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(
    policy: GNNPolicy,
    optimizer: optim.Optimizer,
    phase: int,
    episode: int,
    rewards: List[float],
    best_reward: float,
    config: dict,
    path: str,
):
    """Save training state to resume later."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "phase": phase,
        "episode": episode,
        "rewards": rewards,
        "best_reward": best_reward,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)
    print(f"  [Checkpoint saved] {path}")


def load_checkpoint(
    path: str, policy: GNNPolicy, optimizer: optim.Optimizer, device: str = "cpu"
) -> dict:
    """Load training state from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  [Checkpoint loaded] phase={checkpoint['phase']}, "
          f"episode={checkpoint['episode']}, "
          f"best={checkpoint['best_reward']:.1f}")
    return checkpoint


# ═══════════════════════════════════════════════════════════════
#  Phase training
# ═══════════════════════════════════════════════════════════════

def train_phase(
    policy: GNNPolicy,
    optimizer: optim.Optimizer,
    generator: ProceduralGraphGenerator,
    bridge: VMBridge,
    phase: int,
    num_episodes: int,
    batch_size: int = 3,
    entropy_coef: float = 0.1,
    reward_scale: float = 1.0,
    log_every: int = 50,
    save_every: int = 100,
    checkpoint_dir: Optional[str] = None,
    device: str = "cpu",
    start_episode: int = 0,
    initial_rewards: Optional[List[float]] = None,
    graph_pool_size: int = 0,
    pool_pass_rate_threshold: float = 0.8,
) -> Tuple[List[float], List[int], float]:
    """
    Train the policy for a single curriculum phase.

    By default, each episode samples a fresh random graph from
    ProceduralGraphGenerator. This forces generalization but creates
    a signal-to-noise problem for REINFORCE when the action space is
    large (Phase 3): the policy's gradient for a successful Switch
    routing doesn't transfer to the next episode's different graph.

    When graph_pool_size > 0 (recommended for Phase 3), training uses
    a fixed pool of graphs. The policy sees each graph multiple times,
    building consistent gradient momentum. After each full pass through
    the pool, if the pass rate exceeds pool_pass_rate_threshold, a
    fresh pool is generated. This turns chaotic procedural noise into
    a sequence of tractable stepping stones.

    Args:
        graph_pool_size: If > 0, train on a fixed pool of this many
                         graphs, cycling through them. When pass rate
                         exceeds threshold, generate a new pool.
        pool_pass_rate_threshold: Fraction of successful episodes
                                  needed in a pool pass to trigger
                                  rotation (default 0.8 = 80%).

    Returns:
        (episode_rewards, graph_sizes, best_reward)
    """
    policy.train()
    # Per-graph contextual baselines keyed by topological hash.
    # Each graph (MDP) gets its own EMA baseline, preventing the
    # destructive gradient interference from a shared scalar baseline.
    baselines: Dict[str, RunningBaseline] = {}
    episode_rewards: List[float] = initial_rewards if initial_rewards else []
    graph_sizes: List[int] = []
    best_reward = max(episode_rewards) if episode_rewards else -float("inf")
    deadlock_count = 0

    pool_names = [t.split(".")[-1] for t in PHASE_CONFIG[phase]["op_pool"]]
    phase_label = f"Phase {phase} [{', '.join(pool_names)}]"

    try:
        from tqdm import tqdm
        progress = tqdm(range(start_episode, num_episodes),
                        desc=phase_label, unit="ep")
    except ImportError:
        progress = range(start_episode, num_episodes)

    # ── Graph pool management (topological persistence) ──
    # Generate a pool of graphs and cycle through them, rotating
    # when pass rate exceeds the threshold.
    use_pool = (graph_pool_size > 0)
    pool: List[Tuple[ComputeGraph, List]] = []
    pool_idx = 0
    pool_cycle_number = 0
    pool_cycle_rewards: List[float] = []  # rewards in current pool pass

    def _build_pool():
        """Generate a fresh pool of phase-appropriate graphs."""
        nonlocal pool, pool_idx, pool_cycle_rewards, pool_cycle_number
        pool = []
        for _ in range(graph_pool_size):
            g = generator.generate(phase=phase)
            suite = generator.generate_test_suite(g, batch_size)
            pool.append((g, suite))
        pool_idx = 0
        pool_cycle_number = 0
        pool_cycle_rewards = []
        # Clear old baselines when pool rotates — the new pool has
        # different topologies, so old baseline entries are stale.
        baselines.clear()
        print(f"  [Pool] New {len(pool)}g pool, cycle 0 starting...")
        # Show graph hashes for traceability
        for i, (g, _) in enumerate(pool):
            gh = graph_to_hash(g)[:8]
            topo = g.get_topological_order()
            types = [g.nodes[n].type for n in topo]
            print(f"    G{i}: hash={gh}  types={types}")
        try:
            progress.set_postfix_str(f"pool={len(pool)}g cyc=0")
        except (AttributeError, TypeError):
            pass

    if use_pool:
        _build_pool()

    for episode in progress:
        # ── Get graph (pool or fresh) ────────────────────
        if use_pool:
            graph, test_suite = pool[pool_idx]
            pool_idx += 1
            if pool_idx >= len(pool):
                pool_idx = 0
                pool_cycle_number += 1
                # Evaluate pass rate for the completed cycle
                if len(pool_cycle_rewards) > 0:
                    cycle_pass = sum(
                        1 for r in pool_cycle_rewards if r > -50
                    ) / len(pool_cycle_rewards)
                    cycle_dead = sum(
                        1 for r in pool_cycle_rewards if r <= -900
                    ) / len(pool_cycle_rewards)
                    if cycle_pass >= pool_pass_rate_threshold:
                        print(
                            f"\n  [Pool] Cycle {pool_cycle_number} complete: "
                            f"pass={cycle_pass:.0%} dead={cycle_dead:.0%} "
                            f">= threshold ({pool_pass_rate_threshold:.0%})"
                            f" -> ROTATING to fresh graphs..."
                        )
                        _build_pool()
                    else:
                        print(
                            f"  [Pool] Cycle {pool_cycle_number} complete: "
                            f"pass={cycle_pass:.0%} dead={cycle_dead:.0%} "
                            f"< threshold ({pool_pass_rate_threshold:.0%})"
                            f" -> repeating pool"
                        )
                        try:
                            progress.set_postfix_str(
                                f"pool={len(pool)}g cyc={pool_cycle_number}"
                            )
                        except (AttributeError, TypeError):
                            pass
                pool_cycle_rewards = []
        else:
            graph = generator.generate(phase=phase)
            test_suite = generator.generate_test_suite(graph, batch_size)

        # ── Resolve per-graph baseline ──────────────────
        # Each distinct topology gets its own EMA baseline.
        # This isolates each MDP's advantage signal and prevents
        # cross-graph gradient interference.
        graph_hash = graph_to_hash(graph)
        if graph_hash not in baselines:
            baselines[graph_hash] = RunningBaseline(alpha=0.9)
        graph_baseline = baselines[graph_hash]

        env = DataflowGymEnv(graph, bridge, test_suite)
        graph_size = len(env.topo_order)

        # ── Train one episode ────────────────────────────
        reward, info = train_episode(
            env, policy, optimizer, graph_baseline,
            entropy_coef=entropy_coef,
            reward_scale=reward_scale,
            device=device,
        )

        episode_rewards.append(reward)
        graph_sizes.append(graph_size)

        if reward > best_reward:
            best_reward = reward
        if reward <= -900:
            deadlock_count += 1
        if use_pool:
            pool_cycle_rewards.append(reward)

        # ── Logging ──────────────────────────────────────
        local_ep = episode - start_episode + 1
        # Show average of all active baselines (avoids misleading single-MDP value)
        avg_bl = sum(b.get() for b in baselines.values()) / max(len(baselines), 1)

        if isinstance(progress, range):
            if (episode + 1) % log_every == 0:
                recent = episode_rewards[-log_every:]
                avg = sum(recent) / len(recent) if recent else 0.0
                recent_deadlocks = sum(1 for r in recent if r <= -900)
                print(
                    f"  [{phase_label}] "
                    f"Ep {episode + 1:4d}/{num_episodes} | "
                    f"avg: {avg:7.1f} | "
                    f"best: {best_reward:7.1f} | "
                    f"dead: {recent_deadlocks:2d}/{log_every} | "
                    f"sz: {graph_size} | "
                    f"bl: {avg_bl:.1f}"
                )
        else:
            recent = episode_rewards[-50:]
            avg = sum(recent) / len(recent) if recent else 0.0
            progress.set_postfix({
                "avg": f"{avg:.1f}",
                "best": f"{best_reward:.1f}",
                "bl": f"{avg_bl:.1f}",
                "dead": f"{deadlock_count}/{local_ep}",
            })

        # ── Save checkpoint ──────────────────────────────
        if checkpoint_dir and (episode + 1) % save_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"phase{phase}_ep{episode + 1}.pt",
            )
            save_checkpoint(
                policy, optimizer, phase, episode + 1,
                episode_rewards, best_reward, {}, ckpt_path,
            )

    # ── Phase summary ────────────────────────────────────
    last_50 = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
    avg = sum(last_50) / len(last_50) if last_50 else 0.0
    total_episodes = num_episodes - start_episode

    print(f"\n  [{phase_label} complete] "
          f"Best: {best_reward:.1f}, "
          f"Avg-50: {avg:.1f}, "
          f"Deadlock: {deadlock_count}/{total_episodes} "
          f"({100 * deadlock_count / max(total_episodes, 1):.1f}%), "
          f"Size: {min(graph_sizes)}-{max(graph_sizes)} ops")

    return episode_rewards, graph_sizes, best_reward


# ═══════════════════════════════════════════════════════════════
#  Curriculum runner
# ═══════════════════════════════════════════════════════════════

def curriculum_train(
    policy: GNNPolicy,
    optimizer: optim.Optimizer,
    generator: ProceduralGraphGenerator,
    bridge: VMBridge,
    config: Optional[dict] = None,
    device: str = "cpu",
) -> Dict[int, dict]:
    """
    Run the full curriculum across all three phases.

    The policy is trained sequentially on Phase 1 → Phase 2 → Phase 3,
    with its weights carried forward between phases. The curriculum
    scales complexity: linear chains → fan-out branching → conditionals.

    Returns:
        {phase: {"rewards": [...], "sizes": [...], "best": float}}
    """
    if config is None:
        config = DEFAULT_CONFIG

    phase_results: Dict[int, dict] = {}

    for phase in [1, 2, 3]:
        num_episodes = config["episodes_per_phase"].get(phase, 500)
        batch_size = config.get("batch_size", 3)
        checkpoint_dir = config.get("checkpoint_dir")

        pool_names = [t.split(".")[-1] for t in PHASE_CONFIG[phase]["op_pool"]]
        print(f"\n{'=' * 66}")
        print(f"  PHASE {phase}: {', '.join(pool_names)} "
              f"({num_episodes} episodes)")
        print(f"  Graph: {PHASE_CONFIG[phase]['num_ops'][0]}-"
              f"{PHASE_CONFIG[phase]['num_ops'][1]} ops, "
              f"{PHASE_CONFIG[phase]['num_inputs'][0]}-"
              f"{PHASE_CONFIG[phase]['num_inputs'][1]} inputs")
        print(f"  Test batch: {batch_size} cases per graph")
        print(f"{'=' * 66}")

        rewards, sizes, best = train_phase(
            policy=policy,
            optimizer=optimizer,
            generator=generator,
            bridge=bridge,
            phase=phase,
            num_episodes=num_episodes,
            batch_size=batch_size,
            entropy_coef=config.get("entropy_coef", 0.1),
            reward_scale=config.get("reward_scale", 1.0),
            log_every=config.get("log_every", 50),
            save_every=config.get("save_every", 100),
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

        phase_results[phase] = {
            "rewards": rewards,
            "sizes": sizes,
            "best": best,
            "avg_last_50": (
                sum(rewards[-50:]) / min(len(rewards[-50:]), 50)
                if rewards else 0.0
            ),
        }

    return phase_results


# ═══════════════════════════════════════════════════════════════
#  Progressive Scheduling Curriculum (Phase 4+)
# ═══════════════════════════════════════════════════════════════

PROGRESSIVE_DEFAULTS = {
    "num_episodes": 5000,
    "initial_pool_size": 5,
    "max_candidates": 50,
    "val_set_size": 5,
    "k": 2,
    "entropy_coef": 0.07,
    "clip_epsilon": 0.2,
    "ppo_epochs": 3,
    "floor": 80,
    "grad_threshold": 0.90,
    "log_every": 200,
    "save_every": 500,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "learning_rate": 5e-4,
    "max_registers": None,
    "register_penalty_alpha": 0.0,
}


def progressive_schedule_train(
    policy: SchedulingPolicy,
    optimizer: optim.Optimizer,
    generator: ProceduralGraphGenerator,
    num_episodes: int = 5000,
    initial_pool_size: int = 5,
    max_candidates: int = 50,
    val_set_size: int = 5,
    k: int = 2,
    entropy_coef: float = 0.07,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 3,
    floor: int = 80,
    grad_threshold: float = 0.90,
    log_every: int = 200,
    save_every: int = 500,
    checkpoint_dir: Optional[str] = None,
    device: str = "cpu",
    max_registers: Optional[int] = None,
    register_penalty_alpha: float = 0.0,
) -> dict:
    """
    Progressive scheduling curriculum for K-issue microarchitectures.

    Trains a SchedulingPolicy using PPO-lite on a dynamically-rotating pool
    of procedural graphs. Graphs are sorted by difficulty (CPF cycles at K)
    and enter the active pool from a waiting list. When the policy achieves
    @CPF mastery >= grad_threshold for >= floor episodes, the graph graduates
    and the next-hardest graph is promoted from the waiting list.

    A held-out validation set (never trained on) is evaluated every log_every
    episodes to measure zero-shot generalization.

    Args:
        policy: SchedulingPolicy instance (default uses 4-layer 256-dim GAT).
        optimizer: torch optimizer.
        generator: ProceduralGraphGenerator for graph creation.
        num_episodes: total training episodes.
        initial_pool_size: number of graphs in the active pool at start.
        max_candidates: total candidate graphs to generate and sort.
        val_set_size: number of graphs held out for validation.
        k: issue width for the scheduling simulator.
        entropy_coef: entropy bonus coefficient for PPO-lite.
        clip_epsilon: PPO clip range.
        ppo_epochs: gradient steps per collected episode.
        floor: minimum episodes before a graph can graduate.
        grad_threshold: @CPF mastery threshold for graduation.
        log_every: validation and metrics logging interval.
        save_every: checkpoint saving interval.
        checkpoint_dir: directory for checkpoints (None = no saves).
        device: "cpu" or "cuda".

    Returns:
        dict with keys:
          - "rewards": list of per-episode rewards
          - "graduated": list of (ep, gid, n_ops, mastery) tuples
          - "val_history": list of (ep, avg_diff, diffs_list, pass_count)
          - "num_graduates": total graphs graduated
          - "best_reward": best episode reward seen
          - "deadlock_count": total episodes with reward <= -900
    """
    def compute_cpf(graph):
        sched = schedule_cpf(graph)
        env = SchedulingGymEnv(graph, None, [], max_exec_units=k, latency=DEFAULT_LATENCY,
                               max_registers=max_registers,
                               register_penalty_alpha=register_penalty_alpha)
        env.reset()
        for nid in sched:
            _, r, d, info = env.step(nid)
        return info.get("cycles", 0), r

    def evaluate(graph):
        """Evaluate policy on a held-out graph, return (cycles, reward)."""
        env = SchedulingGymEnv(graph, None, [], max_exec_units=k, latency=DEFAULT_LATENCY,
                               max_registers=max_registers,
                               register_penalty_alpha=register_penalty_alpha)
        env.reset()
        while not env.done:
            obs = encode_scheduling_obs(env)
            obs_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in obs.items()}
            nidx, _ = policy(obs_dev)
            if nidx is None:
                break
            _, pol_r, _, pol_info = env.step(obs["node_names"][nidx])
        return pol_info.get("cycles", 0), pol_r

    policy.train()

    # ── Generate candidate pool, sort by difficulty ──
    print(f"Generating {max_candidates} candidate graphs...")
    candidates = []
    for i in range(max_candidates):
        g = generator.generate(phase=4)
        c, r = compute_cpf(g)
        candidates.append((g, len(g.get_topological_order()), c, r))
    candidates.sort(key=lambda x: x[2])  # sort by CPF cycles (easiest first)

    # Held-out validation set
    val_set = candidates[-val_set_size:]
    candidates = candidates[:-val_set_size]

    # Active pool + waiting list
    gid_counter = count()
    cpf_cache: Dict[int, Tuple[int, float]] = {}

    def make_entry(g, n_ops, cpf_c, cpf_r):
        gid = next(gid_counter)
        cpf_cache[gid] = (cpf_c, cpf_r)
        return (g, gid, n_ops)

    pool = [make_entry(g, n, c, r) for g, n, c, r in candidates[:initial_pool_size]]
    waiting = [make_entry(g, n, c, r) for g, n, c, r in candidates[initial_pool_size:]]

    print(f"\nProgressive curriculum:")
    print(f"  Active pool: {len(pool)} graphs (easiest)")
    print(f"  Waiting list: {len(waiting)} graphs")
    print(f"  Validation set: {len(val_set)} graphs (held out)")
    print(f"  K={k}, entropy={entropy_coef}, floor={floor}, grad_threshold={grad_threshold:.0%}")
    print(f"  Total episodes: {num_episodes}")

    for g, gid, n_ops in pool:
        print(f"  G{gid:3d}: {n_ops:2d}ops CPF={cpf_cache[gid][0]}c")

    # ── Training state ───────────────────────────────
    baselines: Dict[int, RunningBaseline] = {}
    rewards: List[float] = []
    graduated: List[Tuple[int, int, int, float]] = []
    per_graph_eps: Dict[int, list] = {}
    val_history: List[Tuple[int, float, list, int]] = []
    deadlock_count = 0
    best_reward = -float("inf")

    def get_mastery(gid):
        entries = per_graph_eps.get(gid, [])
        if not entries:
            return 0.0
        return sum(e[2] for e in entries[-50:]) / len(entries[-50:])

    # ── Main training loop ───────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING")
    print(f"{'='*70}\n")

    for ep in range(num_episodes):
        if not pool:
            print(f"Ep {ep}: active pool empty — done.")
            break

        # Performance-weighted sampling
        mastery = {gid: get_mastery(gid) for _, gid, _ in pool}
        weights = {gid: max(0.1, 1.0 - m) for gid, m in mastery.items()}
        total = sum(weights.values())
        rng = np.random.RandomState(ep * 117 + 42)
        probs = [weights[gid] / total for _, gid, _ in pool]
        idx = rng.choice(len(pool), p=probs)
        g, gid, n_ops = pool[idx]

        if gid not in baselines:
            baselines[gid] = RunningBaseline(alpha=0.9)
        bl = baselines[gid]

        env = SchedulingGymEnv(g, None, [], max_exec_units=k, latency=DEFAULT_LATENCY,
                               max_registers=max_registers,
                               register_penalty_alpha=register_penalty_alpha)
        reward, info = train_scheduling_episode_ppo(
            env, policy, optimizer, bl,
            entropy_coef=entropy_coef,
            reward_scale=1.0,
            clip_epsilon=clip_epsilon,
            ppo_epochs=ppo_epochs,
            device=device,
        )

        rewards.append(reward)

        if reward > best_reward:
            best_reward = reward
        if reward <= -900:
            deadlock_count += 1

        # Track @CPF hits
        if gid not in per_graph_eps:
            per_graph_eps[gid] = []
        cpf_c, cpf_r = cpf_cache[gid]
        cpf_hit = 1 if reward >= cpf_r - 0.5 else 0
        per_graph_eps[gid].append((ep, reward, cpf_hit))

        # Graduation check
        mg = get_mastery(gid)
        if mg >= grad_threshold and len(per_graph_eps[gid]) >= floor:
            graduated.append((ep, gid, n_ops, mg))
            pool.pop(idx)
            if gid in baselines:
                del baselines[gid]

            if waiting:
                new_entry = waiting.pop(0)
                pool.append(new_entry)
                print(
                    f"  Ep {ep:5d}: G{gid:3d} GRADUATED ({mg*100:.0f}%) -> "
                    f"G{new_entry[1]:3d} promoted ({new_entry[2]}ops, "
                    f"CPF={cpf_cache[new_entry[1]][0]}c)"
                )
            else:
                print(f"  Ep {ep:5d}: G{gid:3d} GRADUATED ({mg*100:.0f}%) — waiting empty!")

        # Validation and logging
        if ep % log_every == 0 or ep == num_episodes - 1:
            val_diffs = []
            for vg, v_ops, v_cpf_c, v_cpf_r in val_set:
                pol_c, pol_r = evaluate(vg)
                val_diffs.append(pol_c - v_cpf_c)
            avg_diff = sum(val_diffs) / len(val_diffs)
            pass_count = sum(1 for d in val_diffs if d <= 0)
            val_history.append((ep, avg_diff, val_diffs, pass_count))

            recent_r = rewards[-log_every:] if len(rewards) >= log_every else rewards
            avg_r = sum(recent_r) / len(recent_r)
            pool_str = ", ".join(
                f"G{gid}={get_mastery(gid)*100:.0f}%" for _, gid, _ in pool
            )
            print(
                f"Ep {ep:5d} | reward={avg_r:6.1f} | "
                f"pool={len(pool)}g | grad={len(graduated)} | "
                f"val={avg_diff:+3.0f}c ({pass_count}/{len(val_set)} PASS)"
            )
            if pool_str:
                print(f"        active: {pool_str}")

        # Save checkpoint
        if checkpoint_dir and (ep + 1) % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"progressive_ep{ep + 1}.pt",
            )
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode": ep + 1,
                "rewards": rewards,
                "graduated": graduated,
                "val_history": val_history,
                "best_reward": best_reward,
                "config": {
                    "k": k,
                    "entropy_coef": entropy_coef,
                    "clip_epsilon": clip_epsilon,
                    "ppo_epochs": ppo_epochs,
                    "floor": floor,
                    "grad_threshold": grad_threshold,
                },
                "timestamp": datetime.now().isoformat(),
            }, ckpt_path)
            print(f"  [Checkpoint] {ckpt_path}")

    return {
        "rewards": rewards,
        "graduated": graduated,
        "val_history": val_history,
        "num_graduates": len(graduated),
        "best_reward": best_reward,
        "deadlock_count": deadlock_count,
        "pool_end": len(pool),
    }


# ═══════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Curriculum training for the NN Compiler policy."
    )
    parser.add_argument(
        "--phases", type=int, nargs="+", default=[1, 2, 3],
        help="Phases to train (e.g., --phases 1 2)"
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=None,
        help="Episodes per phase (e.g., --episodes 300 500 1000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=3,
        help="Test cases per graph"
    )
    parser.add_argument(
        "--entropy", type=float, default=0.1,
        help="Entropy bonus coefficient"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden", type=int, default=64,
        help="GNN hidden dimension"
    )
    parser.add_argument(
        "--layers", type=int, default=2,
        help="Number of GAT layers"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to train on (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--save-every", type=int, default=100,
        help="Save checkpoint every N episodes per phase"
    )
    parser.add_argument(
        "--log-every", type=int, default=50,
        help="Log metrics every N episodes"
    )

    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ── Random seed ──────────────────────────────────────
    torch.manual_seed(args.seed)

    # ── VM binary ────────────────────────────────────────
    vm_path = find_vm_binary()
    if not verify_vm(vm_path):
        print(f"ERROR: VM binary not found at {vm_path}")
        print("Build with: cargo build --release")
        sys.exit(1)
    print(f"VM:      {vm_path}")
    bridge = VMBridge(vm_path)

    # ── Generator ────────────────────────────────────────
    generator = ProceduralGraphGenerator(seed=args.seed)

    # ── Policy ───────────────────────────────────────────
    policy = GNNPolicy(
        node_feat_dim=9,
        hidden_dim=args.hidden,
        num_layers=args.layers,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    print(f"Policy:  {sum(p.numel() for p in policy.parameters())} parameters")

    # ── Configuration ────────────────────────────────────
    episodes_per_phase = {1: 300, 2: 500, 3: 1000}
    if args.episodes:
        for i, phase in enumerate([1, 2, 3]):
            if i < len(args.episodes):
                episodes_per_phase[phase] = args.episodes[i]

    config = {
        "episodes_per_phase": episodes_per_phase,
        "batch_size": args.batch_size,
        "entropy_coef": args.entropy,
        "reward_scale": 1.0,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "checkpoint_dir": args.checkpoint_dir,
    }

    # ── Resume from checkpoint ───────────────────────────
    start_phase = args.phases[0]
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume, policy, optimizer, device)
        start_phase = checkpoint["phase"]

    # ── Train selected phases ────────────────────────────
    print(f"Phases:  {args.phases}")
    print(f"Episodes: {episodes_per_phase}")
    print(f"Entropy:  {args.entropy}")
    print(f"Batch:    {args.batch_size} cases/graph")
    print()

    all_results = {}

    for phase in args.phases:
        if phase < start_phase:
            print(f"Skipping Phase {phase} (already completed, "
                  f"resuming from Phase {start_phase})")
            continue

        num_episodes = episodes_per_phase.get(phase, 500)

        # If resuming from a checkpoint in this phase, pick up where we left off
        resume_episode = 0
        resume_rewards = None
        if checkpoint and checkpoint.get("phase") == phase:
            resume_episode = checkpoint.get("episode", 0)
            resume_rewards = checkpoint.get("rewards", None)
            print(f"  Resuming Phase {phase} from episode {resume_episode}")

        rewards, sizes, best = train_phase(
            policy=policy,
            optimizer=optimizer,
            generator=generator,
            bridge=bridge,
            phase=phase,
            num_episodes=num_episodes,
            start_episode=resume_episode,
            initial_rewards=resume_rewards,
            batch_size=config["batch_size"],
            entropy_coef=config["entropy_coef"],
            reward_scale=config["reward_scale"],
            log_every=config["log_every"],
            save_every=config["save_every"],
            checkpoint_dir=config["checkpoint_dir"],
            device=device,
        )

        all_results[phase] = {
            "rewards": rewards,
            "sizes": sizes,
            "best": best,
        }

    # ── Save final model ─────────────────────────────────
    if config["checkpoint_dir"]:
        final_path = os.path.join(
            config["checkpoint_dir"], "final.pt"
        )
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "results": {
                str(k): {
                    "best": v["best"],
                    "avg_last_50": (
                        sum(v["rewards"][-50:]) / min(len(v["rewards"][-50:]), 50)
                        if v["rewards"] else 0.0
                    ),
                }
                for k, v in all_results.items()
            },
            "timestamp": datetime.now().isoformat(),
        }, final_path)
        print(f"\nFinal model saved to {final_path}")

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'=' * 66}")
    print("  CURRICULUM TRAINING COMPLETE")
    print(f"{'=' * 66}")
    for phase, result in all_results.items():
        last_50 = result["rewards"][-50:] if len(result["rewards"]) >= 50 else result["rewards"]
        avg = sum(last_50) / len(last_50) if last_50 else 0.0
        sizes = result["sizes"]
        print(f"  Phase {phase}: best={result['best']:.1f}, "
              f"avg-50={avg:.1f}, "
              f"sizes={min(sizes)}-{max(sizes)} ops")


if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════╗")
    print("║  NN Compiler v0: Curriculum Training Loop    ║")
    print("╚═══════════════════════════════════════════════╝")
    main()
