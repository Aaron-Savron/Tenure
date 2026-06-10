"""
nn_compiler -- RL environment + GNN policy + REINFORCE training
for the dataflow virtual CPU compiler.

Architecture:
  ComputeGraph -> DataflowGymEnv -> VMBridge -> Rust VM (reward)
  GNNPolicy (GAT convs) <- encode_observation(env)
  train() loops: collect_episode() -> REINFORCE update
"""

from .compiler_env import (
    ComputeGraph,
    ComputeNode,
    NodeType,
    RoutingType,
    VMBridge,
    DataflowGymEnv,
    SchedulingGymEnv,
    create_dot_product_graph,
    create_matmul_2x2_graph,
    create_relu_graph,
)

from .policy import (
    GNNPolicy,
    SchedulingPolicy,
    encode_observation,
    encode_scheduling_obs,
    decode_action,
)

from .graph_generator import (
    ProceduralGraphGenerator,
    evaluate_graph,
    PHASE_CONFIG,
)

from .scheduler_baseline import (
    critical_path_height,
    schedule_cpf,
    schedule_topological,
)

from .train import (
    train,
    collect_episode,
    train_episode,
    collect_scheduling_episode,
    train_scheduling_episode,
    collect_scheduling_episode_ppo,
    train_scheduling_episode_ppo,
    ppo_schedule_train,
    RunningBaseline,
)

from .curriculum_train import (
    curriculum_train,
    train_phase,
    progressive_schedule_train,
    PROGRESSIVE_DEFAULTS,
)
