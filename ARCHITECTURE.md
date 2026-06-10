# Architecture Overview

This document describes how Tenure works end to end. The system has four main parts: a graph generator, a cycle accurate simulator, a GAT policy network, and a training loop. Each part is described below.

## Graph generation

The system generates random computation graphs for training. These are directed acyclic graphs where each node is an operation (Add, Mul, Switch, Merge, CmpGeZ) and each edge is a data dependency. There are four phases of increasing complexity:

Phase 1: Linear chains of 3 to 4 Add and Mul nodes. This is the simplest possible structure and teaches the model basic dependency tracking.

Phase 2: Branching topologies with 4 to 6 nodes and fan out. Nodes can have multiple consumers, creating register pressure as one value must stay alive for several downstream operations.

Phase 3: Conditional graphs with 6 to 8 nodes including Switch, Merge, and CmpGeZ. These model if else control flow where a value is routed down one of two paths.

Phase 4: Parallel lane topologies with 3 to 5 independent lanes of 2 to 4 operations each, converging through a reduction tree. This is the primary training distribution for the scheduling policy. The lanes create competition for registers because multiple independent values are live at the same time.

## The scheduling simulator

The simulator models a K issue processor where up to K instructions can start executing each clock cycle. Each instruction takes a fixed latency (1 cycle for Add, 3 cycles for Mul). Instructions complete in background and their dependents become ready only after all parent instructions have finished.

The simulator tracks register pressure through consumer counts. Each value produced by an instruction occupies a register until its last consumer has started executing. When the number of live registers exceeds the hardware budget, the simulator checks whether the next instruction would free a register. If it would not, the pipeline stalls until a register becomes available.

The reward function combines three terms:
- Cycle count (negative, larger is worse)
- A small queue penalty (0.1 times the maximum issue queue depth)
- An integral register penalty

The integral penalty is the key innovation. It is computed as:

    alpha * sum over all cycles of max(0, live_registers - max_registers)

This means every cycle spent over the register budget costs alpha. Earlier versions used a peak penalty that only looked at the maximum number of live registers, which meant the model could sprint into a high pressure zone, pay a flat fine, and come out ahead. The integral penalty closes that loophole by taxing duration, not magnitude.

## The GAT policy

The policy is a 4 layer, 4 head Graph Attention Network with 256 hidden units. It takes 13 features per node:

- Features 0 to 6: One hot opcode type (Input, Mul, Add, Switch, CmpGeZ, Output, Merge)
- Feature 7: Whether the node is ready to fire (all dependencies satisfied)
- Feature 8: Whether the node has been executed
- Feature 9: Normalized node height (longest path to output)
- Feature 10: Normalized downstream latency sum
- Feature 11: Normalized critical path distance (CPD)
- Feature 12: Register pressure (outstanding values divided by max_registers, capped at 1.0)

The action head outputs a priority score for each node. A masked softmax restricts choices to the ready nodes. During evaluation, the model picks the highest priority ready node greedily.

## Training

Training uses PPO with a per graph EMA baseline, entropy bonus, and clip epsilon of 0.2. The progressive curriculum generates 50 candidate graphs, sorts them by CPF cycle count, and maintains an active pool of 5 graphs plus a waiting list. When the policy achieves 90 percent CPF mastery over the last 50 episodes on a graph, that graph graduates and the next hardest graph is promoted.

The column 13 gradient is amplified by 5x to help the model learn the register pressure feature faster. This was necessary because the pressure signal is sparse early in training when the model does not yet know how to create pressure in its schedules.

## The OOD benchmark

The out of distribution benchmark tests the trained model on three families of hand built subgraphs that mimic real deep learning workloads:

- MLP blocks: Deep linear chains of Multiply operations. These test the model's ability to sprint through simple sequential work with no register competition.

- Residual Add: Fork join diamonds where a produce value must stay alive across multiple parallel lanes. These test tenure dilation, the model's ability to manage long lived values under register pressure.

- LayerNorm: Wide fan in reduction trees where many inputs converge to a single output. These test the model's behavior under high pressure spikes that collapse quickly.

Each graph is evaluated with both the model and the CPF baseline, and the benchmark reports cycles, peak register pressure, spill penalty, and Max-CPD pick rate bucketed by register pressure level.
