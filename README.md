<p align="center">
  <h1 align="center">Tenure</h1>
  <p align="center"><em>A GAT compiler that learned when to push and when to wait.</em></p>
</p>

<p align="center">
  <a href="ARCHITECTURE.md"><img src="https://img.shields.io/badge/architecture-deepdive-8A2BE2" alt="Architecture"></a>
  <a href="RESULTS.md"><img src="https://img.shields.io/badge/results-experimental-00BFFF" alt="Results"></a>
  <a href="BREAKTHROUGHS.md"><img src="https://img.shields.io/badge/breakthroughs-conceptual-FF6B6B" alt="Breakthroughs"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

A graph attention network that learned to schedule computer instructions better than the textbook algorithm. It does this by watching how crowded the processor's temporary storage (registers) is, and deciding when to push urgent work and when to hold back.

## What it does

Every compiler has to decide what order to run instructions in. The standard approach is simple: always work on the most urgent thing first. This is called Critical Path First (CPF). It works great most of the time because finishing urgent work quickly frees up resources.

But there is a catch. Processors have a limited number of registers (small, fast storage slots for in-progress values). If you run too much urgent work at once, you run out of registers and the whole pipeline stalls. This is the fundamental tension this project explores.

The Tenure model learned a smarter strategy:
- When registers are plentiful, work on the most urgent thing at full speed.
- When registers are tight, pause the urgent work and pick something that frees up a register first.

This behavior was not programmed. It emerged from training a 4-layer GAT on a carefully designed reward function that taxes every clock cycle spent over the register budget, not just the peak violation.

## Quick start

Requires Python 3.10+, PyTorch, and PyTorch Geometric. The Rust VM is optional if you only want to run the scheduling environment.

```bash
pip install torch torch-geometric numpy
```

Run the OOD benchmark on the trained model:

```bash
python -m nn_compiler.ood_eval nn_compiler/production_v1.pt
```

To train a new model from scratch:

```bash
python -m nn_compiler.curriculum_train --phases 4
```

## Papers

| Document | What it covers |
|----------|---------------|
| [Architecture](ARCHITECTURE.md) | Full system design: graph generation, simulator, GAT policy, training pipeline, OOD benchmark |
| [Results](RESULTS.md) | Complete experimental trajectory across all 6 training regimes and OOD generalization data |
| [Breakthroughs](BREAKTHROUGHS.md) | The 4 key conceptual advances: peak penalty limits, integral penalty, rational agent behavior, distributed register management |

## Key idea

The core insight is that punishing peak register usage does not work. A model can sprint past a peak violation, pay a flat fine, and come out ahead. But punishing every cycle spent over budget changes the math completely. Suddenly every extra cycle in a high pressure zone costs real reward, and the model is forced to manage register tenure, not just peak magnitude.

This is called an integral over time penalty, and it is the single change that made everything else possible.

## Project structure

```
nn_compiler/
  compiler_env.py       Core environment: graph representation, VM bridge, scheduling simulator
  graph_generator.py    Generates random computation graphs for training
  scheduler_baseline.py CPF heuristic used as the baseline target
  policy.py             GNN and GAT policy networks
  curriculum_train.py   Training loops for both routing and scheduling
  dl_subgraphs.py       Hand-built DL subgraphs for OOD testing
  ood_eval.py           Benchmark module for out-of-distribution evaluation
  production_v1.pt      Trained model checkpoint
```

## The model

A 4 layer, 4 head Graph Attention Network with 256 hidden units. It reads 13 features per node: opcode type, ready status, execution state, node height, downstream latency, critical path distance, and register pressure. The register pressure feature is the key new signal that was not present in the original architecture.

The final checkpoint (production_v1.pt) has a column 13 weight norm of 0.611, meaning it learned to use the register pressure signal precisely and efficiently rather than amplifying it into a noisy shout.

## License

MIT
