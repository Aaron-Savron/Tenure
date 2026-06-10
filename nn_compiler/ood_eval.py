"""
ood_eval.py -- OOD Generalization Benchmark Suite for the GAT Scheduler.

Consolidates three DL subgraph families (MLP Block, Residual Add, LayerNorm)
into a reusable evaluation harness. Loads a trained SchedulingPolicy checkpoint
and prints a structured evaluation matrix comparing model vs CPF behavior across
register pressure buckets.

Usage:
    from nn_compiler.ood_eval import OODBenchmark
    benchmark = OODBenchmark(alpha=1.0, max_registers=3, k=2)
    benchmark.evaluate_checkpoint("checkpoints/integral_curriculum/integral_final.pt")

Or from CLI:
    python -m nn_compiler.ood_eval [checkpoint_path]
"""
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from nn_compiler.compiler_env import SchedulingGymEnv, DEFAULT_LATENCY
from nn_compiler.policy import SchedulingPolicy, encode_scheduling_obs
from nn_compiler.scheduler_baseline import schedule_cpf
from nn_compiler.dl_subgraphs import (
    create_mlp_block,
    create_residual_add,
    create_layer_norm,
)


# Default test suite -- 9 DL subgraph variants across 3 families
DEFAULT_TEST_GRAPHS = [
    ("MLP Block (1 layer, W=3)",
     lambda: create_mlp_block(num_hidden=1, layer_width=3)),
    ("MLP Block (2 layers, W=2)",
     lambda: create_mlp_block(num_hidden=2, layer_width=2)),
    ("MLP Block (3 layers, W=2)",
     lambda: create_mlp_block(num_hidden=3, layer_width=2)),
    ("Residual Add (3 lanes, depth=2)",
     lambda: create_residual_add(num_lanes=3, lane_depth=2)),
    ("Residual Add (4 lanes, depth=2)",
     lambda: create_residual_add(num_lanes=4, lane_depth=2)),
    ("Residual Add (5 lanes, depth=1)",
     lambda: create_residual_add(num_lanes=5, lane_depth=1)),
    ("LayerNorm (4 channels)",
     lambda: create_layer_norm(num_channels=4)),
    ("LayerNorm (6 channels)",
     lambda: create_layer_norm(num_channels=6)),
    ("LayerNorm (8 channels)",
     lambda: create_layer_norm(num_channels=8)),
]


class OODBenchmark:
    """Reusable OOD generalization benchmark for GAT scheduling policies.

    Args:
        alpha: Integral penalty coefficient (register_penalty_alpha).
        max_registers: Physical register file size.
        k: Issue width for the cycle simulator.
        device: Torch device for inference.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_registers: int = 3,
        k: int = 2,
        chaos: bool = False,
        unit_limit: Optional[Dict[str, int]] = None,
        latency_distribution: Optional[Dict[str, Tuple[int, int]]] = None,
        device: str = "cpu",
    ):
        self.alpha = alpha
        self.max_registers = max_registers
        self.k = k
        self.chaos = chaos
        self.device = device

        # Chaos defaults: single Mul port, variable latencies
        if chaos and unit_limit is None:
            unit_limit = {"Mul": 1}
        if chaos and latency_distribution is None:
            latency_distribution = {
                "Mul": (2, 4),    # 2-4 cycles
                "Add": (1, 2),    # 1-2 cycles
                "Sub": (1, 2),    # 1-2 cycles
                "Div": (3, 6),    # 3-6 cycles
                "CmpGeZ": (1, 2), # 1-2 cycles
                "Switch": (1, 1), # deterministic
                "Merge": (1, 1),  # deterministic
            }
        self.unit_limit = unit_limit if unit_limit is not None else {}
        self.latency_distribution = latency_distribution if latency_distribution is not None else {}

    def build_mlp_block(self, num_hidden=2, layer_width=4):
        """Thin wrapper around create_mlp_block for discoverability."""
        return create_mlp_block(num_hidden=num_hidden, layer_width=layer_width)

    def build_residual_add(self, num_lanes=3, lane_depth=2):
        """Thin wrapper around create_residual_add for discoverability."""
        return create_residual_add(num_lanes=num_lanes, lane_depth=lane_depth)

    def build_layer_norm(self, num_channels=6):
        """Thin wrapper around create_layer_norm for discoverability."""
        return create_layer_norm(num_channels=num_channels)

    def _evaluate_model(self, graph, policy) -> Dict[str, Any]:
        """Evaluate the policy on a single graph. Returns full metrics dict."""
        # Build topo_order index mapping for action index conversion
        topo = graph.get_topological_order()

        env = SchedulingGymEnv(
            graph, None, [],
            max_exec_units=self.k, latency=DEFAULT_LATENCY,
            max_registers=self.max_registers,
            register_penalty_alpha=self.alpha,
            unit_limit=self.unit_limit,
            latency_distribution=self.latency_distribution,
        )
        env.reset()

        cpd_picks = 0
        total_steps = 0
        high_cpd = 0
        high_steps = 0
        mid_cpd = 0
        mid_steps = 0
        low_cpd = 0
        low_steps = 0

        while not env.done:
            obs = encode_scheduling_obs(env)
            with torch.no_grad():
                obs_dev = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in obs.items()
                }
                probs = policy._get_priority_distribution(obs_dev)
            action_idx = torch.argmax(probs).item()
            N = len(topo)

            # Map 2N action index to topo_order node for CPD tracking
            if action_idx < N:
                chosen_nid = topo[action_idx]
            else:
                chosen_nid = topo[action_idx - N]

            # CPD tracking uses n_ready_mask (N-length, indexed by node_names)
            node_names = obs["node_names"]
            n_ready = obs.get("n_ready_mask", obs.get("ready_mask"))
            # Get ready indices in node_names space
            ready_indices = [j for j, name in enumerate(node_names) if
                            j < len(n_ready) and n_ready[j]]

            if len(ready_indices) > 1 and action_idx < N:
                cpd_vals = []
                for j in ready_indices:
                    cpd_vals.append(obs["x"][j, 11].item())

                max_cpd = max(cpd_vals)
                chosen_cpd = obs["x"][node_names.index(chosen_nid), 11].item() if chosen_nid in node_names else 0
                is_max_cpd = (chosen_cpd >= max_cpd - 1e-6)
                reg_p = obs["x"][node_names.index(chosen_nid), 12].item() if chosen_nid in node_names else 0

                cpd_picks += 1 if is_max_cpd else 0
                total_steps += 1

                if reg_p >= 0.75:
                    high_cpd += 1 if is_max_cpd else 0
                    high_steps += 1
                elif reg_p >= 0.25:
                    mid_cpd += 1 if is_max_cpd else 0
                    mid_steps += 1
                else:
                    low_cpd += 1 if is_max_cpd else 0
                    low_steps += 1

            _, _, _, info = env.step(action_idx)

        cycles = info.get("cycles", 0)
        peak = info.get("max_live_registers", 0)
        spill = info.get("spill_penalty", 0)
        spill_steps = info.get("spill_steps", 0)
        struct_stalls = info.get("struct_stalls", 0)

        # CPF baseline (issues only — no spill/reload, uses topo indices)
        cpf_sched = schedule_cpf(graph)
        cpf_env = SchedulingGymEnv(
            graph, None, [],
            max_exec_units=self.k, latency=DEFAULT_LATENCY,
            max_registers=self.max_registers,
            register_penalty_alpha=self.alpha,
            unit_limit=self.unit_limit,
            latency_distribution=self.latency_distribution,
        )
        cpf_env.reset()
        for nid in cpf_sched:
            cpf_idx = topo.index(nid)
            _, _, _, cpf_info = cpf_env.step(cpf_idx)
        cpf_cycles = cpf_info.get("cycles", 0)
        cpf_peak = cpf_info.get("max_live_registers", 0)
        cpf_spill = cpf_info.get("spill_penalty", 0)
        cpf_spill_steps = cpf_info.get("spill_steps", 0)
        cpf_struct_stalls = cpf_info.get("struct_stalls", 0)

        def pct(p, t):
            return 100 * p / max(t, 1)

        return {
            "n_ops": len(topo),
            "model_cycles": cycles,
            "cpf_cycles": cpf_cycles,
            "cycle_diff": cycles - cpf_cycles,
            "model_peak": peak,
            "cpf_peak": cpf_peak,
            "model_spill": spill,
            "cpf_spill": cpf_spill,
            "model_spill_steps": spill_steps,
            "cpf_spill_steps": cpf_spill_steps,
            "struct_stalls": struct_stalls,
            "cpf_struct_stalls": cpf_struct_stalls,
            "overall_max_cpd": pct(cpd_picks, total_steps),
            "high_max_cpd": pct(high_cpd, high_steps) if high_steps > 0 else None,
            "mid_max_cpd": pct(mid_cpd, mid_steps) if mid_steps > 0 else None,
            "low_max_cpd": pct(low_cpd, low_steps) if low_steps > 0 else None,
            "high_steps": high_steps,
            "mid_steps": mid_steps,
            "low_steps": low_steps,
        }

    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        test_graphs: Optional[List[Tuple[str, callable]]] = None,
    ) -> List[Dict[str, Any]]:
        """Load a SchedulingPolicy checkpoint and evaluate on all test graphs.

        Args:
            checkpoint_path: Path to .pt checkpoint with 'policy_state_dict' key.
            test_graphs: Optional list of (label, graph_fn) pairs. Defaults to
                         all MLP / Residual Add / LayerNorm variants.

        Returns:
            List of per-graph result dicts.
        """
        if test_graphs is None:
            test_graphs = DEFAULT_TEST_GRAPHS

        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        policy = SchedulingPolicy(node_feat_dim=20).to(self.device)
        policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        policy.eval()

        # Print feature column norms (for 15-dim feature space)
        for col in range(15):
            w = policy.input_proj.weight.data[col]
            print(f"  Col {col} norm: {w.norm().item():.4f}")
        print(f"  Config: K={self.k}, max_registers={self.max_registers}, "
              f"alpha={self.alpha} (integral)")
        print()

        # Run evaluation
        results = []
        for label, g_fn in test_graphs:
            g = g_fn()
            n_ops = len(g.get_topological_order())
            r = self._evaluate_model(g, policy)
            r["label"] = label
            results.append(r)

            inv = ""
            if r["high_max_cpd"] is not None and r["high_max_cpd"] < 60:
                inv = " *** INVERSION ***"
            elif r["high_max_cpd"] is not None and r["high_max_cpd"] < 65:
                inv = " * partial *"

            print(f"  {label} ({n_ops} ops):")
            s = r.get('struct_stalls', 0)
            cs = r.get('cpf_struct_stalls', 0)
            stalls_str = f" | Stalls {s}(CPF={cs})" if s > 0 or cs > 0 else ""
            print(f"    Cycles {r['model_cycles']} (CPF {r['cpf_cycles']}, "
                  f"diff {r['cycle_diff']:+d}) | Peak {r['model_peak']} | "
                  f"Spill {r['model_spill']:.2f} ({r['model_spill_steps']} steps){stalls_str}")
            hi = f"{r['high_max_cpd']:.1f}%" if r['high_max_cpd'] is not None else "N/A"
            mi = f"{r['mid_max_cpd']:.1f}%" if r['mid_max_cpd'] is not None else "N/A"
            lo = f"{r['low_max_cpd']:.1f}%" if r['low_max_cpd'] is not None else "N/A"
            print(f"    Max-CPD: High={hi}({r['high_steps']}s) "
                  f"Mid={mi}({r['mid_steps']}s) Low={lo}({r['low_steps']}s){inv}")

        self._print_summary(results)
        return results

    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print the structured evaluation matrix with family convergence."""
        print()
        print("=" * 95)
        print("  OOD GENERALIZATION SUMMARY")
        print("=" * 95)
        hdr = (f"{'Graph':<34s} | {'Cyc':>4s} {'CPF':>4s} {'D':>4s} | "
               f"{'Pk':>3s} {'Spill':>6s} {'Stalls':>6s} | "
               f"{'High%':>6s} {'Mid%':>6s} {'Low%':>6s} | {'Inv?':>5s}")
        print(hdr)
        print("-" * 105)

        for r in results:
            inv = ("YES!" if r["high_max_cpd"] is not None and r["high_max_cpd"] < 60
                   else "part" if r["high_max_cpd"] is not None and r["high_max_cpd"] < 65
                   else "no")
            hi = f"{r['high_max_cpd']:.1f}" if r['high_max_cpd'] is not None else "N/A"
            mi = f"{r['mid_max_cpd']:.1f}" if r['mid_max_cpd'] is not None else "N/A"
            lo = f"{r['low_max_cpd']:.1f}" if r['low_max_cpd'] is not None else "N/A"
            short = r["label"][:33]
            stalls = r.get('struct_stalls', 0)
            print(f"{short:<34s} | {r['model_cycles']:>4d} {r['cpf_cycles']:>4d} "
                  f"{r['cycle_diff']:+4d} | {r['model_peak']:>3d} "
                  f"{r['model_spill']:>6.2f} {stalls:>6d} | "
                  f"{hi:>6s} {mi:>6s} {lo:>6s} | {inv:>5s}")

        # Family-level convergence
        print()
        print("  FAMILY CONVERGENCE ANALYSIS")
        print("-" * 50)
        for family_name in ["MLP", "Residual Add", "LayerNorm"]:
            family = [r for r in results if family_name in r["label"]]
            if not family:
                continue
            high_vals = [r["high_max_cpd"] for r in family
                         if r["high_max_cpd"] is not None]
            low_vals = [r["low_max_cpd"] for r in family
                        if r["low_max_cpd"] is not None]
            avg_high = np.mean(high_vals) if high_vals else 0
            avg_low = np.mean(low_vals) if low_vals else 0
            avg_diff = np.mean([r["cycle_diff"] for r in family])
            avg_spill = np.mean([r["model_spill"] for r in family])
            cpf_spill = np.mean([r["cpf_spill"] for r in family])

            inv_status = ""
            if avg_high < avg_low - 20 and avg_high > 0:
                inv_status = " *** INVERSION ***"
            elif avg_high < avg_low - 10 and avg_high > 0:
                inv_status = " * partial *"

            print(f"  {family_name}: avg_diff={avg_diff:+.1f}c, "
                  f"high={avg_high:.1f}%, low={avg_low:.1f}%, "
                  f"spill={avg_spill:.2f}(CPF={cpf_spill:.2f}){inv_status}")


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "checkpoints/integral_curriculum/integral_final.pt"

    benchmark = OODBenchmark(alpha=1.0, max_registers=3, k=2)
    benchmark.evaluate_checkpoint(ckpt)
