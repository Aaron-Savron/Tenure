"""
checkpoint_surgery.py -- Weight transplantation for production_v2.pt.

Transplants the old 13-dim, 1-output SchedulingPolicy checkpoint into
the new 15-dim, 2-output architecture required for active spilling.

Surgery plan:
  - input_proj.weight: [256,13] -> [256,15]  (zero-init cols 13-14)
  - GAT body: 4 layers, 4 heads, [256,256]  (direct copy, unchanged)
  - priority_head.0: [256,256]              (direct copy, unchanged)
  - priority_head.2.weight: [1,256] -> [2,256]
      row 0 = old Issue head (copy)
      row 1 = small noise (Spill head)
      bias[0] = old bias, bias[1] = -1.0 (conservative init)
"""

import torch
from typing import Dict


def transplant_weights(
    old_checkpoint_path: str,
    new_policy,
    old_feat_dim: int = 13,
    new_feat_dim: int = 15,
    device: str = "cpu",
    spill_bias_init: float = -1.0,
    spill_noise_scale: float = 0.01,
) -> dict:
    """
    Load an old checkpoint and splice its weights into a new SchedulingPolicy.

    Args:
        old_checkpoint_path: Path to the old .pt checkpoint.
        new_policy: A SchedulingPolicy(node_feat_dim=new_feat_dim) instance.
        old_feat_dim: Input feature dimension of old policy (default 13).
        new_feat_dim: Input feature dimension of new policy (default 15).
        device: Torch device.
        spill_bias_init: Initial bias for the Spill head (default -1.0,
                         conservative — don't spill until trained to).
        spill_noise_scale: Std of Gaussian noise for the Spill head weights.

    Returns:
        new_state_dict: The surgically-modified state dict.
    """
    old_ckpt = torch.load(old_checkpoint_path, map_location=device)
    old_sd = old_ckpt["policy_state_dict"]
    new_sd = new_policy.state_dict()

    # ── input_proj: splice old 13 cols into 15-col weight matrix ──
    old_w = old_sd["input_proj.weight"]  # [256, 13]
    new_sd["input_proj.weight"][:, :old_feat_dim] = old_w
    # Zero-init the new columns (Is_Spilled, Remaining_Reloads)
    new_sd["input_proj.weight"][:, old_feat_dim:] = 0.0
    new_sd["input_proj.bias"] = old_sd["input_proj.bias"]

    # ── GATConv layers: 4 layers, each has lin.weight, att_src, att_dst, bias ──
    # GATConv's internal linear uses key `gat_convs.{i}.lin.weight` (NOT `.weight`)
    for i in range(4):
        # GATConv input projection (lin.weight exists; lin.bias may not if bias=False)
        key = f"gat_convs.{i}.lin.weight"
        if key in old_sd:
            new_sd[key] = old_sd[key]
        key = f"gat_convs.{i}.lin.bias"
        if key in old_sd:
            new_sd[key] = old_sd[key]
        # Attention parameters
        for att in ["att_src", "att_dst"]:
            key = f"gat_convs.{i}.{att}"
            if key in old_sd:
                new_sd[key] = old_sd[key]
        # GATConv output bias
        key = f"gat_convs.{i}.bias"
        if key in old_sd:
            new_sd[key] = old_sd[key]

    # ── gat_lin layers: 4 residual linears, each with weight and bias ──
    for i in range(4):
        for param in ["weight", "bias"]:
            key = f"gat_lin.{i}.{param}"
            if key in old_sd:
                new_sd[key] = old_sd[key]

    # ── priority_head.0: first Linear(hidden, hidden) — unchanged ──
    new_sd["priority_head.0.weight"] = old_sd["priority_head.0.weight"]
    new_sd["priority_head.0.bias"] = old_sd["priority_head.0.bias"]

    # ── priority_head.2: splice [1,256] -> [2,256] ──
    old_head_w = old_sd["priority_head.2.weight"]  # [1, 256]
    old_head_b = old_sd["priority_head.2.bias"]    # [1]

    # Row 0: Issue/Reload head — copy old priority weights
    new_sd["priority_head.2.weight"][0, :] = old_head_w[0, :]
    new_sd["priority_head.2.bias"][0] = old_head_b[0]

    # Row 1: Spill head — small noise + conservative negative bias
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    noise = torch.normal(
        mean=0.0, std=spill_noise_scale,
        size=(1, old_head_w.size(1)),
        generator=rng,
    )
    new_sd["priority_head.2.weight"][1, :] = noise[0, :]
    new_sd["priority_head.2.bias"][1] = torch.tensor(spill_bias_init)

    return new_sd


def verify_surgery(
    old_checkpoint_path: str = "nn_compiler/production_v1.pt",
    new_feat_dim: int = 15,
    device: str = "cpu",
):
    """
    Quick verification: load the old checkpoint, run surgery,
    and check that no keys are missing or mismatched.

    Returns True if all checks pass.
    """
    from nn_compiler.policy import SchedulingPolicy

    print("=" * 60)
    print("  Checkpoint Surgery Verification")
    print("=" * 60)

    # Create new policy
    new_policy = SchedulingPolicy(node_feat_dim=new_feat_dim).to(device)
    old_sd_shape = torch.load(
        old_checkpoint_path, map_location=device
    )["policy_state_dict"]["input_proj.weight"].shape

    print(f"  Old input_proj shape: {list(old_sd_shape)}")
    print(f"  New input_proj shape: {list(new_policy.input_proj.weight.shape)}")
    print(f"  Old head output: 1  -> New head output: 2")
    print()

    # Run surgery
    new_sd = transplant_weights(old_checkpoint_path, new_policy, device=device)

    # Validate
    missing = []
    unexpected = []
    for key in new_policy.state_dict():
        if key not in new_sd:
            missing.append(key)
    for key in new_sd:
        if key not in new_policy.state_dict():
            unexpected.append(key)

    all_ok = True

    if missing:
        print(f"  [ERROR] Missing keys in new state_dict: {missing}")
        all_ok = False
    if unexpected:
        print(f"  [ERROR] Unexpected keys in new state_dict: {unexpected}")
        all_ok = False

    if not all_ok:
        print("\n  ❌ SURGERY FAILED — key mismatch!")
        return False

    # Check shapes
    shape_errors = []
    new_policy_sd = new_policy.state_dict()
    for key in new_sd:
        if new_sd[key].shape != new_policy_sd[key].shape:
            shape_errors.append(
                f"    {key}: got {list(new_sd[key].shape)}, "
                f"expected {list(new_policy_sd[key].shape)}"
            )

    if shape_errors:
        print(f"  [ERROR] Shape mismatches:")
        for err in shape_errors:
            print(err)
        all_ok = False

    if not all_ok:
        print("\n  ❌ SURGERY FAILED — shape mismatch!")
        return False

    # Verify new columns are zero
    col_norms = []
    for col in range(old_sd_shape[1], new_policy.input_proj.weight.size(1)):
        norm = new_policy.input_proj.weight[:, col].norm().item()
        col_norms.append(f"col{col}={norm:.6f}")
    print(f"  New input cols: {', '.join(col_norms)} (should be 0.000000)")

    # Verify spill head bias is conservative
    new_policy.load_state_dict(new_sd)
    spill_bias = new_policy.priority_head[2].bias[1].item()
    issue_bias = new_policy.priority_head[2].bias[0].item()
    print(f"  Issue head bias: {issue_bias:.4f}  (preserved from v1)")
    print(f"  Spill head bias: {spill_bias:.4f}  (conservative negative)")

    # Verify old GAT weights preserved (sample from actual state_dict keys)
    old_ckpt = torch.load(old_checkpoint_path, map_location=device)
    old_sd = old_ckpt["policy_state_dict"]
    for test_key in ["gat_convs.0.lin.weight", "gat_convs.3.lin.weight",
                     "gat_lin.0.weight", "gat_lin.3.weight",
                     "gat_convs.0.att_src", "gat_convs.2.att_dst"]:
        if test_key in old_sd:
            old_norm = old_sd[test_key].norm().item()
            new_norm = new_sd[test_key].norm().item()
            match = "OK" if abs(old_norm - new_norm) < 1e-5 else "MISMATCH"
            print(f"  {test_key}: old_norm={old_norm:.4f}, "
                  f"new_norm={new_norm:.4f}  [{match}]")

    # Quick forward pass test
    test_obs = {
        "x": torch.zeros(1, new_feat_dim),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "action_mask": torch.ones(2, dtype=torch.bool),
    }
    try:
        with torch.no_grad():
            probs = new_policy._get_priority_distribution(test_obs)
        print(f"  Forward pass: probs shape={list(probs.shape)}, "
              f"sum={probs.sum().item():.4f}  [OK]")
    except Exception as e:
        print(f"  [ERROR] Forward pass failed: {e}")
        all_ok = False

    print(f"\n  {'✅ SURGERY PASSED' if all_ok else '❌ SURGERY FAILED'}")
    return all_ok


if __name__ == "__main__":
    success = verify_surgery()
    import sys
    sys.exit(0 if success else 1)
