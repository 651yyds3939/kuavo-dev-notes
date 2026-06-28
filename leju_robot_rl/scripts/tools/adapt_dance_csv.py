#!/usr/bin/env python3
"""Adapt full-body dance CSV to Kuavo S46 default standing while preserving choreography."""

from __future__ import annotations

import argparse
import os

import numpy as np

# Kuavo S46 training / deploy default standing (leg_l1..l6, leg_r1..r6)
DEFAULT_LEG_POSE = np.array(
    [
        0.0,
        0.0,
        -0.27,
        0.52,
        -0.3,
        0.0,
        0.0,
        0.0,
        -0.27,
        0.52,
        -0.3,
        0.0,
    ],
    dtype=np.float64,
)

# Conservative in-place profile (previous behavior)
LEG_DELTA_SCALE_INPLACE = np.array(
    [
        0.75,
        0.55,
        0.45,
        0.60,
        0.55,
        0.70,
        0.75,
        0.55,
        0.45,
        0.60,
        0.55,
        0.70,
    ],
    dtype=np.float64,
)

# Per-joint scale on in-phase / anti-phase leg components (6-DoF per side)
LEG_IN_PHASE_SCALE = np.array([0.12, 0.12, 0.28, 0.38, 0.35, 0.30], dtype=np.float64)
LEG_ANTI_PHASE_SCALE = np.array([0.06, 0.06, 0.70, 0.92, 0.85, 0.65], dtype=np.float64)

ARM_DELTA_BOOST = np.array(
    [
        1.05,
        1.10,
        1.10,
        1.08,
        1.00,
        1.00,
        1.00,
        1.05,
        1.00,
        1.10,
        1.10,
        1.08,
        1.00,
        1.00,
    ],
    dtype=np.float64,
)

# URDF-inspired clamp (radians), index aligned with DEFAULT_LEG_POSE
LEG_LIMITS = np.array(
    [
        [-0.50, 0.50],
        [-0.45, 0.45],
        [-1.30, 0.55],
        [-0.10, 1.50],
        [-0.80, 0.45],
        [-0.45, 0.45],
        [-0.50, 0.50],
        [-0.45, 0.45],
        [-1.30, 0.55],
        [-0.10, 1.50],
        [-0.80, 0.45],
        [-0.45, 0.45],
    ],
    dtype=np.float64,
)

ARM_LIMITS = np.array(
    [
        [-1.8, 1.8],
        [0.05, 1.5],
        [-1.5, 1.5],
        [-1.8, 0.0],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-1.8, 1.8],
        [-1.5, -0.05],
        [-1.5, 1.5],
        [-1.8, 0.0],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
    ],
    dtype=np.float64,
)


def _load_joint_rows(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("time,"):
                continue
            vals = [float(x) for x in line.split(",") if x]
            if len(vals) < 26:
                continue
            if len(vals) >= 27:
                vals = vals[1:27]
            else:
                vals = vals[:26]
            rows.append(vals)
    if not rows:
        raise ValueError(f"No valid joint rows found in {path}")
    return np.asarray(rows, dtype=np.float64)


def _smooth_legs(legs: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return legs
    pad = window // 2
    padded = np.pad(legs, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(legs)
    for j in range(legs.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return out


def _remove_leg_drift(legs: np.ndarray, factor: float = 0.85) -> np.ndarray:
    """Remove slow forward-bias trend on hip/knee pitch channels."""
    out = legs.copy()
    drift_cols = [2, 3, 8, 9]
    frame_idx = np.arange(legs.shape[0], dtype=np.float64)
    for col in drift_cols:
        coeff = np.polyfit(frame_idx, legs[:, col], deg=1)
        trend = np.polyval(coeff, frame_idx)
        out[:, col] = legs[:, col] - factor * (trend - trend[0])
    return out


def _decompose_inplace_leg_delta(leg_delta: np.ndarray) -> np.ndarray:
    """Split leg motion into symmetric bounce + alternating weight-shift (in-place dance)."""
    delta_l = leg_delta[:, :6]
    delta_r = leg_delta[:, 6:12]
    in_phase = 0.5 * (delta_l + delta_r)
    anti_phase = 0.5 * (delta_l - delta_r)
    new_l = in_phase * LEG_IN_PHASE_SCALE[None, :] + anti_phase * LEG_ANTI_PHASE_SCALE[None, :]
    new_r = in_phase * LEG_IN_PHASE_SCALE[None, :] - anti_phase * LEG_ANTI_PHASE_SCALE[None, :]
    return np.hstack([new_l, new_r])


def _highpass_leg_delta(legs: np.ndarray, window: int = 31) -> np.ndarray:
    """Keep rhythmic leg motion while removing slow COM-drifting trends."""
    centered = legs - DEFAULT_LEG_POSE[None, :]
    if window <= 1:
        return centered
    pad = window // 2
    padded = np.pad(centered, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    low = np.empty_like(centered)
    for j in range(centered.shape[1]):
        low[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return centered - low


def _symmetrize_leg_delta(leg_delta: np.ndarray) -> np.ndarray:
    """Mirror left/right hip motion to avoid permanent staggered stance."""
    out = leg_delta.copy()
    pairs = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
    for left, right in pairs:
        avg = 0.5 * (out[:, left] + out[:, right])
        out[:, left] = avg
        out[:, right] = avg
    return out


def _clamp_legs(legs: np.ndarray) -> np.ndarray:
    out = legs.copy()
    for j in range(12):
        out[:, j] = np.clip(out[:, j], LEG_LIMITS[j, 0], LEG_LIMITS[j, 1])
    return out


def _clamp_arms(arms: np.ndarray) -> np.ndarray:
    out = arms.copy()
    for j in range(14):
        out[:, j] = np.clip(out[:, j], ARM_LIMITS[j, 0], ARM_LIMITS[j, 1])
    return out


def _loop_closure(traj: np.ndarray, blend_len: int = 20) -> np.ndarray:
    adapted = traj.copy()
    blend_len = min(blend_len, adapted.shape[0] // 10)
    start_pose = adapted[0].copy()
    for i in range(blend_len):
        alpha = (i + 1) / blend_len
        idx = adapted.shape[0] - blend_len + i
        adapted[idx] = (1.0 - alpha) * adapted[idx] + alpha * start_pose
    return adapted


def adapt_dance_trajectory(raw: np.ndarray, profile: str = "inplace") -> np.ndarray:
    legs = raw[:, :12].copy()
    arms = raw[:, 12:26].copy()

    if profile == "arms_only":
        legs = np.tile(DEFAULT_LEG_POSE, (raw.shape[0], 1))
        arm_ref = arms[0].copy()
        arm_delta = (arms - arm_ref) * 1.12
        arms = arm_ref + arm_delta
        arms = _clamp_arms(arms)
        return _loop_closure(np.hstack([legs, arms]))

    if profile == "fullbody_inplace":
        legs = _remove_leg_drift(legs, factor=0.92)
        leg_delta = _highpass_leg_delta(legs, window=31)
        leg_delta = _decompose_inplace_leg_delta(leg_delta)
        legs = DEFAULT_LEG_POSE[None, :] + leg_delta
        legs = _smooth_legs(legs, window=3)
        legs = _clamp_legs(legs)

        arm_ref = arms[0].copy()
        arm_delta = (arms - arm_ref) * ARM_DELTA_BOOST[None, :]
        arms = arm_ref + arm_delta
        arms = _clamp_arms(arms)
        return _loop_closure(np.hstack([legs, arms]))

    if profile == "expressive":
        legs = _remove_leg_drift(legs, factor=0.55)
        leg_delta = (legs - legs[0]) * LEG_DELTA_SCALE_INPLACE[None, :]
        leg_delta = _symmetrize_leg_delta(leg_delta)
        legs = DEFAULT_LEG_POSE[None, :] + leg_delta
        legs = _smooth_legs(legs, window=3)
        legs = _clamp_legs(legs)

        arm_ref = arms[0].copy()
        arm_delta = (arms - arm_ref) * ARM_DELTA_BOOST[None, :]
        arms = arm_ref + arm_delta
        arms = _clamp_arms(arms)
        return _loop_closure(np.hstack([legs, arms]))

    # inplace (legacy)
    legs = _remove_leg_drift(legs, factor=0.85)
    leg_ref = legs[0]
    leg_delta = (legs - leg_ref) * LEG_DELTA_SCALE_INPLACE[None, :]
    legs = DEFAULT_LEG_POSE[None, :] + leg_delta
    legs = _smooth_legs(legs, window=5)
    legs = _clamp_legs(legs)
    arms = _clamp_arms(arms)
    return _loop_closure(np.hstack([legs, arms]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Kuavo dance CSV for in-place full-body tracking.")
    parser.add_argument(
        "--input",
        default="kuavo_action_PERFECT_LIMIT_RAD.csv",
        help="Source dance CSV (26 or 27 columns per row).",
    )
    parser.add_argument(
        "--output",
        default="kuavo_action_ADAPTED_RAD.csv",
        help="Adapted CSV output (exactly 26 joint columns).",
    )
    parser.add_argument(
        "--profile",
        choices=("inplace", "expressive", "arms_only", "fullbody_inplace"),
        default="inplace",
        help="fullbody_inplace=alternating leg rhythm + boosted arms; arms_only=fixed legs.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    in_path = args.input if os.path.isabs(args.input) else os.path.join(repo_root, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(repo_root, args.output)

    raw = _load_joint_rows(in_path)
    adapted = adapt_dance_trajectory(raw, profile=args.profile)

    np.savetxt(out_path, adapted, delimiter=",", fmt="%.6f")
    print(f"Profile: {args.profile}")
    print(f"Input : {in_path} ({raw.shape[0]} frames x {raw.shape[1]} joints)")
    print(f"Output: {out_path} ({adapted.shape[0]} frames x {adapted.shape[1]} joints)")
    print(f"Frame0 leg vs default max abs diff: {np.max(np.abs(adapted[0, :12] - DEFAULT_LEG_POSE)):.4f} rad")
    print("Arm joint ranges (max-min):")
    for j in range(12, 26):
        print(f"  j{j:2d}: {adapted[:, j].max() - adapted[:, j].min():.3f} rad")
    print("Leg joint ranges (max-min):")
    for j in range(12):
        print(f"  j{j:2d}: {adapted[:, j].max() - adapted[:, j].min():.3f} rad")


if __name__ == "__main__":
    main()
