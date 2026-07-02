#!/usr/bin/env python3
"""Merge LAFAN1 kuavo_dance legs with native S54 Kuavo arms (proven arm semantics)."""

from __future__ import annotations

import argparse
import os

import numpy as np


def _load_26(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
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
        raise ValueError(f"No 26-DoF rows in {path}")
    return np.asarray(rows, dtype=np.float64)


def merge(
    legs_path: str,
    arms_path: str,
    output_path: str,
    arm_blend: float = 1.0,
) -> None:
    legs = _load_26(legs_path)
    arms_src = _load_26(arms_path)
    n = legs.shape[0]
    src_idx = np.linspace(0, arms_src.shape[0] - 1, n).astype(int)
    out = legs.copy()
    arm_delta = arms_src[src_idx, 12:26] - arms_src[0, 12:26]
    out[:, 12:26] = legs[0, 12:26] + arm_blend * arm_delta
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savetxt(output_path, out, delimiter=",", fmt="%.6f")
    print(f"Legs : {legs_path} ({legs.shape[0]} frames)")
    print(f"Arms : {arms_path} ({arms_src.shape[0]} frames, resampled)")
    print(f"Output: {output_path}")
    print(f"Leg amp mean: {np.mean([out[:, i].max() - out[:, i].min() for i in range(12)]):.3f} rad")
    print(f"Arm amp mean: {np.mean([out[:, i].max() - out[:, i].min() for i in range(12, 26)]):.3f} rad")


def main() -> None:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legs",
        default=os.path.join(repo, "kuavo_action_LAFAN1_g1_dance1_DANCE_RAD.csv"),
    )
    parser.add_argument(
        "--arms",
        default=os.path.join(repo, "kuavo_action_S49_FROM_S54_INPLACE_RAD.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(repo, "kuavo_action_HYBRID_LAFAN1LEGS_S54ARMS_RAD.csv"),
    )
    parser.add_argument("--arm-blend", type=float, default=1.0)
    args = parser.parse_args()
    merge(args.legs, args.arms, args.output, arm_blend=args.arm_blend)


if __name__ == "__main__":
    main()
