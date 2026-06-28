#!/usr/bin/env python3
"""Adapt Kuavo S54 (27-DoF, with waist) dance CSV to Gen-4 robots (S46/S49, 26-DoF).

S54 joint order (Leg×12 + waist_yaw + Arm×14):
  leg_l1..6, leg_r1..6, waist_yaw, zarm_l1..7, zarm_r1..7

Gen-4 order (Leg×12 + Arm×14, no waist):
  leg_l1..6, leg_r1..6, zarm_l1..7, zarm_r1..7
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Isaac / LejuLab S54 default standing (from KuavoS54ArticulationCfg)
S54_LEG_DEFAULT = np.array(
    [0.0, 0.0, -0.4, 0.69, -0.33, 0.0, 0.0, 0.0, -0.4, 0.69, -0.33, 0.0],
    dtype=np.float64,
)

# Kuavo Gen-4 deploy / training default (S46/S49)
GEN4_LEG_DEFAULT = np.array(
    [0.0, 0.0, -0.27, 0.52, -0.3, 0.0, 0.0, 0.0, -0.27, 0.52, -0.3, 0.0],
    dtype=np.float64,
)

# S49 URDF-inspired limits (radians), aligned with leg + arm indices
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

WAIST_TO_SHOULDER_SCALE = 0.35


def _load_s54_motion(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load S54 CSV: root_pos(3), root_quat(4), dof_pos(27)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("body_"):
                continue
            vals = [float(x) for x in line.split(",") if x]
            if len(vals) < 34:
                continue
            rows.append(vals[:34])
    if not rows:
        raise ValueError(f"No valid S54 rows (need >=34 cols) in {path}")
    data = np.asarray(rows, dtype=np.float64)
    root_pos = data[:, :3].copy()
    root_quat = data[:, 3:7].copy()
    joints = data[:, 7:34].copy()
    return root_pos, root_quat, joints


def _s54_to_gen4_joints(s54_joints: np.ndarray, waist_to_shoulder: float) -> np.ndarray:
    """Drop waist_yaw and fold part of its motion into shoulder pitch."""
    legs = s54_joints[:, :12]
    waist = s54_joints[:, 12]
    arms = s54_joints[:, 13:27].copy()

    # S49 arms mount on base_link; approximate torso yaw with symmetric shoulder pitch.
    arms[:, 0] += waist_to_shoulder * waist
    arms[:, 7] += waist_to_shoulder * waist

    legs = GEN4_LEG_DEFAULT[None, :] + (legs - S54_LEG_DEFAULT[None, :])
    return np.hstack([legs, arms])


def _center_root_xy(root_pos: np.ndarray) -> np.ndarray:
    out = root_pos.copy()
    out[:, :2] -= root_pos[0, :2]
    return out


def _downsample(traj: np.ndarray, src_fps: int, dst_fps: int) -> np.ndarray:
    if src_fps == dst_fps:
        return traj
    n_src = traj.shape[0]
    duration = (n_src - 1) / float(src_fps)
    n_dst = int(round(duration * dst_fps)) + 1
    src_t = np.linspace(0.0, duration, n_src)
    dst_t = np.linspace(0.0, duration, n_dst)
    out = np.empty((n_dst, traj.shape[1]), dtype=np.float64)
    for j in range(traj.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, traj[:, j])
    return out


def _smooth(traj: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return traj
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(traj)
    padded = np.pad(traj, ((pad, pad), (0, 0)), mode="edge")
    for j in range(traj.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return out


def _clamp_joints(joints: np.ndarray) -> np.ndarray:
    out = joints.copy()
    for j in range(12):
        out[:, j] = np.clip(out[:, j], LEG_LIMITS[j, 0], LEG_LIMITS[j, 1])
    for j in range(14):
        out[:, j + 12] = np.clip(out[:, j + 12], ARM_LIMITS[j, 0], ARM_LIMITS[j, 1])
    return out


def _loop_closure(traj: np.ndarray, blend_len: int = 25) -> np.ndarray:
    out = traj.copy()
    blend_len = min(blend_len, max(1, out.shape[0] // 10))
    start = out[0].copy()
    for i in range(blend_len):
        alpha = (i + 1) / blend_len
        idx = out.shape[0] - blend_len + i
        out[idx] = (1.0 - alpha) * out[idx] + alpha * start
    return out


def _remove_leg_drift(legs: np.ndarray, factor: float = 0.9) -> np.ndarray:
    out = legs.copy()
    frame_idx = np.arange(legs.shape[0], dtype=np.float64)
    for col in (2, 3, 8, 9):
        coeff = np.polyfit(frame_idx, legs[:, col], deg=1)
        trend = np.polyval(coeff, frame_idx)
        out[:, col] = legs[:, col] - factor * (trend - trend[0])
    return out


def adapt_s54_dance(
    input_path: str,
    output_path: str,
    src_fps: int = 120,
    dst_fps: int = 50,
    waist_to_shoulder: float = WAIST_TO_SHOULDER_SCALE,
    joint_only: bool = True,
) -> None:
    root_pos, root_quat, s54_joints = _load_s54_motion(input_path)
    root_pos = _center_root_xy(root_pos)

    gen4 = _s54_to_gen4_joints(s54_joints, waist_to_shoulder)
    legs = _remove_leg_drift(gen4[:, :12])
    gen4[:, :12] = legs
    gen4 = _smooth(gen4, window=3)
    gen4 = _clamp_joints(gen4)
    gen4 = _loop_closure(gen4)

    gen4_ds = _downsample(gen4, src_fps, dst_fps)
    root_pos_ds = _downsample(root_pos, src_fps, dst_fps)
    root_quat_ds = _downsample(root_quat, src_fps, dst_fps)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if joint_only:
        np.savetxt(output_path, gen4_ds, delimiter=",", fmt="%.6f")
    else:
        deploy = np.hstack([root_pos_ds, root_quat_ds, gen4_ds, np.zeros_like(gen4_ds)])
        np.savetxt(output_path, deploy, delimiter=",", fmt="%.6f")

    duration = (gen4_ds.shape[0] - 1) / float(dst_fps)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Frames: {gen4_ds.shape[0]} @ {dst_fps} Hz ({duration:.2f}s)")
    print(f"Leg range (max-min) per joint:")
    for j in range(12):
        print(f"  leg[{j:2d}]: {gen4_ds[:, j].max() - gen4_ds[:, j].min():.3f} rad")
    print(f"Arm range (max-min) per joint:")
    for j in range(12, 26):
        print(f"  arm[{j:2d}]: {gen4_ds[:, j].max() - gen4_ds[:, j].min():.3f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Kuavo S54 dance CSV to Gen-4 (S49/S46).")
    parser.add_argument(
        "--input",
        default="/home/lwy/kuavo_all/LejuLab-Train/source/leju_robot/leju_robot/assets/motion_data/mimic/csv_data/kuavos54_dance_120fps.csv",
    )
    parser.add_argument(
        "--output",
        default="kuavo_action_S49_FROM_S54_50FPS_RAD.csv",
    )
    parser.add_argument("--src-fps", type=int, default=120)
    parser.add_argument("--dst-fps", type=int, default=50)
    parser.add_argument("--waist-to-shoulder", type=float, default=WAIST_TO_SHOULDER_SCALE)
    parser.add_argument("--with-root", action="store_true", help="Export deploy-style CSV with root pose.")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = args.output if os.path.isabs(args.output) else os.path.join(repo_root, args.output)
    adapt_s54_dance(
        args.input,
        out_path,
        src_fps=args.src_fps,
        dst_fps=args.dst_fps,
        waist_to_shoulder=args.waist_to_shoulder,
        joint_only=not args.with_root,
    )


if __name__ == "__main__":
    main()
