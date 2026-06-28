#!/usr/bin/env python3
"""Adapt Unitree G1 LAFAN1 retarget CSV to Kuavo Gen-4 (S46/S49) 26-DoF dance reference.

Source format (g1/*.csv, 30 FPS, 36 columns):
  root_pos(3), root_quat(4),
  left/right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll),
  waist (yaw, roll, pitch),
  left/right arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)

Output format (26 columns, Kuavo RL/deploy order):
  leg_l1..6, leg_r1..6, zarm_l1..7, zarm_r1..7
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# G1 T-pose leg defaults per side: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
# (from LAFAN1 rerun_visualize.py)
G1_LEG_DEFAULT_SIDE = np.array([-0.15, 0.0, 0.0, 0.3, -0.15, 0.0], dtype=np.float64)

# Kuavo Gen-4 standing per side: leg_l1..6 = roll, yaw, pitch, knee, ankle_pitch, ankle_roll
GEN4_LEG_DEFAULT = np.array(
    [0.0, 0.0, -0.27, 0.52, -0.3, 0.0, 0.0, 0.0, -0.27, 0.52, -0.3, 0.0],
    dtype=np.float64,
)
GEN4_LEG_DEFAULT_SIDE = GEN4_LEG_DEFAULT[:6]

# G1 T-pose arms (left 7 + right 7)
G1_ARM_DEFAULT = np.array(
    [
        0.0,
        1.57,
        0.0,
        1.57,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.57,
        0.0,
        1.57,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

# Kuavo S49 in-place dance neutral arms (frame 0 of existing adapted CSV)
KUAVO_ARM_DEFAULT = np.array(
    [
        0.0186,
        0.1420,
        -0.0446,
        -0.2027,
        0.0439,
        0.0581,
        0.0,
        -0.0272,
        -0.1283,
        0.2574,
        -0.1937,
        0.0313,
        -0.0429,
        0.0,
    ],
    dtype=np.float64,
)

# Map G1 leg side [pitch, roll, yaw, knee, ankle_pitch, ankle_roll]
# to Kuavo side [roll, yaw, pitch, knee, ankle_pitch, ankle_roll]
G1_TO_KUAVO_LEG_IDX = (1, 2, 0, 3, 4, 5)
G1_TO_KUAVO_LEG_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

# Optional per-joint sign on 14 arm channels (left 7 + right 7)
G1_TO_KUAVO_ARM_SIGN = np.ones(14, dtype=np.float64)

WAIST_YAW_TO_SHOULDER = 0.35

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


def _load_g1_motion(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("body_"):
                continue
            vals = [float(x) for x in line.split(",") if x]
            if len(vals) < 36:
                continue
            rows.append(vals[:36])
    if not rows:
        raise ValueError(f"No valid G1 rows (need >=36 cols) in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, :3].copy(), data[:, 3:7].copy(), data[:, 7:36].copy()


def _map_g1_leg_side(g1_side: np.ndarray) -> np.ndarray:
    return np.array(
        [G1_TO_KUAVO_LEG_SIGN[i] * g1_side[G1_TO_KUAVO_LEG_IDX[i]] for i in range(6)],
        dtype=np.float64,
    )


def _rebase_g1_legs(g1_legs: np.ndarray) -> np.ndarray:
    g1_default_mapped = np.hstack(
        [_map_g1_leg_side(G1_LEG_DEFAULT_SIDE), _map_g1_leg_side(G1_LEG_DEFAULT_SIDE)]
    )
    mapped = np.empty_like(g1_legs)
    mapped[:, :6] = np.stack([_map_g1_leg_side(g1_legs[i, :6]) for i in range(g1_legs.shape[0])])
    mapped[:, 6:12] = np.stack([_map_g1_leg_side(g1_legs[i, 6:12]) for i in range(g1_legs.shape[0])])
    return GEN4_LEG_DEFAULT[None, :] + (mapped - g1_default_mapped[None, :])


def _g1_to_kuavo_joints(g1_joints: np.ndarray, waist_to_shoulder: float) -> np.ndarray:
    """Convert one frame batch of G1 joint vectors to Kuavo 26-DoF."""
    g1_legs = np.hstack([g1_joints[:, 0:6], g1_joints[:, 6:12]])
    waist_yaw = g1_joints[:, 12]
    g1_arms = g1_joints[:, 15:29].copy()

    legs = _rebase_g1_legs(g1_legs)
    arms = KUAVO_ARM_DEFAULT[None, :] + G1_TO_KUAVO_ARM_SIGN[None, :] * (
        g1_arms - G1_ARM_DEFAULT[None, :]
    )

    # S49 has no waist; fold torso yaw into symmetric shoulder pitch.
    arms[:, 0] += waist_to_shoulder * waist_yaw
    arms[:, 7] += waist_to_shoulder * waist_yaw
    return np.hstack([legs, arms])


def _trim_by_time(traj: np.ndarray, src_fps: int, start_sec: float | None, end_sec: float | None) -> np.ndarray:
    n = traj.shape[0]
    i0 = 0 if start_sec is None else int(max(0, round(start_sec * src_fps)))
    i1 = n if end_sec is None else int(min(n, round(end_sec * src_fps)))
    if i0 >= i1:
        raise ValueError(f"Invalid trim range: start={start_sec}, end={end_sec}, frames={n}")
    return traj[i0:i1]


def _resample(traj: np.ndarray, src_fps: int, dst_fps: int) -> np.ndarray:
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


def _smooth(traj: np.ndarray, window: int = 3) -> np.ndarray:
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


def adapt_lafan1_g1_dance(
    input_path: str,
    output_path: str,
    src_fps: int = 30,
    dst_fps: int = 50,
    start_sec: float | None = None,
    end_sec: float | None = None,
    waist_to_shoulder: float = WAIST_YAW_TO_SHOULDER,
    joint_only: bool = True,
    profile: str | None = None,
) -> None:
    root_pos, root_quat, g1_joints = _load_g1_motion(input_path)

    if start_sec is not None or end_sec is not None:
        g1_joints = _trim_by_time(g1_joints, src_fps, start_sec, end_sec)
        root_pos = _trim_by_time(root_pos, src_fps, start_sec, end_sec)
        root_quat = _trim_by_time(root_quat, src_fps, start_sec, end_sec)

    kuavo = _g1_to_kuavo_joints(g1_joints, waist_to_shoulder)
    legs = _remove_leg_drift(kuavo[:, :12])
    kuavo[:, :12] = legs
    kuavo = _smooth(kuavo, window=3)
    kuavo = _clamp_joints(kuavo)
    kuavo = _loop_closure(kuavo)
    kuavo = _resample(kuavo, src_fps, dst_fps)

    if profile is not None:
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from adapt_dance_csv import adapt_dance_trajectory

        kuavo = adapt_dance_trajectory(kuavo, profile=profile)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if joint_only:
        np.savetxt(output_path, kuavo, delimiter=",", fmt="%.6f")
    else:
        root_pos = _resample(root_pos, src_fps, dst_fps)
        root_quat = _resample(root_quat, src_fps, dst_fps)
        root_pos[:, :2] -= root_pos[0, :2]
        deploy = np.hstack([root_pos, root_quat, kuavo, np.zeros_like(kuavo)])
        np.savetxt(output_path, deploy, delimiter=",", fmt="%.6f")

    duration = (kuavo.shape[0] - 1) / float(dst_fps)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Frames: {kuavo.shape[0]} @ {dst_fps} Hz ({duration:.2f}s)")
    print(f"Frame0 leg_l5/l6: {kuavo[0, 4]:.4f}, {kuavo[0, 5]:.4f}  leg_r5/r6: {kuavo[0, 10]:.4f}, {kuavo[0, 11]:.4f}")
    print("Leg joint ranges (max-min):")
    for j in range(12):
        print(f"  leg[{j:2d}]: {kuavo[:, j].max() - kuavo[:, j].min():.3f} rad")
    print("Arm joint ranges (max-min):")
    for j in range(12, 26):
        print(f"  arm[{j:2d}]: {kuavo[:, j].max() - kuavo[:, j].min():.3f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt LAFAN1 Unitree G1 CSV to Kuavo S49/S46 dance reference.")
    parser.add_argument(
        "--input",
        default="/home/lwy/kuavo_all/LAFAN1_Retargeting_Dataset/g1/dance1_subject1.csv",
        help="Source G1 LAFAN1 CSV (36 columns).",
    )
    parser.add_argument(
        "--output",
        default="kuavo_action_LAFAN1_g1_dance1_raw.csv",
        help="Output CSV path (26 joint columns by default).",
    )
    parser.add_argument("--src-fps", type=int, default=30)
    parser.add_argument("--dst-fps", type=int, default=50)
    parser.add_argument("--start-sec", type=float, default=None, help="Optional trim start time in seconds.")
    parser.add_argument("--end-sec", type=float, default=None, help="Optional trim end time in seconds.")
    parser.add_argument("--waist-to-shoulder", type=float, default=WAIST_YAW_TO_SHOULDER)
    parser.add_argument("--with-root", action="store_true", help="Export deploy-style CSV with root pose.")
    parser.add_argument(
        "--profile",
        choices=("inplace", "expressive", "arms_only", "fullbody_inplace"),
        default=None,
        help="Optional second pass via adapt_dance_csv.py logic.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = args.output if os.path.isabs(args.output) else os.path.join(repo_root, args.output)

    adapt_lafan1_g1_dance(
        args.input,
        out_path,
        src_fps=args.src_fps,
        dst_fps=args.dst_fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        waist_to_shoulder=args.waist_to_shoulder,
        joint_only=not args.with_root,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
