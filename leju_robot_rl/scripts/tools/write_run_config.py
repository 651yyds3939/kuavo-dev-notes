#!/usr/bin/env python3
"""Generate RUN_CONFIG.md from a training run's params/env.yaml + tfevents final rewards."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 最终奖励提取（从 TensorBoard tfevents 读取尾部均值）
# ---------------------------------------------------------------------------

# 需要显示在报告里的关键指标（TensorBoard tag → 表格中的显示名）
FINAL_REWARD_TAGS = [
    ("Episode_Reward/track_punch_arms",           "track_punch_arms"),
    ("Episode_Reward/track_leg_standing",         "track_leg_standing"),
    ("Episode_Reward/track_ankle_pitch_standing", "track_ankle_pitch_standing"),
    ("Episode_Reward/penalty_ankle_pitch_standing_soft", "penalty_ankle_pitch_standing_soft"),
    ("Episode_Reward/track_punch_legs",            "track_punch_legs"),
    ("Episode_Reward/penalty_leg_joint_vel_l2",    "penalty_leg_joint_vel_l2"),
    ("Episode_Reward/penalty_leg_joint_acc_l2",    "penalty_leg_joint_acc_l2"),
    ("Episode_Reward/penalty_feet_motion_l2",      "penalty_feet_motion_l2"),
    ("Episode_Reward/penalty_feet_airborne",       "penalty_feet_airborne"),
    ("Episode_Reward/feet_slide",                  "feet_slide"),
    ("Episode_Reward/flat_orientation_l2",         "flat_orientation_l2"),
    ("Episode_Reward/base_height",                 "base_height"),
    ("Episode_Reward/penalty_root_squat",          "penalty_root_squat"),
    ("Episode_Reward/penalty_foot_pitch",          "penalty_foot_pitch"),
    ("Episode_Reward/penalty_foot_link_flat",      "penalty_foot_link_flat"),
    ("Episode_Reward/feet_contact_force_velocity", "feet_contact_force_velocity"),
    ("Episode_Reward/base_lin_vel_xy_stationary",  "base_lin_vel_xy_stationary"),
    ("Episode_Reward/penalty_root_xy_displacement","penalty_root_xy_displacement"),
    ("Episode_Reward/action_rate_l2",              "action_rate_l2"),
    ("Episode_Reward/action_smoothness_l2",        "action_smoothness_l2"),
    ("Episode_Reward/undesired_contacts",          "undesired_contacts"),
    ("Episode_Termination/base_contact",            "base_contact 终止率"),
    ("Episode_Termination/dof_pos_illegal",         "dof_pos_illegal 终止率"),
    ("Train/mean_reward",                           "mean_reward (总)"),
]


def _find_tfevents_file(run_dir: Path) -> Optional[Path]:
    """Locate the first events.out.tfevents.* file in run_dir."""
    candidates = sorted(run_dir.glob("events.out.tfevents.*"))
    return candidates[0] if candidates else None


def extract_final_rewards(run_dir: Path, tail_fraction: float = 0.10) -> dict[str, str]:
    """Extract tail-average reward values from the tfevents file.

    Args:
        run_dir: Training run directory containing events.out.tfevents.*.
        tail_fraction: Fraction of the data to average over (default 10%).

    Returns:
        Dict mapping display_name → formatted value string, or {} on failure.
    """
    tfevents = _find_tfevents_file(run_dir)
    if tfevents is None:
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}

    try:
        ea = EventAccumulator(str(run_dir))
        ea.Reload()
    except Exception:
        return {}

    available = set(ea.Tags().get("scalars", []))

    results: dict[str, str] = {}
    for tag, display_name in FINAL_REWARD_TAGS:
        if tag not in available:
            continue
        try:
            events = ea.Scalars(tag)
        except Exception:
            continue
        if not events:
            continue
        n = len(events)
        tail_n = max(1, int(n * tail_fraction))
        tail_vals = [e.value for e in events[-tail_n:]]
        avg = sum(tail_vals) / len(tail_vals)

        # 用统一的精度格式化
        if abs(avg) < 0.001:
            results[display_name] = f"{avg:.2e}"
        elif abs(avg) < 1.0:
            results[display_name] = f"{avg:.4f}"
        else:
            results[display_name] = f"{avg:.2f}"

    return results


# ---------------------------------------------------------------------------
# env.yaml 解析
# ---------------------------------------------------------------------------

@dataclass
class RewardSnapshot:
    csv_paths: list[str] = field(default_factory=list)
    track_arms_func: Optional[str] = None
    track_arms_weight: Optional[str] = None
    track_arms_std: Optional[str] = None
    track_arms_min_upright: Optional[str] = None
    track_legs_func: Optional[str] = None
    track_legs_weight: Optional[str] = None
    track_legs_std: Optional[str] = None
    track_legs_min_upright: Optional[str] = None
    arm_roll: Optional[str] = None
    flat_orientation: Optional[str] = None
    base_height: Optional[str] = None
    base_height_target: Optional[str] = None
    penalty_root_squat: Optional[str] = None
    penalty_root_min_height: Optional[str] = None
    penalty_foot_pitch: Optional[str] = None
    penalty_foot_link_flat: Optional[str] = None
    feet_contact_force_velocity: Optional[str] = None
    base_lin_vel_xy: Optional[str] = None
    penalty_root_xy_displacement: Optional[str] = None
    action_rate: Optional[str] = None
    action_smoothness: Optional[str] = None
    penalty_feet_motion: Optional[str] = None
    penalty_feet_airborne: Optional[str] = None


def _reward_block(text: str, name: str) -> Optional[str]:
    pattern = rf"  {re.escape(name)}:(.*?)(?=\n  \w[\w]*:|\nterminations:|\ncurriculum:|\nevents:|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return f"  {name}:" + match.group(1)


def _field(block: Optional[str], key: str) -> Optional[str]:
    if not block:
        return None
    match = re.search(rf"{re.escape(key)}: ([^\n]+)", block)
    return match.group(1).strip() if match else None


def parse_env_yaml(text: str) -> RewardSnapshot:
    snap = RewardSnapshot()
    snap.csv_paths = sorted(set(re.findall(r"csv_path: (\S+)", text)))

    arms = _reward_block(text, "track_punch_arms")
    legs = _reward_block(text, "track_punch_legs")
    snap.track_arms_func = _field(arms, "func")
    snap.track_arms_weight = _field(arms, "weight")
    snap.track_arms_std = _field(arms, "std")
    snap.track_arms_min_upright = _field(arms, "min_upright")
    snap.track_legs_func = _field(legs, "func")
    snap.track_legs_weight = _field(legs, "weight")
    snap.track_legs_std = _field(legs, "std")
    snap.track_legs_min_upright = _field(legs, "min_upright")

    for name, attr in (
        ("arm_roll_penalty", "arm_roll"),
        ("flat_orientation_l2", "flat_orientation"),
        ("base_lin_vel_xy_stationary", "base_lin_vel_xy"),
        ("penalty_root_xy_displacement", "penalty_root_xy_displacement"),
        ("action_rate_l2", "action_rate"),
        ("action_smoothness_l2", "action_smoothness"),
        ("penalty_foot_pitch", "penalty_foot_pitch"),
        ("penalty_foot_link_flat", "penalty_foot_link_flat"),
        ("feet_contact_force_velocity", "feet_contact_force_velocity"),
        ("penalty_feet_motion_l2", "penalty_feet_motion"),
        ("penalty_feet_airborne", "penalty_feet_airborne"),
    ):
        block = _reward_block(text, name)
        setattr(snap, attr, _field(block, "weight"))

    base_height = _reward_block(text, "base_height")
    snap.base_height = _field(base_height, "weight")
    snap.base_height_target = _field(base_height, "target_height")

    penalty_root = _reward_block(text, "penalty_root_squat")
    snap.penalty_root_squat = _field(penalty_root, "weight")
    snap.penalty_root_min_height = _field(penalty_root, "min_height")

    return snap


# ---------------------------------------------------------------------------
# 版本推断
# ---------------------------------------------------------------------------

def _float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def primary_csv(snap: RewardSnapshot) -> str:
    if not snap.csv_paths:
        return "（未找到 csv_path）"
    for path in snap.csv_paths:
        if "track_punch" in path or path.endswith(".csv"):
            return path
    return snap.csv_paths[0]


def describe_csv(csv_name: str) -> str:
    if "FROM_S54" in csv_name or csv_name.startswith("kuavo_action_S54"):
        return "S54 原生 Kuavo retarget 到 S49，yaw 小，腿幅 ~0.28 rad"
    if "HYBRID" in csv_name:
        return "LAFAN1 大步腿 + S54 手臂 merge，hip yaw 大"
    if "LAFAN1" in csv_name and "INPLACE" in csv_name:
        return "LAFAN1 G1 retarget，腿幅被压到 ~0.38 rad"
    if "LAFAN1" in csv_name and "DANCE" in csv_name:
        return "LAFAN1 G1 全幅舞蹈 retarget，腿幅 ~1.07 rad"
    return "自定义参考轨迹 CSV"


def infer_reward_version(snap: RewardSnapshot, csv_name: str) -> tuple[str, str]:
    flat = _float(snap.flat_orientation)
    action_rate = _float(snap.action_rate)
    foot_pitch = _float(snap.penalty_foot_pitch)
    foot_link_flat = _float(snap.penalty_foot_link_flat)
    root_squat = _float(snap.penalty_root_squat)
    root_xy = _float(snap.penalty_root_xy_displacement)
    arms_w = _float(snap.track_arms_weight)
    has_upright = snap.track_arms_min_upright is not None or (
        snap.track_arms_func and "upright" in snap.track_arms_func
    )

    # v18: v17 structure plus direct foot-motion/airborne suppression
    legs_w = _float(snap.track_legs_weight)
    if (
        has_upright
        and root_squat is not None and root_squat <= -14.0
        and flat is not None and flat <= -10.0
        and legs_w is not None and legs_w <= 0.1
        and root_xy is not None and -1.0 < root_xy < -0.1
        and action_rate is not None and -0.02 < action_rate < -0.001
    ):
        # Presence of both direct foot terms distinguishes v18 from v17.
        if snap.penalty_feet_motion is not None and snap.penalty_feet_airborne is not None:
            return "v18", "双脚静止接触 + 腿部连续平衡微调 + 手臂 CSV（禁止小碎步）"

    # v17: arms-only deployable + fixed-leg reference observation + mild root anchor + leg damping
    legs_w = _float(snap.track_legs_weight)
    if (
        has_upright
        and root_squat is not None and root_squat <= -14.0
        and flat is not None and flat <= -10.0
        and foot_link_flat is None
        and foot_pitch is None
        and legs_w is not None and legs_w <= 0.1
        and root_xy is not None and -1.0 < root_xy < -0.1
        and action_rate is not None and -0.02 < action_rate < -0.001
    ):
        return "v17", "腿部观测/奖励统一固定站姿 + 手臂 CSV + 温和原地锚定/腿部阻尼/踝关节防勾脚"

    # v15/v16: arms upright gate + v2 stability + legs zero weight + no foot constraints
    if (
        has_upright
        and root_squat is not None and root_squat <= -14.0
        and flat is not None and flat <= -10.0
        and foot_link_flat is None
        and foot_pitch is None
    ):
        if legs_w is not None and legs_w <= 0.1:
            return "v15/v16", "纯上半身手臂跟踪 + 腿部站姿奖励（旧识别逻辑，检查 env.yaml 细节）"
        return "v14", "v12 底座 + foot_pitch 强攻勾脚 + 移除 mimic 门控 min_height"

    # v11: arms upright gate + v2-level squat (>=8) + foot_link_flat present + action_rate ~ -0.12
    if (
        has_upright
        and root_squat is not None
        and root_squat <= -7.0
        and foot_link_flat is not None
        and action_rate is not None
        and action_rate <= -0.08
    ):
        return "v11", "手臂 upright 门控 + v2 级防跪 + 温和脚下约束"

    if "HYBRID" in csv_name and action_rate is not None and action_rate >= -0.001:
        return "v5", "HYBRID CSV + 强 mimic + 极弱 smoothness"

    if "LAFAN1" in csv_name and "INPLACE" in csv_name and foot_pitch is not None and foot_pitch <= -5:
        return "v3", "LAFAN1 INPLACE + anti toe-up"

    if foot_link_flat is not None and foot_link_flat <= -1.5:
        if foot_pitch is not None and foot_pitch <= -6.0:
            return "v9", "S54 + 脚掌贴地过强（易一出生就倒，勿用）"
        return "v9.1", "S54 + 防漂移/丝滑 + 脚掌贴地（易崩溃，勿用）"

    if (
        ("FROM_S54" in csv_name or "S54" in csv_name)
        and action_rate is not None
        and action_rate <= -0.08
        and root_xy is not None
        and root_xy <= -1.0
        and foot_pitch is not None
        and -5.0 < foot_pitch <= -3.5
        and foot_link_flat is None
    ):
        return "v10", "v8 基线 + 适度防勾脚（无脚板 link 惩罚）"

    if ("FROM_S54" in csv_name or "S54" in csv_name) and root_xy is not None and root_xy <= -1.0:
        return "v7", "S54 + 强防漂移 + root XY 锚定（修 v6 后退）"

    if ("FROM_S54" in csv_name or "S54" in csv_name) and action_rate is not None and action_rate >= -0.06:
        return "v6", "S54 + 轻 stable / 易后退"

    if ("FROM_S54" in csv_name or "S54" in csv_name) and action_rate is not None and action_rate <= -0.08:
        if root_xy is not None and root_xy <= -1.0:
            return "v8", "S54 + 强 smoothness + 防漂移（易勾脚）"
        return "v8", "S54 + 强 smoothness"

    if flat is not None and flat <= -10 and root_squat is not None and root_squat <= -10:
        return "v2", "anti-kneel，S54 + 强稳定 + upright gate"

    if flat is not None and flat <= -6 and (root_squat is None or root_squat > -10):
        return "v1.5", "S54 + 轻度 upright gate + 中等稳定"

    if flat is not None and flat >= -2 and not has_upright:
        return "v1", "早期 S49 punch，弱稳定项"

    if arms_w is not None and arms_w >= 20:
        return "v4/v5", "强 mimic 配置（自动识别）"

    return "custom", "未能精确匹配已知版本，见下方权重表"


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------

def fmt_track_arms(snap: RewardSnapshot) -> str:
    if snap.track_arms_weight is None:
        return "无"
    parts = [snap.track_arms_weight]
    if snap.track_arms_std:
        parts.append(f"std={snap.track_arms_std}")
    if snap.track_arms_func and "upright" in snap.track_arms_func:
        parts.append(f"upright_exp，min_upright={snap.track_arms_min_upright or '?'}")
    else:
        parts.append("无 upright gate")
    return "，".join(parts)


def fmt_track_legs(snap: RewardSnapshot) -> str:
    if snap.track_legs_weight is None:
        return "无"
    parts = [snap.track_legs_weight]
    if snap.track_legs_std:
        parts.append(f"std={snap.track_legs_std}")
    if snap.track_legs_func and "upright" in snap.track_legs_func:
        parts.append(f"upright_exp，min_upright={snap.track_legs_min_upright or '?'}")
    else:
        parts.append("无 upright gate")
    return "，".join(parts)


def fmt_optional(weight: Optional[str], extra: str = "") -> str:
    if weight is None:
        return "无"
    return f"{weight}{extra}"


def find_checkpoints(run_dir: Path) -> list[str]:
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    return [p.name for p in models]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def write_run_config(run_dir: Path, *, training_status: str = "completed") -> Path:
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.is_file():
        raise FileNotFoundError(f"Missing {env_yaml}")

    snap = parse_env_yaml(env_yaml.read_text(encoding="utf-8"))
    csv_name = primary_csv(snap)
    version, version_desc = infer_reward_version(snap, csv_name)
    checkpoints = find_checkpoints(run_dir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    status_label = {
        "completed": "训练正常结束",
        "started": "训练已开始（配置快照）",
        "interrupted": "训练中断，配置来自已保存的 env.yaml",
    }.get(training_status, training_status)

    # --- 训练结果行 ---
    if training_status == "completed":
        result_line = "<!-- TODO: 看视频后手动填入，例如：站立跳舞 / 跪着跳舞 / 勾脚跳舞 / 乱飘 / 手臂不动 -->"
    elif training_status == "interrupted":
        result_line = "训练中断，待确认"
    else:
        result_line = "训练进行中…"

    # --- checkpoint 列表 ---
    if checkpoints:
        ckpt_lines = f"- 检查点：{', '.join(checkpoints[-5:])}"
        if len(checkpoints) > 5:
            ckpt_lines += f"（共 {len(checkpoints)} 个，仅列最近 5 个）"
    else:
        ckpt_lines = "- 检查点：尚无 model_*.pt"

    # --- 最终奖励值 ---
    final_rewards = {}
    if training_status in ("completed", "interrupted"):
        final_rewards = extract_final_rewards(run_dir)

    final_reward_table = ""
    if final_rewards:
        rows = "\n".join(
            f"| {name} | {value} |" for name, value in final_rewards.items()
        )
        final_reward_table = f"""
## 最终收敛奖励值（尾部 10% 均值，来自 TensorBoard）

| 指标 | 收敛值 |
|------|--------|
{rows}
"""
    else:
        final_reward_table = f"""
## 最终收敛奖励值

<!-- 未找到 TensorBoard tfevents 文件；训练中请等待结束后重跑
     `python3 scripts/tools/write_run_config.py {run_dir.name} --status completed` -->
"""

    # --- 组装 ---
    content = f"""# 训练配置记录

训练结果：{result_line}

| 字段 | 值 |
|------|-----|
| **Run ID** | `{run_dir.name}` |
| **奖励版本** | **{version}**（{version_desc}） |
| **参考 CSV** | `{csv_name}` |
| **CSV 来源** | {describe_csv(csv_name)} |
| **文档状态** | {status_label} |
| **自动生成时间** | {now} |

## 关键奖励权重（来自 `params/env.yaml`）

| 项 | 值 |
|----|-----|
| track_punch_arms | {fmt_track_arms(snap)} |
| track_punch_legs | {fmt_track_legs(snap)} |
| arm_roll_penalty | {fmt_optional(snap.arm_roll)} |
| flat_orientation_l2 | {fmt_optional(snap.flat_orientation)} |
| base_height | {fmt_optional(snap.base_height, f"，target={snap.base_height_target}" if snap.base_height_target else "")} |
| penalty_root_squat | {fmt_optional(snap.penalty_root_squat, f"，min_height={snap.penalty_root_min_height}" if snap.penalty_root_min_height else "")} |
| penalty_foot_pitch | {fmt_optional(snap.penalty_foot_pitch)} |
| penalty_foot_link_flat | {fmt_optional(snap.penalty_foot_link_flat)} |
| feet_contact_force_velocity | {fmt_optional(snap.feet_contact_force_velocity)} |
| base_lin_vel_xy_stationary | {fmt_optional(snap.base_lin_vel_xy)} |
| penalty_root_xy_displacement | {fmt_optional(snap.penalty_root_xy_displacement)} |
| action_rate_l2 | {fmt_optional(snap.action_rate)} |
| action_smoothness_l2 | {fmt_optional(snap.action_smoothness)} |
{final_reward_table}
## 备注

{ckpt_lines}
- 训练结果行由人肉看视频后填入；其余字段自动生成。
- 本文件由 `scripts/tools/write_run_config.py` 自动生成，请勿手改；重跑脚本可覆盖更新。
"""

    out_path = run_dir / "RUN_CONFIG.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate RUN_CONFIG.md for training runs.")
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Run directory(ies) containing params/env.yaml",
    )
    parser.add_argument(
        "--all-under",
        type=Path,
        help="Regenerate RUN_CONFIG.md for every subdir that has params/env.yaml",
    )
    parser.add_argument(
        "--status",
        choices=("completed", "started", "interrupted"),
        default="completed",
        help="Training status note embedded in the markdown",
    )
    args = parser.parse_args(argv)

    run_dirs: list[Path] = list(args.run_dirs)
    if args.all_under:
        root = args.all_under.resolve()
        run_dirs.extend(
            sorted({p.parent.parent for p in root.glob("*/params/env.yaml")})
        )

    if not run_dirs:
        parser.error("Provide run_dirs and/or --all-under")

    errors = 0
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        try:
            out = write_run_config(run_dir, training_status=args.status)
            print(f"[OK] {out}")
        except Exception as exc:  # noqa: BLE001 — CLI tool reports all failures
            print(f"[FAIL] {run_dir}: {exc}", file=sys.stderr)
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
