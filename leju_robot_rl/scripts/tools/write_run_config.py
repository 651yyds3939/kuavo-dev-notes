#!/usr/bin/env python3
"""Generate RUN_CONFIG.md from a training run's params/env.yaml."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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
    base_lin_vel_xy: Optional[str] = None
    penalty_root_xy_displacement: Optional[str] = None
    action_rate: Optional[str] = None
    action_smoothness: Optional[str] = None


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
    root_squat = _float(snap.penalty_root_squat)
    arms_w = _float(snap.track_arms_weight)
    upright = snap.track_arms_min_upright
    has_upright = upright is not None or (
        snap.track_arms_func and "upright" in snap.track_arms_func
    )

    base_lin_vel = _float(snap.base_lin_vel_xy)
    root_xy = _float(snap.penalty_root_xy_displacement)

    if "HYBRID" in csv_name and action_rate is not None and action_rate >= -0.001:
        return "v5", "HYBRID CSV + 强 mimic + 极弱 smoothness"

    if "LAFAN1" in csv_name and "INPLACE" in csv_name and foot_pitch is not None and foot_pitch <= -5:
        return "v3", "LAFAN1 INPLACE + anti toe-up"

    if ("FROM_S54" in csv_name or "S54" in csv_name) and action_rate is not None and action_rate <= -0.01:
        if root_xy is not None and root_xy <= -1.0:
            return "v7", "S54 CSV + 强防漂移 + root XY 锚定（修 v6 后退）"
        if base_lin_vel is not None and base_lin_vel >= -0.1:
            return "v6", "S54 CSV + 官方量级 smoothness + 轻稳定（易后退）"
        return "v6/v7", "S54 CSV + 强 smoothness"

    if flat is not None and flat <= -10 and root_squat is not None and root_squat <= -10:
        return "v2", "anti-kneel，S54 + 强稳定 + upright gate"

    if flat is not None and flat <= -6 and (root_squat is None or root_squat > -10):
        return "v1.5", "S54 + 轻度 upright gate + 中等稳定"

    if flat is not None and flat >= -2 and not has_upright:
        return "v1", "早期 S49 punch，弱稳定项"

    if arms_w is not None and arms_w >= 20:
        return "v4/v5", "强 mimic 配置（自动识别）"

    return "custom", "未能精确匹配已知版本，见下方权重表"


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


def write_run_config(run_dir: Path, *, training_status: str = "completed") -> Path:
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.is_file():
        raise FileNotFoundError(f"Missing {env_yaml}")

    snap = parse_env_yaml(env_yaml.read_text(encoding="utf-8"))
    csv_name = primary_csv(snap)
    version, version_desc = infer_reward_version(snap, csv_name)
    checkpoints = find_checkpoints(run_dir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    status_line = {
        "completed": "训练正常结束",
        "started": "训练已开始（配置快照）",
        "interrupted": "训练中断，配置来自已保存的 env.yaml",
    }.get(training_status, training_status)

    ckpt_lines = ""
    if checkpoints:
        ckpt_lines = f"- 检查点：{', '.join(checkpoints[-5:])}"
        if len(checkpoints) > 5:
            ckpt_lines += f"（共 {len(checkpoints)} 个，仅列最近 5 个）"
    else:
        ckpt_lines = "- 检查点：尚无 model_*.pt"

    content = f"""# 训练配置记录

| 字段 | 值 |
|------|-----|
| **Run ID** | `{run_dir.name}` |
| **奖励版本** | **{version}**（{version_desc}） |
| **参考 CSV** | `{csv_name}` |
| **CSV 来源** | {describe_csv(csv_name)} |
| **文档状态** | {status_line} |
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
| base_lin_vel_xy_stationary | {fmt_optional(snap.base_lin_vel_xy)} |
| penalty_root_xy_displacement | {fmt_optional(snap.penalty_root_xy_displacement)} |
| action_rate_l2 | {fmt_optional(snap.action_rate)} |
| action_smoothness_l2 | {fmt_optional(snap.action_smoothness)} |

## 备注

{ckpt_lines}
- 本文件由 `scripts/tools/write_run_config.py` 自动生成，请勿手改；重跑脚本可覆盖更新。
"""

    out_path = run_dir / "RUN_CONFIG.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


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
