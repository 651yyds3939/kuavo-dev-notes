#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuavo NUC MCP Server — 将 moveit_auto_grasp.py 中已验证的 ROS 执行逻辑暴露为 MCP 原子工具。

部署：可运行于 NUC（与 WBC 同机）或拷贝至 Orin（ROS_MASTER_URI 指向 NUC）。
依赖：pip install fastmcp

启动：
  source devel/setup.bash
  python3 src/demo/vla_grasp/mcp/mcp_kuavo_server.py

⚠️ 不修改 vla_bt_daemon.py / bt/ — 本文件为独立 MCP 升级通道。
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLA_DIR = os.path.dirname(_SCRIPT_DIR)
if _VLA_DIR not in sys.path:
    sys.path.insert(0, _VLA_DIR)

import moveit_auto_grasp as mag
from claw_safe import build_close_cmd, build_open_cmd

import moveit_commander
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from kuavo_msgs.msg import armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from sensor_msgs.msg import JointState

try:
    from fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit(
        "fastmcp 未安装。请执行: pip install fastmcp\n原始错误: %s" % exc
    )

# ---------------------------------------------------------------------------
# ROS 上下文（单例，线程安全初始化）
# ---------------------------------------------------------------------------

_ARM_MODE_LOCK = threading.Lock()
_ARM_CONTROL_MODE: Optional[int] = None
_CTX_LOCK = threading.Lock()
_CTX_READY = False


def _ensure_ros_context() -> None:
    global _CTX_READY
    with _CTX_LOCK:
        if _CTX_READY:
            return
        if not rospy.core.is_initialized():
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node("mcp_kuavo_server", anonymous=True, disable_signals=True)

        rospy.Subscriber("/joint_states", JointState, mag.joint_states_callback, queue_size=1)
        deadline = time.time() + 10.0
        while not mag.has_joint_states and time.time() < deadline and not rospy.is_shutdown():
            rospy.sleep(0.05)
        if not mag.has_joint_states:
            rospy.logwarn("MCP: /joint_states 尚未对齐，部分工具可能不准确")

        mag.last_commanded_joints_rad = np.copy(mag.current_joints_rad)

        try:
            rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
                changeArmCtrlModeRequest(control_mode=2)
            )
            _set_arm_mode(2)
        except Exception as exc:
            rospy.logwarn("MCP: arm_traj_change_mode(2) 失败: %s", exc)

        _CTX_READY = True
        rospy.loginfo("MCP Kuavo ROS context ready")


def _get_arm_pub() -> rospy.Publisher:
    _ensure_ros_context()
    pub = getattr(_get_arm_pub, "_pub", None)
    if pub is None:
        pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
        _get_arm_pub._pub = pub
        rospy.sleep(0.2)
    return pub


def _get_moveit_arm(is_left: bool):
    _ensure_ros_context()
    cache = getattr(_get_moveit_arm, "_cache", {})
    key = "left" if is_left else "right"
    if key not in cache:
        grp = moveit_commander.MoveGroupCommander("left_arm" if is_left else "right_arm")
        grp.set_pose_reference_frame("base_link")
        cache[key] = grp
        _get_moveit_arm._cache = cache
    return cache[key]


def _set_arm_mode(mode: int) -> None:
    global _ARM_CONTROL_MODE
    with _ARM_MODE_LOCK:
        _ARM_CONTROL_MODE = int(mode)


def _locked_xyz_from_raw(x: float, y: float, z: Optional[float], is_left: bool):
    off_x, off_y = mag.tcp_offsets_for_arm(is_left)
    lx = float(x) + off_x
    ly = float(y) + off_y
    lz = float(z) if z is not None else mag.SAFE_LOCKED_Z
    return lx, ly, lz


def _joints_to_json(joints_rad: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(joints_rad, dtype=float).reshape(-1)[:14]]


# ---------------------------------------------------------------------------
# FastMCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="Kuavo Embodied MCP",
    instructions=(
        "Kuavo 14-DOF 人形机器人下位机专家工具集。"
        "视觉采点、MoveIt IK、WBC 批量航点、护胸撤离宏、TTS。"
        "奇异点/无解时 read tool 返回的 error 字段并重试或调用 emergency_joint_space_override。"
    ),
)

TTS_URL = os.environ.get("KUAVO_TTS_URL", "http://192.168.26.1:5000/tts")


@mcp.tool()
def locate_target_yolo(object_name: str = "bottle") -> Dict[str, Any]:
    """
    10 帧 TF2 视觉中值滤波（与 moveit_auto_grasp._collect_vision_targets_tf2_style 一致）。
    斩杀射灯闪烁，返回 base_link 下纯净 (x, y) 及建议锁定 Z。
    object_name 预留语义标签（当前 YOLO 管线按视野内主目标采点）。
    """
    _ensure_ros_context()
    x_hist, y_hist = mag._collect_vision_targets_tf2_style()
    if len(x_hist) < 10:
        return {
            "success": False,
            "error": "YOLO 采点不足 10 帧。请确认 /vla/yolo_target 发布且 x∈[0.30,0.65]。",
            "object_name": object_name,
            "samples": len(x_hist),
        }
    raw_x = float(np.median(x_hist))
    raw_y = float(np.median(y_hist))
    is_left = raw_y > 0.0
    off_x, off_y = mag.tcp_offsets_for_arm(is_left)
    return {
        "success": True,
        "object_name": object_name,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "locked_x": raw_x + off_x,
        "locked_y": raw_y + off_y,
        "locked_z": mag.SAFE_LOCKED_Z,
        "is_left_arm": is_left,
        "tcp_offset_x": off_x,
        "tcp_offset_y": off_y,
        "samples": 10,
    }


@mcp.tool()
def calculate_ik_trajectory(
    x: float,
    y: float,
    z: float,
    is_left: bool,
    step_name: str = "MCP IK",
) -> Dict[str, Any]:
    """
    调用 MoveIt /compute_ik + 官方下垂 Seed（OFFICIAL_IK_SEED_7），
    与 moveit_auto_grasp._solve_pose_ik 完全一致。
    x/y/z 为 base_link 目标点（米）；内部应用 tcp_offsets_for_arm。
    返回 14 轴关节角（弧度）。
    """
    _ensure_ros_context()
    ik_client = mag._resolve_ik_service()
    if ik_client is None:
        return {
            "success": False,
            "error": (
                "MoveIt IK 服务不可用。请确认 move_group.launch 已启动，"
                "且 /compute_ik 或 /move_group/compute_ik 可访问。"
            ),
        }

    is_left = bool(is_left)
    arm_group = _get_moveit_arm(is_left)
    lx, ly, lz = _locked_xyz_from_raw(x, y, z, is_left)
    quat = mag.get_horizontal_claw_quat(lx, ly, is_left)
    pose = mag._build_pose_stamped(lx, ly, lz, quat)

    try:
        _, ee_link = mag._ik_group_profile(is_left)
        arm_group.set_end_effector_link(ee_link)
    except Exception:
        pass

    seed = np.copy(mag.last_commanded_joints_rad)
    q = mag._solve_pose_ik(ik_client, arm_group, is_left, pose, seed, step_name)
    if q is None:
        side = "左" if is_left else "右"
        return {
            "success": False,
            "error": (
                "%s IK 无解（运动学奇异或目标不可达）。"
                "建议：调用 emergency_joint_space_override(is_left=%s) 撤离，"
                "或调整 x/y/z（当前锁定点 x=%.3f y=%.3f z=%.3f）。"
                % (side, is_left, lx, ly, lz)
            ),
            "locked_x": lx,
            "locked_y": ly,
            "locked_z": lz,
            "is_left_arm": is_left,
        }

    mag.last_commanded_joints_rad = np.copy(q)
    return {
        "success": True,
        "joints_rad": _joints_to_json(q),
        "locked_x": lx,
        "locked_y": ly,
        "locked_z": lz,
        "is_left_arm": is_left,
        "ee_link": mag._ik_group_profile(is_left)[1],
    }


@mcp.tool()
def execute_arm_motion_batch(
    joint_chain: List[List[float]],
    total_time: float,
    is_left: Optional[bool] = None,
    close_claw: bool = False,
) -> Dict[str, Any]:
    """
    将关节链批量泵送至 /kuavo_arm_target_poses（publish_arm_trajectory_batch）。
    joint_chain: [[14 rad], ...]；单点链退化为 execute_single_pose。
    total_time: 整段运动秒数。
  close_claw: 是否在运动后安全闭合夹爪（claw_safe 限幅）。
    """
    _ensure_ros_context()
    if not joint_chain:
        return {"success": False, "error": "joint_chain 为空"}

    arm_pub = _get_arm_pub()
    chain_rad = [np.asarray(row, dtype=float) for row in joint_chain]
    for row in chain_rad:
        if row.shape[0] != 14:
            return {"success": False, "error": "每行须为 14 轴关节角（弧度）"}

    if len(chain_rad) == 1:
        mag.execute_single_pose(
            arm_pub,
            chain_rad[0],
            float(total_time),
            step_name="MCP single pose",
            is_left_arm=is_left,
        )
        ok = True
    else:
        ok = mag.publish_arm_trajectory_batch(
            arm_pub, chain_rad, float(total_time), is_left_arm=is_left
        )

    claw_result = None
    if close_claw and is_left is not None:
        pos, vel, effort = build_close_cmd(bool(is_left))
        claw_ok = mag.call_leju_claw(pos, vel, effort, tag="mcp-close")
        claw_result = {"claw_ok": claw_ok, "position": pos, "effort": effort}

    return {
        "success": bool(ok),
        "points": len(chain_rad),
        "total_time_sec": float(total_time),
        "last_commanded_joints_rad": _joints_to_json(mag.last_commanded_joints_rad),
        "claw": claw_result,
    }


@mcp.tool()
def emergency_joint_space_override(is_left: bool) -> Dict[str, Any]:
    """
    22.1/22.2 真机验证的护胸撤离宏：肩膀外摆 75° → 曲肘护胸 ready_angles → init 下垂。
    封装自 execute_vla_style_return / _build_high_safe_joints / _auto_grasp_ready_deg。
    IK 失败或奇异时由大模型主动调用。
    """
    _ensure_ros_context()
    is_left = bool(is_left)
    arm_pub = _get_arm_pub()
    q_base = np.copy(mag.last_commanded_joints_rad)
    side = "左" if is_left else "右"
    rospy.logwarn("MCP: emergency_joint_space_override (%s手)", side)
    try:
        mag.execute_vla_style_return(arm_pub, q_base, is_left)
        pos, vel, effort = build_open_cmd()
        mag.call_leju_claw(pos, vel, effort, tag="mcp-emergency-open")
        return {
            "success": True,
            "message": "%s手已执行肩膀外摆 %d° → 曲肘护胸 → 初始下垂" % (
                side, int(mag.SHOULDER_SWING_AVOID_DEG)
            ),
            "ready_angles_deg": mag._auto_grasp_ready_deg(is_left),
            "init_angles_deg": mag.DUAL_ARM_INIT_DEG,
        }
    except Exception as exc:
        return {"success": False, "error": "撤离宏执行异常: %s" % exc}


@mcp.tool()
def tts_speak(text: str) -> Dict[str, Any]:
    """向下位机 TTS 服务 POST JSON {\"text\": ...}，默认 http://192.168.26.1:5000/tts"""
    payload = json.dumps({"text": str(text)}).encode("utf-8")
    req = urllib.request.Request(
        TTS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return {"success": resp.status == 200, "status": resp.status, "url": TTS_URL}
    except urllib.error.URLError as exc:
        return {"success": False, "error": str(exc), "url": TTS_URL}


@mcp.tool()
def open_claw_both() -> Dict[str, Any]:
    """双爪安全张开（claw_safe.build_open_cmd）。"""
    _ensure_ros_context()
    pos, vel, effort = build_open_cmd()
    ok = mag.call_leju_claw(pos, vel, effort, tag="mcp-open")
    return {"success": bool(ok), "position": pos, "effort": effort}


@mcp.resource("robot://telemetry/status")
def robot_telemetry_status() -> str:
    """
    只读遥测：14 轴 joint_states + 末次下发关节 + 臂控 Mode（若已知）。
  供大模型渐进式披露。
    """
    _ensure_ros_context()
    payload = {
        "joint_names": mag.joint_names_14,
        "joint_positions_rad": _joints_to_json(mag.current_joints_rad),
        "last_commanded_joints_rad": _joints_to_json(mag.last_commanded_joints_rad),
        "has_joint_states": mag.has_joint_states,
        "arm_control_mode": _ARM_CONTROL_MODE,
        "arm_control_mode_hint": "0=keep 1=auto_swing 2=external_control",
        "ros_master_uri": os.environ.get("ROS_MASTER_URI", ""),
        "timestamp": time.time(),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main():
    rospy.loginfo("Starting Kuavo MCP Server (fastmcp)...")
    _ensure_ros_context()
    mcp.run()


if __name__ == "__main__":
    main()
