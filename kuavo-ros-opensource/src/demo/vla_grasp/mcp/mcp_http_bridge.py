#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuavo NUC MCP HTTP Bridge — Python 3.8 兼容，替代 fastmcp（NUC 系统 Python 无法装 fastmcp）。

将 moveit_auto_grasp.py 中已验证的 ROS 执行逻辑暴露为 HTTP JSON API，供 Orin 大模型 / Client 调用。

依赖：
  pip3 install flask

启动（NUC root + WBC/IK/move_group 已就绪，勿与 vla_bt_daemon 同时跑）：
  source devel/setup.bash
  python3 src/demo/vla_grasp/mcp/mcp_http_bridge.py

  可选环境变量：
    KUAVO_MCP_HTTP_HOST=0.0.0.0
    KUAVO_MCP_HTTP_PORT=8765

API 示例：
  curl http://192.168.26.1:8765/health
  curl http://192.168.26.1:8765/tools
  curl http://192.168.26.1:8765/resource/robot/telemetry/status
  curl -X POST http://192.168.26.1:8765/tool/locate_target_yolo \\
       -H 'Content-Type: application/json' -d '{"object_name":"水瓶"}'
  curl -X POST http://192.168.26.1:8765/tool/calculate_ik_trajectory \\
       -H 'Content-Type: application/json' \\
       -d '{"x":0.45,"y":0.08,"z":0.385,"is_left":true}'
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLA_DIR = os.path.dirname(_SCRIPT_DIR)
if _VLA_DIR not in sys.path:
    sys.path.insert(0, _VLA_DIR)

import moveit_auto_grasp as mag
from claw_safe import build_close_cmd, build_open_cmd

import moveit_commander
import numpy as np
import rospy
from flask import Flask, jsonify, request
from kuavo_msgs.msg import armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# ROS 上下文（与 mcp_kuavo_server.py 一致）
# ---------------------------------------------------------------------------

_ARM_MODE_LOCK = threading.Lock()
_ARM_CONTROL_MODE: Optional[int] = None
_CTX_LOCK = threading.Lock()
_CTX_READY = False
_TOOL_LOCK = threading.Lock()

TTS_URL = os.environ.get("KUAVO_TTS_URL", "http://192.168.26.1:5000/tts")
HTTP_HOST = os.environ.get("KUAVO_MCP_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("KUAVO_MCP_HTTP_PORT", "8765"))


def _ensure_ros_context() -> None:
    global _CTX_READY
    with _CTX_LOCK:
        if _CTX_READY:
            return
        if not rospy.core.is_initialized():
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node("mcp_http_bridge", anonymous=True, disable_signals=True)

        rospy.Subscriber("/joint_states", JointState, mag.joint_states_callback, queue_size=1)
        deadline = time.time() + 10.0
        while not mag.has_joint_states and time.time() < deadline and not rospy.is_shutdown():
            rospy.sleep(0.05)
        if not mag.has_joint_states:
            rospy.logwarn("MCP HTTP: /joint_states 尚未对齐，部分工具可能不准确")

        mag.last_commanded_joints_rad = np.copy(mag.current_joints_rad)

        try:
            rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
                changeArmCtrlModeRequest(control_mode=2)
            )
            _set_arm_mode(2)
        except Exception as exc:
            rospy.logwarn("MCP HTTP: arm_traj_change_mode(2) 失败: %s", exc)

        _CTX_READY = True
        rospy.loginfo("MCP HTTP Bridge ROS context ready")


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
# Tool 实现（与 mcp_kuavo_server.py 像素级一致）
# ---------------------------------------------------------------------------


def tool_locate_target_yolo(params: Dict[str, Any]) -> Dict[str, Any]:
    object_name = str(params.get("object_name", "bottle"))
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


def tool_calculate_ik_trajectory(params: Dict[str, Any]) -> Dict[str, Any]:
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

    if "x" not in params or "y" not in params or "is_left" not in params:
        return {"success": False, "error": "缺少必填参数: x, y, is_left（z 可选）"}

    is_left = bool(params["is_left"])
    x = float(params["x"])
    y = float(params["y"])
    z = float(params["z"]) if params.get("z") is not None else None
    step_name = str(params.get("step_name", "MCP IK"))

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


def tool_execute_arm_motion_batch(params: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_ros_context()
    joint_chain = params.get("joint_chain")
    if not joint_chain:
        return {"success": False, "error": "joint_chain 为空"}
    if "total_time" not in params:
        return {"success": False, "error": "缺少必填参数: total_time"}

    total_time = float(params["total_time"])
    is_left = params.get("is_left")
    if is_left is not None:
        is_left = bool(is_left)
    close_claw = bool(params.get("close_claw", False))

    arm_pub = _get_arm_pub()
    chain_rad = [np.asarray(row, dtype=float) for row in joint_chain]
    for row in chain_rad:
        if row.shape[0] != 14:
            return {"success": False, "error": "每行须为 14 轴关节角（弧度）"}

    if len(chain_rad) == 1:
        mag.execute_single_pose(
            arm_pub,
            chain_rad[0],
            total_time,
            step_name="MCP HTTP single pose",
            is_left_arm=is_left,
        )
        ok = True
    else:
        ok = mag.publish_arm_trajectory_batch(
            arm_pub, chain_rad, total_time, is_left_arm=is_left
        )

    claw_result = None
    if close_claw and is_left is not None:
        pos, vel, effort = build_close_cmd(bool(is_left))
        claw_ok = mag.call_leju_claw(pos, vel, effort, tag="mcp-http-close")
        claw_result = {"claw_ok": claw_ok, "position": pos, "effort": effort}

    return {
        "success": bool(ok),
        "points": len(chain_rad),
        "total_time_sec": total_time,
        "last_commanded_joints_rad": _joints_to_json(mag.last_commanded_joints_rad),
        "claw": claw_result,
    }


def tool_emergency_joint_space_override(params: Dict[str, Any]) -> Dict[str, Any]:
    if "is_left" not in params:
        return {"success": False, "error": "缺少必填参数: is_left"}
    _ensure_ros_context()
    is_left = bool(params["is_left"])
    arm_pub = _get_arm_pub()
    q_base = np.copy(mag.last_commanded_joints_rad)
    side = "左" if is_left else "右"
    rospy.logwarn("MCP HTTP: emergency_joint_space_override (%s手)", side)
    try:
        mag.execute_vla_style_return(arm_pub, q_base, is_left)
        pos, vel, effort = build_open_cmd()
        mag.call_leju_claw(pos, vel, effort, tag="mcp-http-emergency-open")
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


def tool_tts_speak(params: Dict[str, Any]) -> Dict[str, Any]:
    if "text" not in params:
        return {"success": False, "error": "缺少必填参数: text"}
    text = str(params["text"])
    payload = json.dumps({"text": text}).encode("utf-8")
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


def tool_open_claw_both(params: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_ros_context()
    pos, vel, effort = build_open_cmd()
    ok = mag.call_leju_claw(pos, vel, effort, tag="mcp-http-open")
    return {"success": bool(ok), "position": pos, "effort": effort}


def resource_robot_telemetry_status() -> Dict[str, Any]:
    _ensure_ros_context()
    positions = _joints_to_json(mag.current_joints_rad)
    commanded = _joints_to_json(mag.last_commanded_joints_rad)
    return {
        "joint_names": list(mag.joint_names_14),
        "joint_positions_rad": positions,
        "joint_positions": positions,  # 兼容 Orin test_mcp_grasp 旧字段名
        "last_commanded_joints_rad": commanded,
        "has_joint_states": mag.has_joint_states,
        "arm_control_mode": _ARM_CONTROL_MODE,
        "arm_control_mode_hint": "0=keep 1=auto_swing 2=external_control",
        "ros_master_uri": os.environ.get("ROS_MASTER_URI", ""),
        "timestamp": time.time(),
    }


TOOL_REGISTRY = {
    "locate_target_yolo": {
        "description": "10 帧 TF2 视觉中值滤波，返回 base_link 下 (x,y) 及锁定 Z",
        "parameters": {
            "object_name": {"type": "string", "default": "bottle", "required": False},
        },
        "handler": tool_locate_target_yolo,
    },
    "calculate_ik_trajectory": {
        "description": "MoveIt IK 逆解，返回 14 轴关节角（弧度）",
        "parameters": {
            "x": {"type": "float", "required": True},
            "y": {"type": "float", "required": True},
            "z": {"type": "float", "required": False},
            "is_left": {"type": "bool", "required": True},
            "step_name": {"type": "string", "default": "MCP IK", "required": False},
        },
        "handler": tool_calculate_ik_trajectory,
    },
    "execute_arm_motion_batch": {
        "description": "批量泵送关节链至 /kuavo_arm_target_poses",
        "parameters": {
            "joint_chain": {"type": "array", "required": True, "note": "[[14 rad], ...]"},
            "total_time": {"type": "float", "required": True},
            "is_left": {"type": "bool", "required": False},
            "close_claw": {"type": "bool", "default": False, "required": False},
        },
        "handler": tool_execute_arm_motion_batch,
    },
    "emergency_joint_space_override": {
        "description": "护胸撤离宏：肩膀外摆 75° → 曲肘护胸 → init 下垂",
        "parameters": {
            "is_left": {"type": "bool", "required": True},
        },
        "handler": tool_emergency_joint_space_override,
    },
    "tts_speak": {
        "description": "POST 文本至 NUC TTS 服务",
        "parameters": {
            "text": {"type": "string", "required": True},
        },
        "handler": tool_tts_speak,
    },
    "open_claw_both": {
        "description": "双爪安全张开（claw_safe）",
        "parameters": {},
        "handler": tool_open_claw_both,
    },
}


def _invoke_tool(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"success": False, "error": "未知工具: %s" % name}
    with _TOOL_LOCK:
        try:
            return entry["handler"](params or {})
        except Exception as exc:
            rospy.logerr("MCP HTTP tool %s 异常: %s\n%s", name, exc, traceback.format_exc())
            return {"success": False, "error": str(exc), "tool": name}


# ---------------------------------------------------------------------------
# Flask HTTP API
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "kuavo-mcp-http-bridge",
        "ros_ready": _CTX_READY,
        "host": HTTP_HOST,
        "port": HTTP_PORT,
    })


@app.route("/tools", methods=["GET"])
def list_tools():
    tools = []
    for name, meta in TOOL_REGISTRY.items():
        tools.append({
            "name": name,
            "description": meta["description"],
            "parameters": meta["parameters"],
            "endpoint": "POST /tool/%s" % name,
        })
    return jsonify({
        "tools": tools,
        "resources": [
            {
                "uri": "robot://telemetry/status",
                "endpoint": "GET /resource/robot/telemetry/status",
            }
        ],
    })


@app.route("/tool/<tool_name>", methods=["POST"])
def call_tool(tool_name: str):
    params = request.get_json(silent=True) or {}
    if tool_name not in TOOL_REGISTRY:
        return jsonify({"success": False, "error": "未知工具: %s" % tool_name}), 404
    result = _invoke_tool(tool_name, params)
    return jsonify(result), 200


@app.route("/resource/robot/telemetry/status", methods=["GET"])
def get_telemetry():
    with _TOOL_LOCK:
        try:
            data = resource_robot_telemetry_status()
            # 顶层扁平返回，兼容 mcp_orin_client / test_mcp_grasp 直接读字段
            return jsonify({"success": True, **data})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500


@app.errorhandler(404)
def not_found(_exc):
    return jsonify({"success": False, "error": "not found"}), 404


def main():
    rospy.loginfo("Starting Kuavo MCP HTTP Bridge on %s:%d ...", HTTP_HOST, HTTP_PORT)
    _ensure_ros_context()
    # threaded=True：Orin 并发请求；_TOOL_LOCK 保证 ROS 调用串行
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
