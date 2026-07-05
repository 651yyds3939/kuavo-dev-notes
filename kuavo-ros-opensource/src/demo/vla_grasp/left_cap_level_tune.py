#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
左臂关节空间调平：无相机、无 IK、右手不动。

两步 workflow
-------------
1) 常规跑一次双臂（带相机），触顶时保存左臂关节:
   python3 src/demo/vla_grasp/bimanual_unscrew.py \\
     _twist_steps:=0 _right_hold_after_grasp:=true _save_left_tune_pose:=true

   → 生成 src/demo/vla_grasp/left_cap_tune_pose.json

2) 无相机，只动左臂到保存位，再交互调 l1-l7:
   python3 src/demo/vla_grasp/left_cap_level_tune.py

   右臂始终锁定为当前 /joint_states（不会动）。

交互: 5+10 / 6-5 (指定度数) | 5+ / 5- (默认步长10°) | p | h | w | q 退出并恢复启动时姿态

也可直接传关节角(度):
  python3 .../_left_joints_deg:="[20.1,-5.3,...]"   # 7 个数
"""
from __future__ import print_function

import json
import math
import os
import re
import signal
import sys

import numpy as np
import rospy
from sensor_msgs.msg import JointState

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import moveit_auto_grasp as mag
import bimanual_unscrew as bu
from kuavo_msgs.msg import armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

LEFT_JOINT_LABELS = [
    "l1 肩", "l2", "l3", "l4 肘", "l5★水平主", "l6 俯仰辅", "l7 拧盖z",
]

# 退出时恢复左臂（Ctrl+C / q 均走 finally）
_EXIT_CTX = {
    "arm_pub": None,
    "q_arm_home": None,
    "q_right_hold": None,
    "restore_sec": 4.0,
    "restore_on_exit": True,
    "restored": False,
}


def _restore_left_arm_home(reason=""):
    ctx = _EXIT_CTX
    if ctx["restored"] or not ctx.get("restore_on_exit"):
        return
    arm_pub = ctx.get("arm_pub")
    q_home = ctx.get("q_arm_home")
    q_r = ctx.get("q_right_hold")
    if arm_pub is None or q_home is None or q_r is None:
        return
    label = "退出→恢复脚本启动时左臂姿态"
    if reason:
        label += " (%s)" % reason
    try:
        bu.execute_hold_right(
            arm_pub, q_home, ctx.get("restore_sec", 4.0), q_r, label,
        )
        print("✅ 左臂已回到启动时姿态 (右臂未动)")
    except Exception as exc:
        print("❌ 恢复左臂失败:", exc)
    ctx["restored"] = True


def _on_sigint(signum, frame):
    """勿用 SIG_DFL，否则进程直接退出、finally 不执行。"""
    print("\n⚠️ Ctrl+C，正在恢复左臂 …")
    raise KeyboardInterrupt


def _resolve_left_deg_from_params():
    """~left_joints_deg 优先；否则读 pose 文件。"""
    if rospy.has_param("~left_joints_deg"):
        raw = rospy.get_param("~left_joints_deg")
        if isinstance(raw, str):
            raw = json.loads(raw.replace("'", '"'))
        deg = [float(x) for x in raw]
        if len(deg) != 7:
            raise ValueError("left_joints_deg 须 7 个数，got %d" % len(deg))
        return deg, "(ros param ~left_joints_deg)"

    pose_file = rospy.get_param("~pose_file", bu.left_tune_pose_path())
    if not os.path.isfile(pose_file):
        raise FileNotFoundError(
            "找不到 %s\n请先运行:\n"
            "  python3 src/demo/vla_grasp/bimanual_unscrew.py "
            "_twist_steps:=0 _save_left_tune_pose:=true" % pose_file
        )
    data = bu.load_left_cap_tune_pose(pose_file)
    if "left_joints_deg" in data:
        return [float(x) for x in data["left_joints_deg"]], pose_file
    return [math.degrees(float(x)) for x in data["left_joints_rad"]], pose_file


def _open_left_claw_hold_right(q_right_hold):
    pos, vel, eff = bu.build_left_claw_cmd_hold_right(12.0, 0.15)
    from claw_safe import get_controller
    get_controller().call(pos, vel, eff, tag="tune-open-left")


def _print_tune_status(q_baseline, q_session_start, step_deg):
    q = mag.last_commanded_joints_rad
    print("\n" + "=" * 58)
    print("步长=%.2f° | 左臂 (deg) | Δ会话 | Δ相对保存位" % step_deg)
    print("-" * 58)
    for i in range(7):
        deg = math.degrees(q[i])
        print("  %s: %8.2f   Δsess %+.2f   Δbase %+.2f" % (
            LEFT_JOINT_LABELS[i], deg,
            math.degrees(q[i] - q_session_start[i]),
            math.degrees(q[i] - q_baseline[i]),
        ))
    print("-" * 58)
    print("建议 bimanual_unscrew DEFAULTS (rad, 相对 IK 触顶解):")
    print("  left_cap_wrist5_bias_rad: %.4f   # l5 主调平" % (q[4] - q_baseline[4]))
    print("  left_cap_wrist6_bias_rad: %.4f   # l6 辅调平" % (q[5] - q_baseline[5]))
    print("  l7 不参与调平；拧盖由 twist_cap 逐步 +l7")
    print("  (当前 l7 相对 IK: %+.2f°，勿写入 wrist bias)" % math.degrees(q[6] - q_baseline[6]))
    print("复制整组 left_joints_deg 覆盖 pose 文件也可直接复现。")
    print("  left_joints_deg:", [round(math.degrees(q[i]), 3) for i in range(7)])
    print("=" * 58)


def _print_help(step_deg):
    print("""
5+10  6-5     关节 ±指定度数 (如 l5 +10°)
5+  5-        关节 ±默认步长 (当前 %.1f°)
p             打印角度
s 10          改默认步长
h             恢复进入调平时的姿态 (保存触顶位)
o             左爪半开
w             写入 left_cap_tune_pose.json
q             退出，左臂回到脚本启动时的姿态
""" % step_deg)


def _parse_joint_cmd(line, step_deg):
    """
    解析 5+10 / 6- / 7+15.5 → (joint 1-7, delta_deg) 或 None。
    """
    m = re.match(r"^([1-7])([+\-])([\d.]+)?$", line.strip())
    if not m:
        return None
    j = int(m.group(1))
    sign = 1.0 if m.group(2) == "+" else -1.0
    if m.group(3) is not None and m.group(3) != "":
        mag_deg = float(m.group(3))
    else:
        mag_deg = step_deg
    return j, sign * mag_deg


def interactive_tune(arm_pub, q_right_hold, q_baseline, move_sec):
    q_session_start = np.copy(mag.last_commanded_joints_rad)
    step_deg = float(rospy.get_param("~step_deg", 10.0))
    _print_help(step_deg)
    _print_tune_status(q_baseline, q_session_start, step_deg)

    while not rospy.is_shutdown():
        try:
            line = input("\n>>> ").strip().lower()
        except KeyboardInterrupt:
            print("")
            return "interrupt"
        except EOFError:
            break
        if not line:
            continue
        if line in ("q", "quit", "exit"):
            return "quit"
        if line in ("?", "help"):
            _print_help(step_deg)
            continue
        if line == "p":
            _print_tune_status(q_baseline, q_session_start, step_deg)
            continue
        if line == "h":
            bu.execute_hold_right(
                arm_pub, q_session_start, move_sec, q_right_hold, "恢复调平起点",
            )
            continue
        if line == "o":
            _open_left_claw_hold_right(q_right_hold)
            continue
        if line == "w":
            q = mag.last_commanded_joints_rad
            meta = {}
            path = bu.save_left_cap_tune_pose(q, q_right_hold, meta)
            print("已写入", path)
            continue
        if line.startswith("s"):
            parts = line.split()
            if len(parts) >= 2:
                step_deg = float(parts[1])
                print("默认步长 → %.2f°" % step_deg)
            continue
        parsed = _parse_joint_cmd(line, step_deg)
        if parsed is not None:
            j, delta = parsed
            slot = j - 1
            q = np.copy(mag.last_commanded_joints_rad)
            q[slot] += math.radians(delta)
            q[bu.RIGHT_SLICE] = q_right_hold[bu.RIGHT_SLICE]
            bu.execute_hold_right(
                arm_pub, q, move_sec, q_right_hold, "l%d %+.1f°" % (j, delta),
            )
            continue
        print("未知命令。例: 5+10  6-  p  q")
    return "eof"


def main():
    rospy.init_node("left_cap_level_tune")
    signal.signal(signal.SIGINT, _on_sigint)

    goto_sec = float(rospy.get_param("~goto_move_sec", 4.0))
    restore_sec = float(rospy.get_param("~restore_move_sec", 4.0))
    move_sec = float(rospy.get_param("~joint_move_sec", 0.9))
    open_claw = bool(rospy.get_param("~open_left_claw", True))
    restore_on_exit = bool(rospy.get_param("~restore_on_exit", True))

    _EXIT_CTX["restore_sec"] = restore_sec
    _EXIT_CTX["restore_on_exit"] = restore_on_exit

    rospy.Subscriber("/joint_states", JointState, mag.joint_states_callback)
    while not mag.has_joint_states and not rospy.is_shutdown():
        rospy.sleep(0.1)
    mag.last_commanded_joints_rad = np.copy(mag.current_joints_rad)

    try:
        left_deg, src = _resolve_left_deg_from_params()
    except (FileNotFoundError, ValueError) as exc:
        rospy.logerr("%s", exc)
        return 1

    try:
        rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
            changeArmCtrlModeRequest(control_mode=2)
        )
    except Exception:
        pass

    arm_pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    rospy.sleep(0.3)
    _EXIT_CTX["arm_pub"] = arm_pub

    # 脚本启动时构型：退出时左臂回到此处（右臂始终锁定 q_right_hold）
    q_arm_home = np.copy(mag.current_joints_rad)
    q_right_hold = np.copy(mag.current_joints_rad)
    _EXIT_CTX["q_arm_home"] = np.copy(q_arm_home)
    _EXIT_CTX["q_right_hold"] = np.copy(q_right_hold)
    q_target = np.copy(mag.current_joints_rad)
    for i in range(7):
        q_target[i] = math.radians(left_deg[i])

    print("=" * 60)
    print("🔧 左臂关节调平 (无相机 / 无 IK / 右手不动)")
    print("   目标来源:", src)
    print("   left_deg:", ["%.2f" % d for d in left_deg])
    print("=" * 60)

    bu.execute_hold_right(
        arm_pub, q_target, goto_sec, q_right_hold,
        "左臂→保存触顶位 (右臂锁定)",
    )

    q_baseline = np.copy(mag.last_commanded_joints_rad)
    if open_claw:
        _open_left_claw_hold_right(q_right_hold)

    rospy.loginfo("✅ 左臂就位，进入交互调平 (右臂未动)")

    try:
        interactive_tune(arm_pub, q_right_hold, q_baseline, move_sec)
    except KeyboardInterrupt:
        pass
    finally:
        _restore_left_arm_home("q/Ctrl+C")
        try:
            rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
                changeArmCtrlModeRequest(control_mode=0)
            )
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
