#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
左臂单独收回，右臂保持当前 /joint_states 不动。

  python3 src/demo/vla_grasp/left_arm_retract_now.py
  python3 src/demo/vla_grasp/left_arm_retract_now.py _mode:=init   # 左臂 init [20,0,0,-30,0,0,0]
  python3 src/demo/vla_grasp/left_arm_retract_now.py _move_sec:=3.5
"""
from __future__ import print_function

import math
import sys
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState

_SCRIPT_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import moveit_auto_grasp as mag
from kuavo_msgs.msg import armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

# auto_grasp_TF2 曲肘护胸 — 左臂 7 轴 (度)
LEFT_READY_DEG = [40.0, 20.0, 0.0, -120.0, 0.0, 0.0, -20.0]
LEFT_INIT_DEG = [20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0]


def main():
    rospy.init_node("left_arm_retract_now", anonymous=True)
    mode = rospy.get_param("~mode", "ready").lower()
    move_sec = float(rospy.get_param("~move_sec", 3.5))

    if mode == "init":
        left_target_deg = LEFT_INIT_DEG
        label = "init"
    else:
        left_target_deg = LEFT_READY_DEG
        label = "曲肘护胸"

    rospy.Subscriber("/joint_states", JointState, mag.joint_states_callback)
    t0 = time.time()
    while not mag.has_joint_states and (time.time() - t0) < 8.0 and not rospy.is_shutdown():
        rospy.sleep(0.05)
    if not mag.has_joint_states:
        print("❌ 无 /joint_states")
        return 1

    try:
        rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
            changeArmCtrlModeRequest(control_mode=2)
        )
    except Exception:
        pass

    pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    rospy.sleep(0.3)

    q = np.copy(mag.current_joints_rad)
    right_deg = [math.degrees(q[i]) for i in range(7, 14)]
    for i, d in enumerate(left_target_deg):
        q[i] = math.radians(d)

    print("=" * 50)
    print("⬅️ 左臂 → %s (%.1fs)，右臂锁定不动" % (label, move_sec))
    print("   左臂目标 (deg):", left_target_deg)
    print("   右臂保持 (deg):", ["%.1f" % d for d in right_deg])
    print("=" * 50)

    target_deg = mag._clamp_elbow_deg([math.degrees(r) for r in q])
    pub.publish(armTargetPoses(times=[move_sec], values=target_deg))
    mag.last_commanded_joints_rad = np.copy(q)
    rospy.sleep(move_sec + 0.2)
    print("✅ 已下发左臂收回指令")
    return 0


if __name__ == "__main__":
    sys.exit(main())
