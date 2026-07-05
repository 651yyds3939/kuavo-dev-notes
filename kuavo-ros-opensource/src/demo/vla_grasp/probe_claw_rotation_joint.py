#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐个旋转左/右臂关节，观察哪个关节让夹爪「原地转」。

用法（机器人已 standing + WBC 已起）:
  python3 src/demo/vla_grasp/probe_claw_rotation_joint.py
  python3 src/demo/vla_grasp/probe_claw_rotation_joint.py _arm:=left _delta_deg:=20
  python3 src/demo/vla_grasp/probe_claw_rotation_joint.py _arm:=left _joint_slots:=5,6,7
  python3 src/demo/vla_grasp/probe_claw_rotation_joint.py _interactive:=false

关节索引（14 轴 slot）:
  左臂 0-6: zarm_l1 … zarm_l7（6=最靠近左夹爪）
  右臂 7-13: zarm_r1 … zarm_r7（13=最靠近右夹爪）

预期: 拧盖用的「夹爪绕竖直轴自转」一般是 l7/r7（slot 6 / 13）。
      l6/r6 常改俯仰/侧倾；l5 影响腕部整体朝向。
"""
from __future__ import print_function

import math
import sys
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState

try:
    from kuavo_msgs.msg import armTargetPoses
except ImportError:
    print("❌ 无法导入 kuavo_msgs，请先: source devel/setup.bash", file=sys.stderr)
    sys.exit(1)

JOINT_NAMES_14 = [
    "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
    "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
    "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint",
    "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
]

LEFT_SLOTS = list(range(0, 7))
RIGHT_SLOTS = list(range(7, 14))

_current_rad = np.zeros(14)
_has_js = False


def _js_cb(msg):
    global _has_js
    for i, name in enumerate(msg.name):
        if name in JOINT_NAMES_14:
            _current_rad[JOINT_NAMES_14.index(name)] = msg.position[i]
    _has_js = True


def _wait_js(timeout=8.0):
    t0 = time.time()
    while not _has_js and (time.time() - t0) < timeout and not rospy.is_shutdown():
        rospy.sleep(0.05)
    if not _has_js:
        raise RuntimeError("超时：未收到 /joint_states")


def _clamp_elbow_deg(deg_list):
    out = list(deg_list)
    if out[3] > 0.0:
        out[3] = 0.0
    if out[10] > 0.0:
        out[10] = 0.0
    return out


def _move_arm(pub, q_rad, duration_sec, label=""):
    deg = _clamp_elbow_deg([math.degrees(r) for r in q_rad])
    if label:
        rospy.loginfo("▶️ %s (%.1fs)", label, duration_sec)
    pub.publish(armTargetPoses(times=[duration_sec], values=deg))
    rospy.sleep(duration_sec + 0.15)


def _parse_slots(text, arm):
    """'5,6,7' → slot 列表；数字指臂内 1-7，不是 14 轴全局 slot。"""
    base = 0 if arm == "left" else 7
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        local = int(part)
        if local < 1 or local > 7:
            raise ValueError("joint 编号须 1-7， got %d" % local)
        out.append(base + (local - 1))
    return out


def _default_probe_slots(arm):
    """默认只测腕部 l4-l7 / r4-r7（臂内 4-7）。"""
    base = 0 if arm == "left" else 7
    return [base + i for i in range(3, 7)]  # slots 3-6 或 10-13


def main():
    rospy.init_node("probe_claw_rotation_joint", anonymous=True)

    arm = rospy.get_param("~arm", "left").lower()
    if arm not in ("left", "right"):
        rospy.logerr("~arm 须 left 或 right")
        return 1

    delta_deg = float(rospy.get_param("~delta_deg", 25.0))
    move_sec = float(rospy.get_param("~move_sec", 1.8))
    hold_sec = float(rospy.get_param("~hold_sec", 2.0))
    interactive = bool(rospy.get_param("~interactive", True))
    also_neg = bool(rospy.get_param("~also_neg", False))
    slots_param = rospy.get_param("~joint_slots", "")

    if slots_param:
        probe_slots = _parse_slots(str(slots_param), arm)
    else:
        probe_slots = _default_probe_slots(arm)

    arm_label = "左" if arm == "left" else "右"
    base_slot = 0 if arm == "left" else 7

    print("=" * 60)
    print("🔄 夹爪旋转关节探测 | %s臂" % arm_label)
    print("   每关节 +%.1f° → 保持 %.1fs → 恢复" % (delta_deg, hold_sec))
    print("   待测 slot: %s" % ", ".join(
        "%d(%s)" % (s, JOINT_NAMES_14[s].replace("_joint", "")) for s in probe_slots
    ))
    print("   interactive=%s  |  Ctrl+C 中止" % interactive)
    print("=" * 60)

    rospy.Subscriber("/joint_states", JointState, _js_cb, queue_size=1)
    pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    rospy.sleep(0.3)
    _wait_js()

    home = np.copy(_current_rad)
    if arm == "left":
        rospy.loginfo(
            "📍 当前左臂腕部 (deg): l5=%.1f l6=%.1f l7=%.1f",
            math.degrees(home[4]), math.degrees(home[5]), math.degrees(home[6]),
        )
    else:
        rospy.loginfo(
            "📍 当前右臂腕部 (deg): r5=%.1f r6=%.1f r7=%.1f",
            math.degrees(home[11]), math.degrees(home[12]), math.degrees(home[13]),
        )

    def run_one(slot, sign):
        local_num = slot - base_slot + 1
        jname = JOINT_NAMES_14[slot]
        delta_rad = math.radians(sign * delta_deg)
        tag = "%s臂 关节%d %s %s%.1f°" % (
            arm_label, local_num, jname,
            "+" if sign > 0 else "", sign * delta_deg,
        )

        if interactive:
            try:
                raw = input("\n>>> 按 Enter 测试 [%s]，输入 s 跳过: " % tag)
            except EOFError:
                raw = ""
            if raw.strip().lower() == "s":
                rospy.loginfo("⏭️ 跳过 %s", tag)
                return

        q_target = np.copy(home)
        q_target[slot] += delta_rad
        _move_arm(pub, q_target, move_sec, tag)
        rospy.loginfo("👀 请看夹爪是否「原地绕轴转」… 保持 %.1fs", hold_sec)
        rospy.sleep(hold_sec)
        _move_arm(pub, home, move_sec, "恢复 %s" % jname)
        rospy.sleep(0.3)

    try:
        for slot in probe_slots:
            run_one(slot, +1)
            if also_neg:
                run_one(slot, -1)

        _move_arm(pub, home, move_sec, "最终确认归位")
        print("\n✅ 探测完成。请记录：哪个关节让夹爪原地旋转（拧盖轴）。")
        print("   参考: zarm_l7 / zarm_r7（臂内第7关节，14轴 slot 6 / 13）")
    except KeyboardInterrupt:
        rospy.logwarn("⚠️ 中断，尝试恢复初始姿态 …")
        _move_arm(pub, home, move_sec, "中断恢复")
    return 0


if __name__ == "__main__":
    sys.exit(main())
