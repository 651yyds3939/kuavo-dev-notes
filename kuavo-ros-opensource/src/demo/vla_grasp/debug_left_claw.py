#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leju 夹爪扫参诊断 — 判断某侧是否真正响应指令。

用法（NUC root + source devel/setup.bash，WBC 已启动）：
  python3 debug_left_claw.py              # 扫左爪
  python3 debug_left_claw.py --side right
  python3 debug_left_claw.py --side both --pause 5
  python3 debug_left_claw.py --safe-lock-test --side left   # 保守：70/80 闭合
  python3 debug_left_claw.py --stall-limit-probe --side left --yes
      # 危险：pos=100 完全闭合，逐步增电流找自锁临界点

判定：
  - 正常：不同 cmd 后 feedback pos 会变化，肉眼夹爪会动
  - 故障：pos 始终不变（如左爪一直 0.0）→ 硬件/驱动问题，改程序无效
  - --safe-lock-test：70/80 保守闭合后能 reopen → 可试抓取（仍看 launch 标定）
  - --stall-limit-probe：找 last_safe_effort；触发自锁后立即停止
"""

import argparse
import os
import sys
import time

import rospy
from kuavo_msgs.msg import lejuClawState
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from claw_safe import (  # noqa: E402
    CLAW_EFFORT_OPEN,
    CLAW_OPEN,
    CLAW_VEL,
    MAX_CLOSE_EFFORT,
    SafeClawController,
    format_claw_state,
    sanitize_cmd,
)

HOLD_OPEN = CLAW_OPEN
SWEEP = [0.0, 10.0, 50.0, 80.0, 85.0]
STATE_NAME = {-1: "Error", 0: "Unknown", 1: "Moving", 2: "Reached", 3: "Grabbed"}


def call_claw(left, right, effort=None):
    if effort is None:
        effort = list(CLAW_EFFORT_OPEN)
    pos, vel, eff = sanitize_cmd([left, right], CLAW_VEL, effort)
    req = controlLejuClawRequest()
    req.data.name = ["left_claw", "right_claw"]
    req.data.position = pos
    req.data.velocity = vel
    req.data.effort = eff
    rospy.wait_for_service("/control_robot_leju_claw")
    return rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)(req), pos, eff


def sweep_side(side, pause, latest):
    idx = 0 if side == "left" else 1
    print(f"\n===== sweep {side} claw (other held at {HOLD_OPEN}) =====")
    prev_pos = None
    stuck = True
    for val in SWEEP:
        effort = list(CLAW_EFFORT_OPEN)
        if val > 50:
            effort[idx] = MAX_CLOSE_EFFORT
        pos = [val, HOLD_OPEN] if side == "left" else [HOLD_OPEN, val]
        res, sent_pos, sent_eff = call_claw(*pos, effort=effort)
        time.sleep(pause)
        msg = latest[0]
        print(f"cmd={sent_pos} effort={sent_eff} ok={res.success} msg={res.message!r}")
        print(f"  -> {format_claw_state(msg)}")
        if msg is not None:
            cur = list(msg.data.position)[idx]
            if prev_pos is not None and abs(cur - prev_pos) > 0.5:
                stuck = False
            prev_pos = cur
    if stuck:
        print(f"\n*** {side} claw looks DEAD: position did not change across sweep ***")
        print("    -> hardware / driver / locked-rotor protection, NOT a grasp_skills mapping bug")
    else:
        print(f"\n{side} claw responds to commands (sweep uses max close pos 85, effort {MAX_CLOSE_EFFORT})")


def confirm_stall_probe(args):
    if args.yes:
        return True
    print("\n" + "!" * 60)
    print("  STALL LIMIT PROBE — 会完全闭合(pos=100)并逐步增大电流")
    print("  可能导致蜗杆自锁。请先退轴，并准备随时 Ctrl+C 断电。")
    print("!" * 60)
    try:
        ans = input("输入 yes 继续: ").strip().lower()
    except EOFError:
        return False
    return ans == "yes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["left", "right", "both"], default="left")
    parser.add_argument("--pause", type=float, default=3.0)
    parser.add_argument(
        "--safe-lock-test",
        action="store_true",
        help="渐进 open→70→open→80→open，模拟抓取前安全自检",
    )
    parser.add_argument(
        "--stall-limit-probe",
        action="store_true",
        help="完全闭合 pos=100，逐步增电流，找 reopen 失败的临界点",
    )
    parser.add_argument("--close-pos", type=float, default=100.0, help="探针闭合目标 position")
    parser.add_argument("--effort-start", type=float, default=0.5)
    parser.add_argument("--effort-step", type=float, default=0.1)
    parser.add_argument("--effort-max", type=float, default=1.5)
    parser.add_argument("--open-effort", type=float, default=0.8, help="每轮 reopen 用的电流")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过交互确认（stall-limit-probe 专用）",
    )
    args = parser.parse_args()

    if args.stall_limit_probe and args.safe_lock_test:
        parser.error("use either --safe-lock-test or --stall-limit-probe, not both")

    rospy.init_node("debug_left_claw", anonymous=True)
    latest = [None]
    rospy.Subscriber("/leju_claw_state", lejuClawState, lambda m: latest.__setitem__(0, m), queue_size=1)
    time.sleep(0.5)

    ctrl = SafeClawController(load_ros_params=True)
    sides = ["left", "right"] if args.side == "both" else [args.side]

    if args.stall_limit_probe:
        if not confirm_stall_probe(args):
            print("Aborted.")
            return
        for side in sides:
            result = ctrl.run_stall_limit_probe(
                side,
                close_pos=args.close_pos,
                effort_start=args.effort_start,
                effort_step=args.effort_step,
                effort_max=args.effort_max,
                open_effort=args.open_effort,
                pause=args.pause,
            )
            print("\n--- summary (%s) ---" % side)
            print("  last_safe_effort: %s" % result["last_safe_effort"])
            print("  lock_effort:      %s" % result["lock_effort"])
    elif args.safe_lock_test:
        all_ok = True
        for side in sides:
            if not ctrl.run_safe_lock_test(side, pause=args.pause):
                all_ok = False
        print("\nOverall safe-lock-test:", "PASS" if all_ok else "FAIL")
    else:
        for side in sides:
            sweep_side(side, args.pause, latest)

    print("\nAlso watch live:  rostopic echo /leju_claw_state")
    print("Grasp uses claw_safe.py: close_left=80 effort=0.5 (override via ~claw_* params)")


if __name__ == "__main__":
    main()
