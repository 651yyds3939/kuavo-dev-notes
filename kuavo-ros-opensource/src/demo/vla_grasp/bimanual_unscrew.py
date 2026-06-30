#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂协同初版：右手抓瓶身固定，左手从上往下接触瓶盖并旋转拧盖。

设计原则（v1）：
  - 分阶段时序：仅右手先动 → 右手抓稳后左手再动，避免双臂同时前伸。
  - 瓶盖位置不由 YOLO 直接给出，由瓶身检测点 + 几何偏移推算。
  - 左手垂直接近 + 小步下降触顶 + zarm_l7 腕关节旋转；非水平侧夹瓶盖。
  - 全程右手关节角锁定在抓握构型（不复位到 init）。

依赖：move_group.launch、/vla/yolo_target、与 moveit_auto_grasp 相同终端矩阵。
实机首跑务必低速、有人监护、急停就绪。
"""

import math
import os
import sys
import time

import moveit_commander
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import moveit_auto_grasp as mag
from claw_safe import (
    CLAW_EFFORT_OPEN,
    CLAW_OPEN,
    CLAW_VEL,
    build_close_cmd,
    build_open_cmd,
    get_controller,
)

try:
    from kuavo_msgs.msg import armTargetPoses
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
except ImportError:
    rospy.logerr("❌ 无法导入 kuavo_msgs，请 source devel/setup.bash")
    sys.exit(1)

LEFT_SLICE = slice(0, 7)
RIGHT_SLICE = slice(7, 14)
LEFT_WRIST_IDX = 6

# 默认几何参数（可通过 ROS param ~ 覆盖，见 load_params()）
DEFAULTS = {
    "bottle_cap_rise_m": 0.12,       # 抓握高度到瓶盖中心的竖直距离
    "cap_offset_x_m": 0.0,           # 瓶盖相对瓶身检测点 X 偏置
    "cap_offset_y_m": 0.0,           # 瓶盖相对瓶身检测点 Y 偏置
    "cap_hover_m": 0.035,            # 接触前悬停高度（相对瓶盖中心）
    "contact_step_m": 0.004,         # 触顶搜索每步下降量
    "contact_max_steps": 10,         # 最多下降步数
    "contact_effort_threshold": 0.55,  # 左爪触顶 effort 阈值
    "twist_deg_per_step": 8.0,       # 每步腕旋转角度
    "twist_steps": 15,               # 拧盖步数（约 120°）
    "left_cap_close_pos": 45.0,      # 拧盖时左爪闭合度（0=开 100=关）
    "left_cap_effort": 0.28,         # 拧盖低 effort，防自锁
    "right_hold_after_grasp": True,  # 抓后是否微抬 5cm 给左手腾空间
    "right_micro_lift_m": 0.05,
    # 工作空间安全：瓶太远/太偏时拒绝执行（防质心前倾，见 34.two_arm_coordination.md §4.4）
    "bottle_x_min_m": 0.30,        # 与 YOLO 采点下限一致
    "bottle_x_max_m": 0.55,        # 比单臂采点 0.65 更严；双臂阶段 B 后质心风险高
    "bottle_y_max_m": 0.05,        # 本任务右手抓瓶，Y>此值拒绝（偏左）
}


def load_params():
    p = {}
    for key, default in DEFAULTS.items():
        p[key] = rospy.get_param("~" + key, default)
    return p


def get_topdown_left_quat(target_x, target_y):
    """左手从上往下：yaw 对准目标，pitch 向下，保留左爪 roll。"""
    robot_zero_x = -0.017
    robot_zero_y = 0.292
    yaw = math.atan2((target_y - robot_zero_y), (target_x - robot_zero_x))
    r_base = mag.euler_to_rotation_matrix(yaw, -1.57079633, 0)
    cr = math.cos(mag.CLAW_ROLL_LEFT)
    sr = math.sin(mag.CLAW_ROLL_LEFT)
    r_local = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return mag.rotation_matrix_to_quaternion(r_base @ r_local)


def validate_bottle_workspace(bottle_x, bottle_y, params):
    """
    检查瓶身目标是否在双臂协同安全工作区内。
    返回 (ok, reason)；失败时 reason 为中文说明。
    """
    x_min = params["bottle_x_min_m"]
    x_max = params["bottle_x_max_m"]
    y_max = params["bottle_y_max_m"]
    if bottle_x < x_min:
        return False, "瓶 X=%.3f 过近 (< %.2fm)，请把瓶稍放远" % (bottle_x, x_min)
    if bottle_x > x_max:
        return False, (
            "瓶 X=%.3f 过远 (> %.2fm)，双臂前伸易质心前倾；请把瓶移近或调大 ~bottle_x_max_m"
            % (bottle_x, x_max)
        )
    if bottle_y > y_max:
        return False, (
            "瓶 Y=%.3f 偏左 (> %.2fm)；本任务为右手抓瓶，请放机器人右侧 (Y<0)"
            % (bottle_y, y_max)
        )
    return True, ""


def compute_cap_target(bottle_x, bottle_y, bottle_z, params):
    """由 YOLO 瓶身检测点推算瓶盖中心（base_link）。"""
    cap_x = bottle_x + params["cap_offset_x_m"]
    cap_y = bottle_y + params["cap_offset_y_m"]
    cap_z = bottle_z + params["bottle_cap_rise_m"]
    return cap_x, cap_y, cap_z


def execute_hold_right(arm_pub, q_14_rad, time_sec, q_right_hold, step_name=""):
    """下发 14 轴轨迹，右手强制保持在 q_right_hold。"""
    q = np.copy(q_14_rad)
    q[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
    if step_name:
        rospy.loginfo("▶️ %s (%.1fs)", step_name, time_sec)
    target_deg = mag._clamp_elbow_deg([math.degrees(r) for r in q])
    arm_pub.publish(armTargetPoses(times=[time_sec], values=target_deg))
    mag.last_commanded_joints_rad = np.copy(q)
    time.sleep(time_sec + 0.5)


def solve_left_ik_holding_right(ik_client, pose_stamped, seed_14, q_right_hold, step_name):
    """
    左手 IK；seed 与结果中右手始终为抓握构型（不用 _freeze_inactive_arm）。
    """
    seed = np.copy(seed_14)
    seed[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
    group_name, ee_link = mag._ik_group_profile(True)
    rospy.loginfo("⏳ IK: %s ...", step_name)

    for link in mag._ee_link_candidates(ee_link):
        req = GetPositionIKRequest()
        req.ik_request.group_name = group_name
        req.ik_request.ik_link_name = link
        ps = PoseStamped()
        ps.header.frame_id = pose_stamped.header.frame_id
        ps.header.stamp = rospy.Time(0)
        ps.pose = pose_stamped.pose
        req.ik_request.pose_stamped = ps
        req.ik_request.robot_state = mag._build_robot_state_seed(seed)
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = rospy.Duration(0.8)
        try:
            resp = ik_client(req)
        except rospy.ServiceException as exc:
            rospy.logwarn("⚠️ IK 服务异常: %s", exc)
            continue
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            continue
        merged = np.copy(seed)
        for j, name in enumerate(resp.solution.joint_state.name):
            if name in mag.joint_names_14:
                merged[mag.joint_names_14.index(name)] = resp.solution.joint_state.position[j]
        merged[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        rospy.loginfo("✅ %s IK 成功 (%s)", step_name, link)
        return merged

    rospy.logerr("❌ %s IK 无解", step_name)
    return None


def build_left_cap_close_cmd(close_pos, effort):
    pos = [float(close_pos), CLAW_OPEN]
    eff = [float(effort), CLAW_EFFORT_OPEN[1]]
    return pos, list(CLAW_VEL), eff


def detect_left_contact(claw, effort_threshold):
    st = claw.last_state
    if st is None or len(st.data.effort) < 1:
        return False
    return float(st.data.effort[0]) >= effort_threshold


def descend_until_contact(arm_pub, ik_client, cap_x, cap_y, cap_z, quat, seed_14,
                          q_right_hold, params):
    """从悬停高度小步下降，直到左爪 effort 触顶或步数用尽。"""
    step_m = params["contact_step_m"]
    max_steps = int(params["contact_max_steps"])
    threshold = params["contact_effort_threshold"]
    claw = get_controller()

    z = cap_z + params["cap_hover_m"]
    q_curr = seed_14
    for i in range(max_steps):
        hover_pose = mag._build_pose_stamped(cap_x, cap_y, z, quat)
        q_try = solve_left_ik_holding_right(
            ik_client, hover_pose, q_curr, q_right_hold,
            f"[左手] 触顶搜索 {i + 1}/{max_steps} z={z:.3f}",
        )
        if q_try is None:
            z -= step_m
            continue
        execute_hold_right(
            arm_pub, q_try, 0.6, q_right_hold,
            f"左手下降 {i + 1}/{max_steps}",
        )
        q_curr = np.copy(mag.last_commanded_joints_rad)
        time.sleep(0.25)
        if detect_left_contact(claw, threshold):
            rospy.loginfo("✅ 左爪触顶检测成功 (effort >= %.2f)", threshold)
            return q_curr
        z -= step_m

    rospy.logwarn("⚠️ 未检测到触顶，使用最后下降位姿继续（请人工确认）")
    return q_curr


def twist_cap(arm_pub, q_start, q_right_hold, params):
    """仅旋转左腕 zarm_l7，右手保持抓瓶。"""
    deg = params["twist_deg_per_step"]
    steps = int(params["twist_steps"])
    q = np.copy(q_start)
    for i in range(steps):
        q[LEFT_WRIST_IDX] += math.radians(deg)
        q[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        execute_hold_right(
            arm_pub, q, 0.45, q_right_hold,
            f"拧盖 {i + 1}/{steps} (+{deg:.0f}°)",
        )
        time.sleep(0.15)
    return q


def run_right_grasp_hold(left_arm, right_arm, arm_pub, ik_client, x_hist, y_hist, params):
    """
    阶段 A：右手抓瓶（复用 moveit_auto_grasp 流程，不抬升不收手）。
    返回 (success, q_right_hold)。
    """
    is_left_arm = False
    arm = right_arm
    off_x, off_y = mag.tcp_offsets_for_arm(is_left_arm)
    locked_x = float(np.median(x_hist)) + off_x
    locked_y = float(np.median(y_hist)) + off_y
    locked_z = mag.SAFE_LOCKED_Z

    if locked_y > 0.05:
        rospy.logwarn(
            "⚠️ 瓶子 Y=%.3f 偏左；本任务设计为右手抓瓶，建议把瓶放机器人右侧 (Y<0)",
            locked_y,
        )

    quat = mag.get_horizontal_claw_quat(locked_x, locked_y, is_left_arm)
    shoulder_x, shoulder_y = -0.017, -0.292
    dist = math.hypot(locked_x - shoulder_x, locked_y - shoulder_y)
    if dist <= mag.PRE_GRASP_DIST + 0.01:
        rospy.logerr("❌ 目标过近，无法预瞄 12cm")
        return False, None

    ratio = (dist - mag.PRE_GRASP_DIST) / dist
    pre_x = shoulder_x + (locked_x - shoulder_x) * ratio
    pre_y = shoulder_y + (locked_y - shoulder_y) * ratio
    grasp_pose = mag._build_pose_stamped(locked_x, locked_y, locked_z, quat)
    pre_pose = mag._build_pose_stamped(pre_x, pre_y, locked_z, quat)

    _, ee_link = mag._ik_group_profile(is_left_arm)
    try:
        arm.set_end_effector_link(ee_link)
    except Exception:
        pass

    q_grasp = mag._solve_pose_ik(
        ik_client, arm, is_left_arm, grasp_pose,
        mag.last_commanded_joints_rad, "[右手] 抓握点",
    )
    if q_grasp is None:
        return False, None

    q_pre = mag._solve_pose_ik(
        ik_client, arm, is_left_arm, pre_pose, q_grasp, "[右手] 预瞄 12cm",
    )
    if q_pre is None:
        q_pre = q_grasp

    ready_rad = np.radians(mag._auto_grasp_ready_deg(is_left_arm))
    mag.execute_single_pose(arm_pub, ready_rad, 2.5, "右手曲肘护胸", is_left_arm)

    q_pre_exec = mag._solve_pose_ik(
        ik_client, arm, is_left_arm, pre_pose,
        mag.last_commanded_joints_rad, "[右手] 退至预瞄",
    )
    if q_pre_exec is None:
        q_pre_exec = q_pre
    mag.execute_single_pose(arm_pub, q_pre_exec, 2.5, "右手预瞄", is_left_arm)
    mag.execute_single_pose(arm_pub, q_grasp, 1.5, "右手水平插入", is_left_arm)

    rospy.loginfo("✊ 右手闭合抓瓶身...")
    pos, vel, effort = build_close_cmd(is_left_arm)
    mag.call_leju_claw(pos, vel, effort, tag="close-right-body")
    time.sleep(2.0)

    q_hold = np.copy(mag.last_commanded_joints_rad)

    if params["right_hold_after_grasp"]:
        lift_z = locked_z + params["right_micro_lift_m"]
        lift_pose = mag._build_pose_stamped(locked_x, locked_y, lift_z, quat)
        q_lift = mag._solve_pose_ik(
            ik_client, arm, is_left_arm, lift_pose, q_hold, "[右手] 微抬腾空间",
        )
        if q_lift is not None:
            mag.execute_single_pose(arm_pub, q_lift, 1.8, "右手微抬", is_left_arm)
            q_hold = np.copy(mag.last_commanded_joints_rad)

    rospy.loginfo("✅ 阶段 A 完成：右手抓稳，关节已锁定")
    return True, q_hold


def run_left_unscrew(arm_pub, ik_client, bottle_x, bottle_y, bottle_z,
                     q_right_hold, params):
    """阶段 B~D：左手推算瓶盖 → 悬停 → 触顶 → 轻夹 → 旋转。"""
    cap_x, cap_y, cap_z = compute_cap_target(bottle_x, bottle_y, bottle_z, params)
    rospy.loginfo(
        "🎯 瓶盖推算位姿: X=%.3f Y=%.3f Z=%.3f (rise=%.0fcm)",
        cap_x, cap_y, cap_z, params["bottle_cap_rise_m"] * 100,
    )

    quat = get_topdown_left_quat(cap_x, cap_y)
    hover_z = cap_z + params["cap_hover_m"]
    hover_pose = mag._build_pose_stamped(cap_x, cap_y, hover_z, quat)

    seed = np.copy(mag.last_commanded_joints_rad)
    q_hover = solve_left_ik_holding_right(
        ik_client, hover_pose, seed, q_right_hold, "[左手] 瓶盖上方悬停",
    )
    if q_hover is None:
        return False

    execute_hold_right(arm_pub, q_hover, 2.5, q_right_hold, "左手移至瓶盖上方")

    left_ready = np.radians([40, 20, 0, -120, 0, 0, -20, 20, 0, 0, -30, 0, 0, 0])
    execute_hold_right(
        arm_pub,
        _merge_left_ready_keep_right(left_ready, q_right_hold),
        2.0,
        q_right_hold,
        "左手曲肘护胸（接近前）",
    )

    q_hover2 = solve_left_ik_holding_right(
        ik_client, hover_pose, mag.last_commanded_joints_rad, q_right_hold,
        "[左手] 护胸后至悬停",
    )
    if q_hover2 is not None:
        execute_hold_right(arm_pub, q_hover2, 2.0, q_right_hold, "左手再次悬停")

    q_contact = descend_until_contact(
        arm_pub, ik_client, cap_x, cap_y, cap_z, quat,
        mag.last_commanded_joints_rad, q_right_hold, params,
    )

    rospy.loginfo("🖐️ 左爪轻夹瓶盖 (close=%.0f effort=%.2f)...",
                  params["left_cap_close_pos"], params["left_cap_effort"])
    pos, vel, effort = build_left_cap_close_cmd(
        params["left_cap_close_pos"], params["left_cap_effort"],
    )
    get_controller().call(pos, vel, effort, tag="close-left-cap")
    time.sleep(1.5)

    twist_cap(arm_pub, q_contact, q_right_hold, params)
    rospy.loginfo("✅ 阶段 B~D 完成：拧盖动作已执行（是否拧开需目视确认）")
    return True


def _merge_left_ready_keep_right(left_ready_rad, q_right_hold):
    q = np.copy(left_ready_rad)
    q[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
    return q


def safe_abort(arm_pub):
    mag.execute_dual_arm_init_home(arm_pub)
    pos, vel, effort = build_open_cmd()
    mag.call_leju_claw(pos, vel, effort, tag="release-abort")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("bimanual_unscrew")
    params = load_params()

    rospy.Subscriber("/joint_states", JointState, mag.joint_states_callback)
    rospy.loginfo("⏳ 等待 /joint_states ...")
    while not mag.has_joint_states and not rospy.is_shutdown():
        rospy.sleep(0.1)
    mag.last_commanded_joints_rad = np.copy(mag.current_joints_rad)

    try:
        rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
            changeArmCtrlModeRequest(control_mode=2)
        )
    except Exception:
        rospy.logwarn("⚠️ /arm_traj_change_mode 失败，继续尝试...")

    arm_pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    rospy.sleep(0.3)

    print("=" * 60)
    print("🤝 双臂协同 v1：右手抓瓶 + 左手拧盖")
    print("   分阶段执行 | 瓶盖=几何推算 | 实机务必有人监护")
    print("=" * 60)

    mag.call_leju_claw(*build_open_cmd(), tag="open")
    time.sleep(1.0)
    mag.execute_dual_arm_init_home(arm_pub)

    x_hist, y_hist = mag._collect_vision_targets_tf2_style()
    if len(x_hist) < 10:
        rospy.logerr("❌ 视觉采集失败")
        safe_abort(arm_pub)
        return

    off_x, off_y = mag.tcp_offsets_for_arm(False)
    bottle_x = float(np.median(x_hist)) + off_x
    bottle_y = float(np.median(y_hist)) + off_y
    bottle_z = mag.SAFE_LOCKED_Z

    ok_ws, ws_reason = validate_bottle_workspace(bottle_x, bottle_y, params)
    if not ok_ws:
        rospy.logerr("❌ 工作空间检查未通过：%s", ws_reason)
        safe_abort(arm_pub)
        return
    rospy.loginfo(
        "✅ 工作空间 OK: X=%.3f Y=%.3f (允许 X∈[%.2f,%.2f] Y≤%.2f)",
        bottle_x, bottle_y,
        params["bottle_x_min_m"], params["bottle_x_max_m"], params["bottle_y_max_m"],
    )

    rospy.loginfo("✅ 视觉就绪，加载 MoveIt ...")
    left_arm = moveit_commander.MoveGroupCommander("left_arm")
    right_arm = moveit_commander.MoveGroupCommander("right_arm")
    for grp in (left_arm, right_arm):
        grp.set_pose_reference_frame("base_link")

    ik_client = mag._resolve_ik_service()
    if ik_client is None:
        safe_abort(arm_pub)
        return

    try:
        ok, q_right_hold = run_right_grasp_hold(
            left_arm, right_arm, arm_pub, ik_client, x_hist, y_hist, params,
        )
        if not ok or q_right_hold is None:
            safe_abort(arm_pub)
            return

        ok = run_left_unscrew(
            arm_pub, ik_client, bottle_x, bottle_y, bottle_z, q_right_hold, params,
        )
        if not ok:
            safe_abort(arm_pub)
            return

        rospy.loginfo("⬅️ 收手：松爪 + 双臂 init ...")
        mag.call_leju_claw(*build_open_cmd(), tag="release")
        time.sleep(0.8)
        mag.execute_dual_arm_init_home(arm_pub)
        try:
            rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
                changeArmCtrlModeRequest(control_mode=0)
            )
        except Exception:
            pass
        print("🎉 双臂协同流程结束（请确认瓶盖是否已拧松）")

    except Exception as exc:
        rospy.logerr("❌ 异常: %s", exc)
        safe_abort(arm_pub)
        raise


if __name__ == "__main__":
    main()
