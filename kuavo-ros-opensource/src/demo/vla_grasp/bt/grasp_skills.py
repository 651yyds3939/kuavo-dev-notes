#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取技能库：规划与分步执行。"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import rospy

from kuavo_msgs.msg import armTargetPoses, ikSolveParam, twoArmHandPoseCmd
from kuavo_msgs.srv import (
    changeArmCtrlMode,
    changeArmCtrlModeRequest,
    twoArmHandPoseCmdSrv,
)

import sys
import os

_VLA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VLA_DIR not in sys.path:
    sys.path.insert(0, _VLA_DIR)
from claw_safe import (
    SafeClawController,
    build_close_cmd,
    build_open_cmd,
)

from . import config

SAFE_LOCKED_Z = 0.385
LIFT_HEIGHT = 0.22
LIFT_HEIGHT_FALLBACKS_M = (0.22, 0.18, 0.14, 0.10)  # 抬升 IK 无解时递减 4cm
CLAW_ROLL_RIGHT = 1.5708
CLAW_ROLL_LEFT = -1.5708

# 夹爪安全参数见 ../claw_safe.py；调试：debug_left_claw.py --safe-lock-test

INIT_ANGLES_DEG = [20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
                   20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0]


class Quaternion:
    def __init__(self):
        self.w = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


def euler_to_rotation_matrix(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    r_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    r_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    r_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return r_yaw @ r_pitch @ r_roll


def rotation_matrix_to_quaternion(r_mat):
    trace = np.trace(r_mat)
    q = Quaternion()
    if trace > 0:
        q.w = math.sqrt(trace + 1.0) / 2
        q.x = (r_mat[2, 1] - r_mat[1, 2]) / (4 * q.w)
        q.y = (r_mat[0, 2] - r_mat[2, 0]) / (4 * q.w)
        q.z = (r_mat[1, 0] - r_mat[0, 1]) / (4 * q.w)
    else:
        i = np.argmax([r_mat[0, 0], r_mat[1, 1], r_mat[2, 2]])
        j, k = (i + 1) % 3, (i + 2) % 3
        t = np.zeros(4)
        t[i] = math.sqrt(r_mat[i, i] - r_mat[j, j] - r_mat[k, k] + 1) / 2
        t[j] = (r_mat[i, j] + r_mat[j, i]) / (4 * t[i])
        t[k] = (r_mat[k, i] + r_mat[i, k]) / (4 * t[i])
        t[3] = (r_mat[k, j] - r_mat[j, k]) / (4 * t[i])
        q.x, q.y, q.z, q.w = t
    norm = math.sqrt(q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2)
    if norm > 0:
        q.w /= norm
        q.x /= norm
        q.y /= norm
        q.z /= norm
    return q


def get_horizontal_claw_quat(target_x, target_y, is_left_arm):
    robot_zero_x = -0.017
    robot_zero_y = 0.292 if is_left_arm else -0.292
    yaw = math.atan2((target_y - robot_zero_y), (target_x - robot_zero_x))
    r_base = euler_to_rotation_matrix(yaw, -1.57079633, 0)
    roll_angle = CLAW_ROLL_LEFT if is_left_arm else CLAW_ROLL_RIGHT
    cr, sr = math.cos(roll_angle), math.sin(roll_angle)
    r_local_z = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rotation_matrix_to_quaternion(r_base @ r_local_z)


@dataclass
class GraspPlan:
    is_left_arm: bool = False
    locked_x: float = 0.0
    locked_y: float = 0.0
    locked_z: float = SAFE_LOCKED_Z
    q_pre: Optional[List[float]] = None
    q_grasp: Optional[List[float]] = None
    q_lift: Optional[List[float]] = None
    q_high_safe: Optional[List[float]] = None
    ready_angles_deg: List[float] = field(default_factory=list)


class GraspSkills:
    """封装 ROS 接口与抓取规划/分步执行。"""

    def __init__(self):
        rospy.wait_for_service("/arm_traj_change_mode")
        rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
            changeArmCtrlModeRequest(control_mode=2)
        )
        self.arm_pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
        rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv")
        self.ik_client = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)
        self._claw = SafeClawController(load_ros_params=True)
        side = rospy.get_param("~claw_verify_on_init", "left")
        if side in ("left", "right", "both"):
            sides = ["left", "right"] if side == "both" else [side]
            for s in sides:
                if not self._claw.verify_side_responsive(s):
                    rospy.logwarn("claw verify failed for %s — check launch 标定 / 退轴", s)

    def call_leju_claw(self, pos, vel, effort, tag="cmd"):
        return self._claw.call(pos, vel, effort, tag=tag)

    def solve_ik(self, ik_req, step_name):
        rospy.loginfo("IK: %s", step_name)
        res = self.ik_client(ik_req)
        if not res.success:
            rospy.logerr("IK failed: %s", step_name)
            return False, None
        return True, list(res.q_arm)

    def execute_pose(self, q_arm, time_sec):
        msg = armTargetPoses(times=[time_sec], values=[math.degrees(q) for q in q_arm])
        self.arm_pub.publish(msg)
        time.sleep(time_sec + 0.5)

    def plan_grasp(self, raw_x, raw_y) -> Optional[GraspPlan]:
        plan = GraspPlan()
        plan.is_left_arm = raw_y > 0.0
        plan.locked_x = raw_x + config.TCP_OFFSET_X
        plan.locked_y = raw_y + (
            config.TCP_OFFSET_Y_LEFT if plan.is_left_arm else config.TCP_OFFSET_Y_RIGHT
        )
        plan.locked_z = SAFE_LOCKED_Z
        plan.ready_angles_deg = (
            [40, 20, 0, -120, 0, 0, -20, 20, 0, 0, -30, 0, 0, 0]
            if plan.is_left_arm
            else [20, 0, 0, -30, 0, 0, 0, 40, -20, 0, -120, 0, 0, -20]
        )

        ik_req = twoArmHandPoseCmd()
        ik_req.use_custom_ik_param = True
        ik_req.joint_angles_as_q0 = True
        ik_param = ikSolveParam()
        ik_param.major_optimality_tol = 1e-3
        ik_param.major_feasibility_tol = 1e-3
        ik_param.minor_feasibility_tol = 1e-3
        ik_param.major_iterations_limit = 500
        ik_param.pos_cost_weight = 0.0
        ik_req.ik_param = ik_param

        quat = get_horizontal_claw_quat(plan.locked_x, plan.locked_y, plan.is_left_arm)
        quat_array = [quat.x, quat.y, quat.z, quat.w]
        seed = [0.0, 0.0, 0.0, -1.57079633, 0.0, 0.0, 0.0]
        ik_req.hand_poses.left_pose.joint_angles = seed
        ik_req.hand_poses.right_pose.joint_angles = seed

        if plan.is_left_arm:
            ik_req.hand_poses.right_pose.pos_xyz = [-0.012, -0.225, -0.265]
            ik_req.hand_poses.right_pose.quat_xyzw = [0.0, 0.0, 0.0, 1.0]
            ik_req.hand_poses.left_pose.quat_xyzw = quat_array
            ik_req.hand_poses.left_pose.pos_xyz = [plan.locked_x, plan.locked_y, plan.locked_z]
        else:
            ik_req.hand_poses.left_pose.pos_xyz = [-0.012, 0.225, -0.265]
            ik_req.hand_poses.left_pose.quat_xyzw = [0.0, 0.0, 0.0, 1.0]
            ik_req.hand_poses.right_pose.quat_xyzw = quat_array
            ik_req.hand_poses.right_pose.pos_xyz = [plan.locked_x, plan.locked_y, plan.locked_z]

        ok, q_grasp = self.solve_ik(ik_req, "step B grasp")
        if not ok:
            return None
        plan.q_grasp = q_grasp

        ik_req.hand_poses.left_pose.joint_angles = list(q_grasp[:7])
        ik_req.hand_poses.right_pose.joint_angles = list(q_grasp[7:])
        robot_zero_x = -0.017
        robot_zero_y = 0.292 if plan.is_left_arm else -0.292
        dist = math.hypot(plan.locked_x - robot_zero_x, plan.locked_y - robot_zero_y)
        ratio = (dist - 0.12) / dist if dist > 1e-6 else 1.0
        pre_x = robot_zero_x + (plan.locked_x - robot_zero_x) * ratio
        pre_y = robot_zero_y + (plan.locked_y - robot_zero_y) * ratio
        if plan.is_left_arm:
            ik_req.hand_poses.left_pose.pos_xyz = [pre_x, pre_y, plan.locked_z]
        else:
            ik_req.hand_poses.right_pose.pos_xyz = [pre_x, pre_y, plan.locked_z]
        ok, q_pre = self.solve_ik(ik_req, "step A pre-aim")
        plan.q_pre = q_pre if ok else q_grasp

        ik_req.hand_poses.left_pose.joint_angles = list(q_grasp[:7])
        ik_req.hand_poses.right_pose.joint_angles = list(q_grasp[7:])
        q_lift = q_grasp
        for h in LIFT_HEIGHT_FALLBACKS_M:
            if plan.is_left_arm:
                ik_req.hand_poses.left_pose.pos_xyz = [plan.locked_x, plan.locked_y, plan.locked_z + h]
            else:
                ik_req.hand_poses.right_pose.pos_xyz = [plan.locked_x, plan.locked_y, plan.locked_z + h]
            ok, q_try = self.solve_ik(ik_req, f"step D lift {int(round(h * 100))}cm")
            if ok:
                q_lift = q_try
                if h < LIFT_HEIGHT - 1e-6:
                    rospy.logwarn(
                        "⚠️ 抬升 %dcm IK 无解，降级为 %dcm",
                        int(round(LIFT_HEIGHT * 100)),
                        int(round(h * 100)),
                    )
                break
        else:
            rospy.logwarn("⚠️ 全部抬升高度 IK 失败，回退使用抓握点（无垂直拔高）")
        plan.q_lift = q_lift

        q_high_safe = list(plan.q_lift)
        if plan.is_left_arm:
            q_high_safe[1] += math.radians(75.0)
        else:
            q_high_safe[8] -= math.radians(75.0)
        plan.q_high_safe = q_high_safe
        return plan

    # --- 分步执行（供 BT 叶子节点调用）---

    def step_open_claw(self):
        pos, vel, effort = build_open_cmd()
        self.call_leju_claw(pos, vel, effort, tag="open")

    def step_move_init(self):
        self.execute_pose([math.radians(x) for x in INIT_ANGLES_DEG], 2.5)

    def step_move_ready(self, plan: GraspPlan):
        self.execute_pose([math.radians(x) for x in plan.ready_angles_deg], 2.5)

    def step_move_pre(self, plan: GraspPlan):
        self.execute_pose(plan.q_pre, 2.5)

    def step_move_grasp(self, plan: GraspPlan):
        self.execute_pose(plan.q_grasp, 1.5)

    def step_close_claw(self, plan: GraspPlan):
        pos, vel, effort = build_close_cmd(plan.is_left_arm)
        tag = "close-left" if plan.is_left_arm else "close-right"
        ok = self.call_leju_claw(pos, vel, effort, tag=tag)
        if not ok:
            rospy.logwarn("close claw aborted (stall?) — skip lift or release manually")

    def step_lift(self, plan: GraspPlan):
        self.execute_pose(plan.q_lift, 2.0)

    def step_retreat(self, plan: GraspPlan):
        self.execute_pose(plan.q_high_safe, 2.0)

    def step_fold_back(self, plan: GraspPlan):
        self.execute_pose([math.radians(x) for x in plan.ready_angles_deg], 3.0)

    def step_move_home(self):
        self.execute_pose([math.radians(x) for x in INIT_ANGLES_DEG], 3.0)

    def step_release_claw(self):
        pos, vel, effort = build_open_cmd()
        self.call_leju_claw(pos, vel, effort, tag="release")

    def execute_grasp_plan(self, plan: GraspPlan):
        """整段执行（调试用，BT 走分步节点）。"""
        self.step_open_claw()
        self.step_move_init()
        self.step_move_ready(plan)
        self.step_move_pre(plan)
        self.step_move_grasp(plan)
        self.step_close_claw(plan)
        self.step_lift(plan)
        self.step_retreat(plan)
        self.step_fold_back(plan)
        self.step_move_home()
        self.step_release_claw()
