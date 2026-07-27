#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂协同初版：右手抓瓶身固定，左手从上往下接触瓶盖并旋转拧盖。

设计原则（v1）：
  - 分阶段时序：仅右手先动 → 右手抓稳后左手再动，避免双臂同时前伸。
  - 瓶盖位置不由 YOLO 直接给出，由瓶身检测点 + 几何偏移推算。
  - 左手垂直接近 + 小步下降触顶；调平 l5(主)/l6(辅)；拧盖仅动 l7(绕 z)。
  - 全程右手关节角锁定在抓握构型（不复位到 init）。

依赖：move_group.launch、/vla/yolo_target、与 moveit_auto_grasp 相同终端矩阵。
实机首跑务必低速、有人监护、急停就绪。
"""

import json
import math
import os
import signal
import sys
import time

import moveit_commander
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from sensor_msgs.msg import JointState

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import moveit_auto_grasp as mag
from claw_safe import (
    CLAW_CLOSE_RIGHT,
    CLAW_EFFORT_CLOSE_RIGHT,
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
# 左臂关节语义（实机）：l5 小臂+末端整体旋转→夹爪水平(主)；l6 俯仰微调(辅)；l7 水平时绕 z 拧盖
LEFT_L5_LEVEL_IDX = 4   # zarm_l5
LEFT_L6_LEVEL_IDX = 5   # zarm_l6
LEFT_L7_TWIST_IDX = 6   # zarm_l7，仅 twist_cap
LEFT_WRIST_IDX = LEFT_L7_TWIST_IDX  # 兼容旧名

# Ctrl+C / 异常时紧急收手上下文（main 写入）
_EMERGENCY_CTX = {
    "arm_pub": None,
    "q_right_hold": None,
    "q_left": None,
    "armed": False,
    "done": False,
}

# 默认几何参数（可通过 ROS param ~ 覆盖，见 load_params()）
DEFAULTS = {
    "bottle_cap_rise_m": 0.12,       # 抓握高度到瓶盖中心的竖直距离
    # ── HSV 瓶盖检测（上位机 Orin 发布 /vla/cap_target，替代几何推算）──
    "use_hsv_cap": True,             # 优先使用 /vla/cap_target，无数据时回退几何
    "cap_offset_x_m": 0.0,           # 瓶盖相对视觉点 X 偏置（仅几何推算模式）
    "cap_offset_y_m": 0.0,           # 瓶盖相对视觉点 Y 偏置
    # 瓶盖 XY 基准：vision=仅视觉 | right_grasp=右手抓点+偏移（推荐）| blend=融合
    "cap_reference_mode": "right_grasp",
    "cap_blend_grasp_weight": 0.65,
    "cap_from_grasp_x_m": 0.022,
    "cap_from_grasp_y_m": -0.050,
    # ── 左手落点微调 left_tcp_extra_*（叠加 MoveIt 左手 TCP 后作用于 IK 位置）──
    # base_link：X 前 / Y 左 / Z 上。每次 ±0.002m（2mm）试一档。
    #
    # 【重要】approach_pitch=-π/2 时，姿态 yaw=atan2(cap_y, cap_x)。
    # 若 extra 同时参与 yaw 计算，改 extra_x 会「转臂+平移」耦合，目视像没效果。
    # left_cap_yaw_decouple_extra=true（默认）时：extra 只平移 IK 位置，不改变 yaw。
    # 验证是否生效：看日志「瓶盖 X 链」里 IK_x 是否随 extra_x 变化（±0.1 应差 20cm）。
    #
    #   现象              调参方向                    示例
    #   偏前              减小 left_tcp_extra_x_m     _left_tcp_extra_x_m:=-0.060
    #   偏后              增大 left_tcp_extra_x_m     _left_tcp_extra_x_m:=-0.048
    #   偏左              减小 left_tcp_extra_y_m     _left_tcp_extra_y_m:=-0.044
    #   偏右              增大 left_tcp_extra_y_m     _left_tcp_extra_y_m:=-0.032
    #   偏低              增大 left_tcp_extra_z_m     _left_tcp_extra_z_m:=0.004
    #   偏高              减小 left_tcp_extra_z_m     _left_tcp_extra_z_m:=-0.004
    #
    # 启动日志 left_extra=(x,y,z) 与「瓶盖 X 链」一并核对；阶段 C 前只动本三项。
    "left_cap_yaw_decouple_extra": True,  # extra 不参与 yaw，避免 extra_x「调了不动」
    # 实机标定 2026-06：URDF left_gripper_tip + HSV 瓶盖落点补偿。
    # 标定点约 cap=(0.48,-0.06)，大范围换位时后续应升级为 yaw-frame/多点拟合补偿。
    "left_tcp_extra_x_m": -0.030,
    "left_tcp_extra_y_m": -0.040,
    "left_tcp_extra_z_m": 0.00,
    # 左手拧盖：水平夹爪(CLAW_ROLL 同右手) + 垂直接近(基座 pitch 与右手侧夹不同)
    "left_cap_approach_pitch_rad": -1.57079633,  # 与右手侧夹同款；实机直连 IK 稳定
    "left_cap_roll_extra_rad": -0.300,
    "left_cap_pitch_extra_rad": 0.050,
    "left_cap_base_roll_extra_rad": 0.0,   # 调平改 l5 关节偏置；quat roll 易坏 IK
    "left_cap_yaw_offset_rad": 0.0,
    # 仅拧松模式叠加的初始水平偏航：负值使左爪朝右预偏，给逆时针旋转留余量。
    "left_cap_loosen_yaw_offset_rad": -0.35,
    # 拧松专用初始姿态/动态补偿；不影响 tighten 的已标定参数。
    "left_cap_loosen_pitch_extra_rad": 0.050,
    # 拧松初始落点单独下移，不改变 tighten 的 left_tcp_extra_z。
    "left_cap_loosen_tcp_extra_z_m": -0.005,
    # 拧松旋转补偿独立标定：Y=0 取消原先每度向左 1mm；正 Z 抵消实机下沉。
    "left_cap_loosen_y_comp_per_deg_m": 0.0,
    "left_cap_loosen_z_comp_per_deg_m": 0.001,
    "left_cap_loosen_pitch_comp_deg_per_deg": 0.0,
    # 触顶后调平偏置（相对 IK 解；不改 IK 目标，仅触顶停稳后叠加）
    # l5 主：小臂+腕部整体旋转；l6 辅：夹爪俯仰（前高后低 → l6 更负 = 前端下压）
    # l7 不参与调平（水平时 l7 为绕夹爪 z 轴拧盖，见 twist_cap）
    "left_cap_wrist5_bias_rad": 0.0,    # l5 ≈ -14.3°
    "left_cap_wrist6_bias_rad": 0.005,  # l6 实机调平微调
    "left_level_bias_contact_only": True, # 仅触顶/夹爪时加 l5/l6，接近与下降 IK 不加
    # 左手 IK 降级：高位无解时降 Z / 换 pitch（右手侧夹 -π/2）
    "left_ik_high_z_drop_m": [0.0, -0.03, -0.05],
    "left_ik_enable_pitch_fallback": True,
    "left_ik_pitch_fallback_rad": -1.57079633,
    # 右手微抬时瓶盖 Z 跟随比例（1.0=全跟；瓶身软/打滑时实际抬升远小于臂端）
    "cap_z_lift_scale": 0.2,
    "cap_re_vision_after_grasp": False,  # 抓后倾斜时 YOLO 中心漂移，反而破坏瓶盖 XY
    "cap_re_vision_frames": 5,                        
    "cap_xy_refine_enable": False,   # 默认关：25点×2IK≈40s，易误以为死循环
    "cap_xy_refine_step_m": 0.003,   # 精搜步长 3mm
    "cap_xy_refine_half_steps": 1,   # 开启时 3×3=9 点（±3mm）
    "cap_xy_refine_max_sec": 12.0,   # 精搜最长时间 (s)
    "cap_xy_probe_z_m": 0.003,       # 每格下探 3mm 读 effort
    "left_contact_preclose_pos": 35.0,  # 触顶前轻夹，便于 effort 检测
    "left_contact_preclose_effort": 0.22,
    "cap_hover_m": 0.030,            # 悬停高度（略降，减少空中误差）
    "contact_step_m": 0.002,         # 触顶搜索每步 2mm（更细）
    "contact_max_steps": 25,         # 最多下降步数
    "contact_mode": "effort_first",  # effort_first=半夹+effort闭环 | geometry_only=纯几何停
    "contact_preclose_before_descend": True,  # 下降前先半夹，否则 effort 永远无效
    "contact_z_stop_above_m": 0.003,  # 软参考高度 cap_z+此值；effort_first 下无触顶可继续下探
    "contact_z_max_below_cap_m": 0.006,  # 硬安全下限：最多低于推算 cap_z 6mm
    "contact_extra_descend_steps": 6,    # 过软参考高度后再试步数
    "contact_effort_threshold": 0.28,    # 半夹触顶阈值（原0.55过高，开爪时永远无触顶）
    # 轨迹间隔：过长会导致 WBC 在段间下垂抖动（原硬编码 +0.5s）
    "arm_trajectory_post_sleep_sec": 0.12,
    "arm_hold_republish_hz": 20.0,   # 段末按固定关节角重复下发，稳住右手
    # 右手抓瓶 Z：在 SAFE_LOCKED_Z 基础上微调（负=更低，抓瓶身下半部）
    "right_grasp_z_offset_m": -0.03,
    # 拧盖：in_place_ik=固定指尖 XYZ + 姿态绕竖直 z 转；cycle_ik=旧轨道棘轮；joint_l7=旧版仅加 l7
    "left_cap_twist_mode": "in_place_ik",
    # 动作开关：tighten=顺时针拧紧（负角度），loosen=逆时针拧松（正角度）。
    # 拧松当前仅复用拧紧补偿，后续仍需实机单独标定。
    "left_cap_twist_action": "tighten",
    "left_cap_twist_cycle_deg": -15.0,
    "left_cap_twist_cycles": 5,
    "left_cap_twist_step_deg": 0.5,
    "left_cap_twist_step_sec": 0.45,
    "left_cap_twist_cartesian_max_step_deg": 0.25,
    "left_cap_twist_cartesian_deg_per_sec": 8.0,
    "left_cap_twist_fk_xy_tolerance_m": 0.0025,
    "left_cap_twist_fk_z_tolerance_m": 0.0015,
    "left_cap_twist_joint_jump_deg": 8.0,
    # 实机标定的有效夹持旋转中心（zarm_l7_end_effector 局部坐标，m）。
    "left_cap_twist_pivot_ee_m": [0.018, 0.010, -0.0525],
    # 世界坐标平移补偿（m/deg）：随绝对累计转角线性变化，回正时自动归零。
    # base_link 中 +X=前、+Y=左；实机向右漂时使用正的 Y 补偿。
    "left_cap_twist_x_comp_per_deg_m": 0.001,
    "left_cap_twist_y_comp_per_deg_m": 0.001,
    # 与当前有效 TCP 配套的实机动态高度补偿。
    "left_cap_twist_z_comp_per_deg_m": -0.0022,
    "left_cap_twist_pitch_offset_deg": 0.0,
    "left_cap_twist_prelevel_sec": 2.0,
    "left_cap_twist_prelevel_step_deg": 0.25,
    "left_cap_twist_pitch_comp_deg_per_deg": -0.60,
    # 实机夹爪前后俯仰轴；当前安装中 local_x 对应尖端上下，local_y 对应左右侧倾。
    "left_cap_twist_level_axis": "local_x",
    # world_horizontal=将上述局部轴投影到世界水平面并固定，避免随拧盖 yaw 一起转。
    "left_cap_twist_pitch_comp_frame": "world_horizontal",
    "left_cap_twist_use_contact_yaw": True,   # 拧盖沿用触顶 yaw，不额外朝右偏航
    "left_cap_twist_skip_align": True,        # 夹紧后不重对准，直接绕 Z 拧
    "left_cap_twist_fallback_joint_l7": True, # cycle_ik 转不动时回退 l7
    "left_cap_twist_face_right_yaw_rad": -1.57079633,  # 仅 use_contact_yaw=False 时叠加
    "left_cap_twist_reclose": True,        # 每轮松爪、回正并重新夹紧
    "left_cap_twist_release_pos": 35.0,    # 周期末左爪松开
    "left_cap_twist_direct_l7_enable": False, # 固定指尖XYZ走连续IK
    "left_cap_twist_seed_l7_enable": True,    # direct_l7=False时，用预旋l7的seed引导IK补偿偏心
    "twist_deg_per_step": 8.0,            # joint_l7 模式专用
    "twist_steps": 15,
    "left_cap_close_pos": 98.0,      # 拧盖最终闭合（claw_safe max_close_pos 默认 98）
    "left_cap_effort": 1.0,          # 夹盖力矩（接口文档推荐 1~2A）
    "left_cap_close_ramp_enable": False,   # 分步闭合；每步须等 Reached 再发下一条
    "left_cap_close_ramp_step": 8.0,      # 步长 8，减少指令次数
    "left_cap_close_ramp_sleep_sec": 0.0,
    "left_cap_close_wait_sec": 0.25,        # call() 内短轮询
    "left_cap_close_settle_sec": 2.5,       # 每步等左爪 Reached 的最长时间
    "left_cap_close_retries": 4,            # 单步服务失败重试
    "left_cap_close_effort_stop": 0.50,
    "left_cap_close_min_pos": 80.0,
    "left_cap_final_squeeze_enable": False,
    "left_cap_final_squeeze_sec": 1.0,
    "left_cap_final_squeeze_passes": 2,
    # claw_safe 限幅（接口文档 position 0~100、effort 1~2A；比旧版 85/0.6A 略放宽）
    "claw_max_close_pos": 98.0, 
    "claw_max_close_effort": 1.2,
    "claw_stall_effort": 1.5,
    "claw_wait_sec": 0.6,
    "right_hold_after_grasp": False,  # 抓后是否微抬 5cm 给左手腾空间
    "right_micro_lift_m": 0.05,
    # 工作空间安全：瓶太远/太偏时拒绝执行（防质心前倾，见 34.two_arm_coordination.md §4.4）
    "bottle_x_min_m": 0.30,        # 与 YOLO 采点下限一致
    "bottle_x_max_m": 0.55,        # 比单臂采点 0.65 更严；双臂阶段 B 后质心风险高
    "bottle_y_max_m": 0.0,         # 双臂右手抓瓶：YOLO Y 必须 ≤0（正=偏左，必歪瓶）
    "bottle_y_min_m": -0.105,      # Y 过负=瓶太偏右，左手侧向/瓶前高位 IK 易无解
    # 右手 TCP 额外补偿（在 moveit_auto_grasp 分参之上）
    "right_tcp_extra_x_m": 0.015,    # +X 往前
    "right_tcp_extra_y_m": -0.023,   # +Y 往左
    # 左手接近速度：仅曲肘→瓶上方段；触顶下降见 left_descend_step_sec（单独控）
    "left_approach_direct_first": True,
    "left_high_approach_m": 0.05,
    "left_ready_move_sec": 4.4,          # 曲肘护胸
    "left_ik_probe_move_sec": 4.0,       # 直连瓶上方
    "left_ik_classic_move_sec": 4.0,     # classic 绕障
    "left_hover_move_sec": 3.2,          # 降至悬停（不含触顶小步）
    "left_descend_step_sec": 0.55,       # 触顶每步，保持不变
    "left_approach_duration_scale": 1.0, # 仅乘 ready/probe/hover/classic，不乘触顶步
    # 曲肘后先抬升+后退中间路点，避免直扑瓶盖蹭桌
    "left_pre_lift_enable": True,
    "left_pre_lift_extra_z_m": 0.06,      # 在 high 高度之上再抬 (m)
    "left_pre_lift_retreat_m": 0.10,      # 瓶盖 XY 朝左肩 retreat (m)
    "left_pre_lift_move_sec": 3.5,
    "left_lateral_m": 0.07,
    "left_pre_forward_m": 0.09,
    "left_final_forward_m": 0.06,
    # 右手抓握验收（空抓则中止，不进入左手阶段）
    # 阈值放低：瓶子细/角度微偏时 effort 可能只有 0.2-0.3，避免误判
    "right_grasp_min_close_pos": 40.0,
    "right_grasp_min_effort": 0.05,
    "right_grasp_accept_close_pos": 70.0,  # 闭合反馈足够大时，即使effort偏低也认为夹到
    # 触顶成功后保存左臂关节，供 left_cap_level_tune.py 无相机复现
    "save_left_tune_pose": False,
    "left_tune_pose_file": "",  # 空=脚本目录下 left_cap_tune_pose.json
    # 指尖 TCP：MoveIt 末端 zarm_l7_end_effector ≈ 爪中心；未标定前建议关闭
    # mode=world_z：只沿 base_link 竖直补偿，不引入侧倾（推荐首调）
    # mode=ee：EE 系向量，需实机标定 dx/dy/dz
    "left_claw_tip_enable": False,
    # ── FK 自动指尖补偿（需 /compute_fk 服务，MoveIt 启动后可用）
    # 夹爪指尖在 zarm_l7_end_effector 坐标系下的物理偏移 (m)，卡尺量一次永久有效
    "gripper_tip_dx_m": 0.04,    # end_effector 系 X（大致=上下，正=往上）
    "gripper_tip_dy_m": -0.03,   # end_effector 系 Y（大致=左右，正=往左）
    "gripper_tip_dz_m": 0.05,    # end_effector 系 Z（大致=前后，正=往前）
    "left_claw_tip_mode": "world_z",
    "left_claw_tip_world_z_m": 0.012,
    "left_claw_tip_world_z_close_m": 0.015,
    "left_claw_tip_preclose_pos": 35.0,
    "left_claw_tip_close_pos": 62.0,
    "left_claw_tip_ee_preclose_m": [0.0, 0.0, -0.012],
    "left_claw_tip_ee_close_m": [0.0, 0.0, -0.015],
}


LEFT_CAP_TUNE_POSE_BASENAME = "left_cap_tune_pose.json"


def load_params():
    p = {}
    for key, default in DEFAULTS.items():
        p[key] = rospy.get_param("~" + key, default)

    action = str(p.get("left_cap_twist_action", "tighten")).strip().lower()
    if action not in ("tighten", "loosen"):
        rospy.logwarn(
            "⚠️ left_cap_twist_action=%r 无效，使用默认 tighten（可选 tighten/loosen）",
            action,
        )
        action = "tighten"
    p["left_cap_twist_action"] = action
    if action == "loosen":
        loosen_yaw = float(p.get("left_cap_loosen_yaw_offset_rad", -0.35))
        p["left_cap_yaw_offset_rad"] = float(p["left_cap_yaw_offset_rad"]) + loosen_yaw
        p["left_cap_pitch_extra_rad"] = float(
            p.get("left_cap_loosen_pitch_extra_rad", p["left_cap_pitch_extra_rad"])
        )
        p["left_tcp_extra_z_m"] = float(p["left_tcp_extra_z_m"]) + float(
            p.get("left_cap_loosen_tcp_extra_z_m", -0.005)
        )
        p["left_cap_twist_y_comp_per_deg_m"] = float(
            p.get("left_cap_loosen_y_comp_per_deg_m", p["left_cap_twist_y_comp_per_deg_m"])
        )
        p["left_cap_twist_z_comp_per_deg_m"] = float(
            p.get("left_cap_loosen_z_comp_per_deg_m", p["left_cap_twist_z_comp_per_deg_m"])
        )
        p["left_cap_twist_pitch_comp_deg_per_deg"] = float(
            p.get(
                "left_cap_loosen_pitch_comp_deg_per_deg",
                p["left_cap_twist_pitch_comp_deg_per_deg"],
            )
        )
        rospy.loginfo(
            "🔓 拧松专用补偿: yaw=%.1f° pitch_extra=%.3f static_z=%+.1fmm "
            "y_comp=%+.3fmm/deg z_comp=%+.3fmm/deg pitch_comp=%.3f°/deg",
            math.degrees(loosen_yaw), p["left_cap_pitch_extra_rad"],
            float(p.get("left_cap_loosen_tcp_extra_z_m", -0.005)) * 1000.0,
            p["left_cap_twist_y_comp_per_deg_m"] * 1000.0,
            p["left_cap_twist_z_comp_per_deg_m"] * 1000.0,
            p["left_cap_twist_pitch_comp_deg_per_deg"],
        )
    cycle_abs_deg = abs(float(p["left_cap_twist_cycle_deg"]))
    p["left_cap_twist_cycle_deg"] = -cycle_abs_deg if action == "tighten" else cycle_abs_deg
    return p


def apply_claw_safe_limits(params):
    """在首次 get_controller() 前写入 claw_safe 私有参数。"""
    rospy.set_param("~claw_max_close_pos", float(params.get("claw_max_close_pos", 98.0)))
    rospy.set_param("~claw_max_close_effort", float(params.get("claw_max_close_effort", 1.2)))
    rospy.set_param("~claw_stall_effort", float(params.get("claw_stall_effort", 1.5)))
    rospy.set_param("~claw_wait_sec", float(params.get("claw_wait_sec", 0.6)))


def left_tune_pose_path(params=None):
    if params and params.get("left_tune_pose_file"):
        return str(params["left_tune_pose_file"])
    return os.path.join(_SCRIPT_DIR, LEFT_CAP_TUNE_POSE_BASENAME)


def save_left_cap_tune_pose(q_14, q_right_hold, meta, path=None, params=None):
    """保存左臂触顶构型（度+弧度），供无相机调平脚本直接关节空间复现。"""
    out = path or left_tune_pose_path(params)
    payload = {
        "description": "左臂瓶盖触顶 IK 解；left_cap_level_tune.py 只动左臂、右臂保持不动",
        "left_joints_deg": [round(math.degrees(float(q_14[i])), 4) for i in range(7)],
        "left_joints_rad": [round(float(q_14[i]), 6) for i in range(7)],
        "right_joints_deg": [round(math.degrees(float(q_right_hold[i])), 4) for i in range(7, 14)],
        "joint_names_left": list(mag.joint_names_14[:7]),
        "cap_x": meta.get("cap_x"),
        "cap_y": meta.get("cap_y"),
        "cap_z": meta.get("cap_z"),
        "hover_z": meta.get("hover_z"),
    }
    if params:
        payload["left_claw_tip_enable"] = bool(params.get("left_claw_tip_enable", True))
        payload["left_claw_tip_ee_preclose_m"] = list(
            params.get("left_claw_tip_ee_preclose_m", [0, 0, 0])
        )
        payload["left_claw_tip_ee_close_m"] = list(
            params.get("left_claw_tip_ee_close_m", [0, 0, 0])
        )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    rospy.loginfo("💾 已保存左臂触顶关节 → %s", out)
    rospy.loginfo(
        "   left_deg=%s",
        payload["left_joints_deg"],
    )
    return out


def load_left_cap_tune_pose(path=None, params=None):
    p = path or left_tune_pose_path(params)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if "left_joints_deg" not in data and "left_joints_rad" not in data:
        raise ValueError("pose 文件缺少 left_joints_deg / left_joints_rad: %s" % p)
    return data


def _left_move_sec(params, key, default, approach_only=False):
    """左手轨迹段时长。approach_only=True 时乘 left_approach_duration_scale（不用于触顶步）。"""
    sec = float(params.get(key, default))
    if approach_only:
        sec *= float(params.get("left_approach_duration_scale", 1.0))
    return sec


def _left_descend_step_sec(params):
    """触顶小步下降时长，不受 approach 倍率影响。"""
    return float(params.get("left_descend_step_sec", 0.55))


def get_topdown_left_quat(target_x, target_y, roll_extra=0.0, pitch_extra=0.0,
                          base_roll_extra=0.0, yaw_extra=0.0, approach_pitch_override=None):
    """
    左手拧盖：水平夹爪（必须加 CLAW_ROLL_LEFT，与右手 get_horizontal_claw_quat 同款）。

    右手侧夹瓶身：R = Rz(yaw) @ Ry(-π/2) @ R_local_z(CLAW_ROLL_RIGHT)
                  approach_pitch=-π/2 → 手臂水平前伸，夹爪横置夹柱体。

    左手拧盖：    同样 R_local_z(CLAW_ROLL_LEFT) 使夹爪水平；但路点在瓶盖正上方，
                  默认 approach_pitch=0（垂直接近），避免整段用 -π/2 导致爪朝向前方。
                  若要与右手完全同式，设 left_cap_approach_pitch_rad:=-1.5708。
    """
    p = _ACTIVE_LEFT_CAP_PARAMS or {}
    if approach_pitch_override is not None:
        approach_pitch = float(approach_pitch_override)
    else:
        approach_pitch = float(p.get("left_cap_approach_pitch_rad", 0.0))

    no_extras = (
        abs(roll_extra) < 1e-9 and abs(pitch_extra) < 1e-9
        and abs(base_roll_extra) < 1e-9 and abs(yaw_extra) < 1e-9
    )
    # 与右手完全一致（approach=-π/2 且无 extras）时直接复用 moveit 函数
    if no_extras and abs(approach_pitch + 1.57079633) < 1e-5:
        return mag.get_horizontal_claw_quat(target_x, target_y, True)

    robot_zero_x = -0.017
    robot_zero_y = 0.292
    yaw = math.atan2((target_y - robot_zero_y), (target_x - robot_zero_x))
    yaw += float(yaw_extra)
    pitch = approach_pitch + float(pitch_extra)
    roll = float(base_roll_extra)
    R_base = mag.euler_to_rotation_matrix(yaw, pitch, roll)
    claw_roll = mag.CLAW_ROLL_LEFT + float(roll_extra)
    cr, sr = math.cos(claw_roll), math.sin(claw_roll)
    R_local = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return mag.rotation_matrix_to_quaternion(R_base @ R_local)


def _left_cap_orientation_kwargs():
    """从 _ACTIVE_LEFT_CAP_PARAMS 读取姿态微调。"""
    if _ACTIVE_LEFT_CAP_PARAMS is None:
        return {}
    p = _ACTIVE_LEFT_CAP_PARAMS
    return {
        "roll_extra": p.get("left_cap_roll_extra_rad", 0.0),
        "pitch_extra": p.get("left_cap_pitch_extra_rad", 0.0),
        "base_roll_extra": p.get("left_cap_base_roll_extra_rad", 0.0),
        "yaw_extra": p.get("left_cap_yaw_offset_rad", 0.0),
    }


# run_left_unscrew 期间：锁定 IK 成功的 approach_pitch，悬停/触顶必须沿用
_ACTIVE_LEFT_IK_CTX = {"approach_pitch": None, "reach_z": None}


def _lock_left_approach_pitch(pitch, reach_z=None):
    if pitch is not None:
        _ACTIVE_LEFT_IK_CTX["approach_pitch"] = float(pitch)
    if reach_z is not None:
        _ACTIVE_LEFT_IK_CTX["reach_z"] = float(reach_z)


def _reset_left_ik_ctx():
    _ACTIVE_LEFT_IK_CTX["approach_pitch"] = None
    _ACTIVE_LEFT_IK_CTX["reach_z"] = None


def _locked_approach_pitch(params, override=None):
    if override is not None:
        return float(override)
    if _ACTIVE_LEFT_IK_CTX.get("approach_pitch") is not None:
        return float(_ACTIVE_LEFT_IK_CTX["approach_pitch"])
    return float(params["left_cap_approach_pitch_rad"])


# run_left_unscrew 期间供姿态/腕偏置读取
_ACTIVE_LEFT_CAP_PARAMS = None
# IK 位置(含 extra) 与 yaw 参考(不含 extra) 分离，见 left_cap_yaw_decouple_extra
_ACTIVE_LEFT_CAP_XY = {"yaw_x": None, "yaw_y": None}
# main 写入，供轨迹稳持参数
_ACTIVE_ARM_PARAMS = None


def _apply_left_claw_level_bias(q_14, contact_phase=False):
    """
    触顶后夹爪水平调平：只动 l5(主)、l6(辅)。
    l7 不在此处理（水平时 l7 为拧盖绕 z，见 twist_cap）。
    left_level_bias_contact_only=True 时，接近/下降 IK 路径不施加。
    """
    if _ACTIVE_LEFT_CAP_PARAMS is None:
        return q_14
    p = _ACTIVE_LEFT_CAP_PARAMS
    if p.get("left_level_bias_contact_only", True) and not contact_phase:
        return q_14
    w5 = float(p.get("left_cap_wrist5_bias_rad", 0.0))
    w6 = float(p.get("left_cap_wrist6_bias_rad", 0.0))
    if abs(w5) < 1e-9 and abs(w6) < 1e-9:
        return q_14
    q = np.copy(q_14)
    q[LEFT_L5_LEVEL_IDX] += w5
    q[LEFT_L6_LEVEL_IDX] += w6
    return q


def _apply_left_wrist_level_bias(q_14, contact_phase=False):
    """兼容旧调用名。"""
    return _apply_left_claw_level_bias(q_14, contact_phase=contact_phase)


def validate_bottle_workspace(raw_x, raw_y, params):
    """
    用 YOLO/TF2 原始瓶身坐标做工作空间检查（勿加 TCP 补偿）。
    TCP 只修正夹爪落点，不代表瓶子在桌面上的真实位置。
    """
    x_min = params["bottle_x_min_m"]
    x_max = params["bottle_x_max_m"]
    y_max = params["bottle_y_max_m"]
    y_min = params["bottle_y_min_m"]
    if raw_x < x_min:
        return False, "瓶视觉 X=%.3f 过近 (< %.2fm)，请把瓶稍放远" % (raw_x, x_min)
    if raw_x > x_max:
        return False, (
            "瓶视觉 X=%.3f 过远 (> %.2fm)，双臂前伸易质心前倾；请把瓶移近或调大 ~bottle_x_max_m"
            % (raw_x, x_max)
        )
    if raw_y > y_max:
        return False, (
            "瓶视觉 Y=%.3f 偏左 (> %.2fm)；右手侧抓会歪瓶！请放机器人正前偏右 (YOLO Y≤%.2f，建议 Y∈[-0.10,0])"
            % (raw_y, y_max, y_max)
        )
    if raw_y < y_min:
        return False, (
            "瓶视觉 Y=%.3f 偏右 (< %.2fm)；左手拧盖 IK 难达，请往左/中间移 (建议 Y∈[%.2f, %.2f])"
            % (raw_y, y_min, y_min, y_max)
        )
    return True, ""


def right_locked_xy(raw_x, raw_y, params):
    """右手抓瓶 TCP：moveit 分参 + 双臂任务额外 X/Y 补偿。"""
    off_x, off_y = mag.tcp_offsets_for_arm(False)
    return (
        raw_x + off_x + params["right_tcp_extra_x_m"],
        raw_y + off_y + params["right_tcp_extra_y_m"],
    )


def left_cap_locked_xy(cap_x, cap_y, params):
    """左手瓶盖 IK 位置 = geom + moveit_tcp + extra；yaw 参考可选不含 extra。"""
    off_x, off_y = mag.tcp_offsets_for_arm(True)
    ex = float(params["left_tcp_extra_x_m"])
    ey = float(params["left_tcp_extra_y_m"])
    pos_x = cap_x + off_x + ex
    pos_y = cap_y + off_y + ey
    yaw_x = cap_x + off_x
    yaw_y = cap_y + off_y
    return pos_x, pos_y, yaw_x, yaw_y


def _left_cap_yaw_xy(pos_x, pos_y, params):
    """IK 位置与朝向解耦：extra 只平移落点，不通过 atan2 牵动臂展 yaw。"""
    if not params.get("left_cap_yaw_decouple_extra", True):
        return pos_x, pos_y
    yaw_x = _ACTIVE_LEFT_CAP_XY.get("yaw_x")
    yaw_y = _ACTIVE_LEFT_CAP_XY.get("yaw_y")
    if yaw_x is None or yaw_y is None:
        return pos_x, pos_y
    return float(yaw_x), float(yaw_y)


def _quat_rotate_vec(q, v):
    """Hamilton 四元数旋转向量 v（q 为单位四元数）。"""
    v = np.asarray(v, dtype=float)
    qv = np.array([q.x, q.y, q.z], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + q.w * t + np.cross(qv, t)


def _calibrate_ee_offset(cap_x, cap_y, params):
    """从 base_link extra 自动反算 EE 系夹爪偏移（一次性标定）。
    返回 (dx, dy, dz) 在 l7 系下的偏移，或 None。
    """
    extra = np.array([
        float(params.get("left_tcp_extra_x_m", 0.0)),
        float(params.get("left_tcp_extra_y_m", 0.0)),
        float(params.get("left_tcp_extra_z_m", 0.0)),
    ])
    if np.all(np.abs(extra) < 1e-5):
        return None

    # 获取 IK 目标四元数（与 _left_cap_pose 一致）
    kw = _left_cap_orientation_kwargs()
    pitch = _locked_approach_pitch(params)
    yaw_x, yaw_y = _left_cap_yaw_xy(cap_x, cap_y, params)
    quat = get_topdown_left_quat(yaw_x, yaw_y, approach_pitch_override=pitch, **kw)

    # 叠加 l5/l6 关节偏置（触顶后的实际夹爪姿态比 IK 目标多这两项旋转）
    w5 = float(params.get("left_cap_wrist5_bias_rad", 0.0))
    w6 = float(params.get("left_cap_wrist6_bias_rad", 0.0))
    if abs(w5) > 1e-6:
        q5 = mag.Quaternion()
        q5.w, q5.z = math.cos(w5 * 0.5), math.sin(w5 * 0.5)
        quat = _quat_mul(quat, q5)  # 绕局部 Z
    if abs(w6) > 1e-6:
        q6 = mag.Quaternion()
        q6.w, q6.y = math.cos(w6 * 0.5), math.sin(w6 * 0.5)
        quat = _quat_mul(quat, q6)  # 绕局部 Y

    # 共轭四元数 = 逆旋转
    q_conj = mag.Quaternion()
    q_conj.w = quat.w
    q_conj.x = -quat.x
    q_conj.y = -quat.y
    q_conj.z = -quat.z

    # delta_base = R(q) * delta_ee → delta_ee = R(q*) * delta_base
    delta_ee = _quat_rotate_vec(q_conj, extra)
    return delta_ee


def _claw_tip_offset_ee(params, tip_phase="preclose"):
    key = "left_claw_tip_ee_close_m" if tip_phase == "close" else "left_claw_tip_ee_preclose_m"
    return np.array(params.get(key, [0.0, 0.0, 0.0]), dtype=float)


def _claw_tip_offset_ee_for_claw_pos(params, claw_pos):
    """按爪开合角在 preclose/close 两行标定之间线性插值 EE 系指尖偏移。"""
    p0 = float(params.get("left_claw_tip_preclose_pos", 35.0))
    p1 = float(params.get("left_claw_tip_close_pos", 62.0))
    o0 = _claw_tip_offset_ee(params, "preclose")
    o1 = _claw_tip_offset_ee(params, "close")
    if p1 <= p0 + 1e-6:
        return o0
    t = float(np.clip((float(claw_pos) - p0) / (p1 - p0), 0.0, 1.0))
    return o0 + t * (o1 - o0)


def _fingertip_target_to_ee_xyz(finger_x, finger_y, finger_z, quat, params, tip_phase="preclose"):
    """
    期望指尖落点 (finger_*) → MoveIt EE 目标。
    world_z：EE 比指尖高 world_z_m（只改竖直，不引入侧倾）
    ee：p_ee = p_tip - R @ v_ee_to_tip
    """
    if not params.get("left_claw_tip_enable", False):
        return float(finger_x), float(finger_y), float(finger_z)
    mode = str(params.get("left_claw_tip_mode", "world_z")).lower()
    if mode == "world_z":
        dz = float(params.get("left_claw_tip_world_z_m", 0.012))
        if tip_phase == "close":
            dz_close = float(params.get("left_claw_tip_world_z_close_m", dz + 0.003))
            dz = dz_close
        return float(finger_x), float(finger_y), float(finger_z) + dz
    if tip_phase == "close":
        claw_pos = float(params.get("left_cap_close_pos", 62.0))
    else:
        claw_pos = float(params.get("left_contact_preclose_pos", 35.0))
    v = _claw_tip_offset_ee_for_claw_pos(params, claw_pos)

    # 🔧 修正：把 l5/l6 调平偏置乘进四元数，得到实际夹爪朝向
    #     quat 是 IK 目标姿态（不含偏置），实际夹爪多转了 l5_bias + l6_bias
    q_corrected = mag.Quaternion()
    q_corrected.w, q_corrected.x = quat.w, quat.x
    q_corrected.y, q_corrected.z = quat.y, quat.z
    w5 = float(params.get("left_cap_wrist5_bias_rad", 0.0))
    w6 = float(params.get("left_cap_wrist6_bias_rad", 0.0))
    if abs(w5) > 1e-6:
        q5 = mag.Quaternion()
        q5.w, q5.z = math.cos(w5 * 0.5), math.sin(w5 * 0.5)
        q_corrected = _quat_mul(q_corrected, q5)
    if abs(w6) > 1e-6:
        q6 = mag.Quaternion()
        q6.w, q6.y = math.cos(w6 * 0.5), math.sin(w6 * 0.5)
        q_corrected = _quat_mul(q_corrected, q6)

    delta = _quat_rotate_vec(q_corrected, v)
    return (
        float(finger_x) - float(delta[0]),
        float(finger_y) - float(delta[1]),
        float(finger_z) - float(delta[2]),
    )

def _log_key_params(params):
    """启动时打印关键参数，确认命令行覆盖是否生效。"""
    lx, ly = mag.tcp_offsets_for_arm(True)
    rx, ry = mag.tcp_offsets_for_arm(False)
    rospy.loginfo(
        "📋 关键参数: right_extra=(%.3f,%.3f) | cap_mode=%s | cap_z_lift_scale=%.2f | "
        "left_extra=(%.3f,%.3f,%.3f) | cap_claw=水平(CLAW_ROLL_L) approach_pitch=%.3f yaw_offset=%.1fdeg | "
        "level: base_roll=%.3f roll_extra=%.3f l5=%.3f l6=%.3f | "
        "left_move: ready=%.1f direct=%.1f hover=%.1f step=%.2f pre_lift=%s+%.2fm | "
        "arm_post_sleep=%.2fs hold_hz=%.0f | moveit_TCP_L=(%.3f,%.3f) R=(%.3f,%.3f)",
        params["right_tcp_extra_x_m"], params["right_tcp_extra_y_m"],
        params["cap_reference_mode"],
        params["cap_z_lift_scale"],
        params["left_tcp_extra_x_m"], params["left_tcp_extra_y_m"],
        params["left_tcp_extra_z_m"],
        params["left_cap_approach_pitch_rad"],
        math.degrees(float(params.get("left_cap_yaw_offset_rad", 0.0))),
        params["left_cap_base_roll_extra_rad"], params["left_cap_roll_extra_rad"],
        params["left_cap_wrist5_bias_rad"], params["left_cap_wrist6_bias_rad"],
        params["left_ready_move_sec"], params["left_ik_probe_move_sec"],
        params["left_hover_move_sec"], params["left_descend_step_sec"],
        params.get("left_pre_lift_enable", True), params.get("left_pre_lift_extra_z_m", 0.06),
        params["arm_trajectory_post_sleep_sec"], params["arm_hold_republish_hz"],
        lx, ly, rx, ry,
    )
    if params.get("left_claw_tip_enable", False):
        mode = str(params.get("left_claw_tip_mode", "world_z"))
        if mode == "world_z":
            rospy.loginfo(
                "📋 指尖TCP: enable mode=world_z | preclose_z=+%.4fm close_z=+%.4fm",
                float(params.get("left_claw_tip_world_z_m", 0.012)),
                float(params.get("left_claw_tip_world_z_close_m",
                                  float(params.get("left_claw_tip_world_z_m", 0.012)) + 0.003)),
            )
        else:
            pre = params.get("left_claw_tip_ee_preclose_m", [0, 0, 0])
            clo = params.get("left_claw_tip_ee_close_m", [0, 0, 0])
            rospy.loginfo(
                "📋 指尖TCP: enable mode=ee | preclose@%.0f=[%.4f,%.4f,%.4f] close@%.0f=[%.4f,%.4f,%.4f]",
                params.get("left_claw_tip_preclose_pos", 35.0),
                pre[0], pre[1], pre[2],
                params.get("left_claw_tip_close_pos", 62.0),
                clo[0], clo[1], clo[2],
            )


def _retreat_xy_toward_shoulder(target_x, target_y, is_left_arm, retreat_m):
    """从肩点向目标退 retreat_m，用于水平预瞄（避免 init/曲肘直跳目标扫桌）。"""
    shoulder_x = -0.017
    shoulder_y = 0.292 if is_left_arm else -0.292
    dist = math.hypot(target_x - shoulder_x, target_y - shoulder_y)
    if dist <= retreat_m + 0.01:
        return target_x, target_y
    ratio = (dist - retreat_m) / dist
    pre_x = shoulder_x + (target_x - shoulder_x) * ratio
    pre_y = shoulder_y + (target_y - shoulder_y) * ratio
    return pre_x, pre_y


def _left_ready_merged(q_right_hold):
    """左手曲肘护胸构型，右手保持抓瓶。"""
    q = np.radians(mag._auto_grasp_ready_deg(True))
    q[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
    return q


def _right_ready_merged(q_left_hold):
    """右手曲肘护胸构型，左手保持当前。"""
    q = np.radians(mag._auto_grasp_ready_deg(False))
    q[LEFT_SLICE] = q_left_hold[LEFT_SLICE]
    return q


def _shoulder_swing_merged(q_ref, is_left_arm, q_other_hold):
    """vla 大鹏展翅：单臂肩膀外摆 75°，另一臂锁定（28.moveit_grasping / execute_vla_style_return）。"""
    q = np.copy(q_ref)
    swing = math.radians(mag.SHOULDER_SWING_AVOID_DEG)
    if is_left_arm:
        q[1] += swing
        q[RIGHT_SLICE] = q_other_hold[RIGHT_SLICE]
    else:
        q[8] -= swing
        q[LEFT_SLICE] = q_other_hold[LEFT_SLICE]
    return q


def _left_at_ready_pose(q_14, q_right_hold, tol_rad=0.12):
    """左手是否已在曲肘护胸（与 _left_ready_merged 接近）。"""
    q_ready = _left_ready_merged(q_right_hold)
    return float(np.max(np.abs(q_14[LEFT_SLICE] - q_ready[LEFT_SLICE]))) < tol_rad


def execute_vla_return_one_arm(arm_pub, q_arm_ref, is_left_arm, q_other_hold, step_prefix=""):
    """单臂 vla 收手：肩膀外摆 → 曲肘护胸（不跳 init，由 execute_bimanual_safe_return 统一 init）。"""
    label = "左" if is_left_arm else "右"
    if is_left_arm and _left_at_ready_pose(q_arm_ref, q_other_hold):
        rospy.loginfo("⏭️ %s%s已在曲肘护胸，跳过外摆+曲肘", step_prefix, label)
        return np.copy(q_arm_ref)
    q_swing = _shoulder_swing_merged(q_arm_ref, is_left_arm, q_other_hold)
    if is_left_arm:
        execute_hold_right(arm_pub, q_swing, 2.0, q_other_hold, step_prefix + f"{label}手肩膀外摆避障")
        q_ready = _left_ready_merged(q_other_hold)
        execute_hold_right(arm_pub, q_ready, 3.0, q_other_hold, step_prefix + f"{label}手曲肘护胸收手")
    else:
        execute_hold_left(arm_pub, q_swing, 2.0, q_other_hold, step_prefix + f"{label}手肩膀外摆避障")
        q_ready = _right_ready_merged(q_other_hold)
        execute_hold_left(arm_pub, q_ready, 3.0, q_other_hold, step_prefix + f"{label}手曲肘护胸收手")
    return np.copy(mag.last_commanded_joints_rad)


def execute_bimanual_safe_return(arm_pub, q_left_last, q_right_hold):
    """
    双臂安全收手：先左后右 vla 大鹏展翅，最后 init。
    禁止双臂直接 execute_dual_arm_init_home（会扫桌面，见 question.md / 28.moveit_grasping.md）。
    """
    rospy.loginfo("⬅️ 双臂 vla 收手（肩膀外摆 → 曲肘 → init）...")
    q_after_left = execute_vla_return_one_arm(
        arm_pub, q_left_last, True, q_right_hold, step_prefix="[收手] ",
    )
    execute_vla_return_one_arm(
        arm_pub, q_right_hold, False, q_after_left, step_prefix="[收手] ",
    )
    mag.execute_dual_arm_init_home(arm_pub)


def compute_cap_target(vision_x, vision_y, bottle_z, params, grasp_x=None, grasp_y=None,
                       grasp_z=None):
    """
    推算瓶盖中心。Z 优先用「右手抓点 + 微抬增量」：右手微抬 5cm 时瓶盖随之升高。
    """
    cap_z_table = bottle_z + params["bottle_cap_rise_m"]
    extra_z = float(params["left_tcp_extra_z_m"])
    if grasp_z is not None:
        ref_grasp_z = mag.SAFE_LOCKED_Z + float(params["right_grasp_z_offset_m"])
        lift_raw = float(grasp_z) - ref_grasp_z
        lift_delta = lift_raw * float(params["cap_z_lift_scale"])
        cap_z = cap_z_table + lift_delta + extra_z
        rospy.loginfo(
            "🎯 瓶盖 Z: table=%.3f + 微抬Δ=%.3f×scale%.2f→%.3f + extra_z=%.3f → %.3f (grasp_z=%.3f)",
            cap_z_table, lift_raw, params["cap_z_lift_scale"], lift_delta, extra_z, cap_z, grasp_z,
        )
    else:
        cap_z = cap_z_table + extra_z
        rospy.loginfo("🎯 瓶盖 Z: table=%.3f + extra_z=%.3f → %.3f", cap_z_table, extra_z, cap_z)
    cap_vx = vision_x + params["cap_offset_x_m"]
    cap_vy = vision_y + params["cap_offset_y_m"]
    mode = str(params["cap_reference_mode"]).lower()

    if grasp_x is not None and mode != "vision":
        cap_gx = grasp_x + params["cap_from_grasp_x_m"]
        cap_gy = grasp_y + params["cap_from_grasp_y_m"]
        if mode == "right_grasp":
            cap_x, cap_y = cap_gx, cap_gy
        else:
            w = float(params["cap_blend_grasp_weight"])
            cap_x = (1.0 - w) * cap_vx + w * cap_gx
            cap_y = (1.0 - w) * cap_vy + w * cap_gy
        rospy.loginfo(
            "🎯 瓶盖 XY: vision(%.3f,%.3f) grasp+off(%.3f,%.3f) → mode=%s → (%.3f,%.3f)",
            cap_vx, cap_vy, cap_gx, cap_gy, mode, cap_x, cap_y,
        )
    else:
        cap_x, cap_y = cap_vx, cap_vy
        rospy.loginfo("🎯 瓶盖 XY: vision(%.3f,%.3f) mode=vision", cap_x, cap_y)

    global _ACTIVE_LEFT_CAP_XY
    cap_geom_x, cap_geom_y = cap_x, cap_y
    cap_x, cap_y, yaw_x, yaw_y = left_cap_locked_xy(cap_geom_x, cap_geom_y, params)
    _ACTIVE_LEFT_CAP_XY = {"yaw_x": yaw_x, "yaw_y": yaw_y}
    off_x, off_y = mag.tcp_offsets_for_arm(True)
    rospy.loginfo(
        "🎯 瓶盖 X 链: geom_x=%.3f + tcp_x=%.3f + extra_x=%.3f → IK_x=%.3f"
        " (yaw_ref_x=%.3f decouple=%s)",
        cap_geom_x, off_x, float(params["left_tcp_extra_x_m"]), cap_x,
        yaw_x, params.get("left_cap_yaw_decouple_extra", True),
    )
    rospy.loginfo(
        "🎯 瓶盖落点(+左手TCP %.3f,%.3f +extra): (%.3f, %.3f, %.3f)",
        off_x, off_y, cap_x, cap_y, cap_z,
    )
    return cap_x, cap_y, cap_z


def _quick_re_vision(params):
    """右手抓稳后快速重采视觉，抵消抓握导致的瓶位微移。"""
    n = int(params["cap_re_vision_frames"])
    x_hist, y_hist = [], []
    rospy.loginfo("👁️ 右手抓稳后重采视觉 (%d 帧)...", n)
    timeout = rospy.Time.now() + rospy.Duration(4.0)
    while len(x_hist) < n and rospy.Time.now() < timeout and not rospy.is_shutdown():
        try:
            msg = rospy.wait_for_message("/vla/yolo_target", PointStamped, timeout=0.3)
            if 0.30 <= msg.point.x <= 0.65:
                x_hist.append(msg.point.x)
                y_hist.append(msg.point.y)
        except Exception:
            pass
    if len(x_hist) < max(3, n // 2):
        return None, None
    rx, ry = float(np.median(x_hist)), float(np.median(y_hist))
    rospy.loginfo("👁️ 重采视觉中值: (%.3f, %.3f) 来自 %d 帧", rx, ry, len(x_hist))
    return rx, ry


def _collect_hsv_cap_target(params, timeout_sec=3.0):
    """
    从 /vla/cap_target 采集 HSV 蓝色瓶盖检测坐标（上位机 Orin 发布）。
    返回 (cap_x, cap_y, cap_z) 或 (None, None, None)。
    """
    if not params.get("use_hsv_cap", True):
        return None, None, None
    rospy.loginfo("🔵 等待 HSV 瓶盖检测 /vla/cap_target (超时 %.1fs)...", timeout_sec)
    try:
        msg = rospy.wait_for_message("/vla/cap_target", PointStamped, timeout=timeout_sec)
        rospy.loginfo("🔵 HSV 瓶盖坐标: X=%.3f Y=%.3f Z=%.3f", msg.point.x, msg.point.y, msg.point.z)
        return float(msg.point.x), float(msg.point.y), float(msg.point.z)
    except Exception:
        rospy.logwarn("⚠️ 未收到 /vla/cap_target，回退几何推算")
        return None, None, None


def _read_left_claw_effort():
    claw = get_controller()
    st = claw.last_state
    if st is None or len(st.data.effort) < 1:
        return 0.0
    return float(st.data.effort[0])


def refine_cap_xy_search(arm_pub, ik_client, cap_x, cap_y, hover_z, quat, seed_14,
                         q_right_hold, params):
    """
    悬停高度可选 XY 网格精搜（默认关闭）。
    开启时 3×3、限时 12s，支持 Ctrl+C / shutdown 中断。
    """
    if not params["cap_xy_refine_enable"]:
        return cap_x, cap_y, seed_14

    t0 = time.time()
    max_sec = float(params["cap_xy_refine_max_sec"])
    preclose = float(params["left_contact_preclose_pos"])
    if preclose > 0:
        pos, vel, eff = build_left_claw_cmd_hold_right(
            preclose, params["left_contact_preclose_effort"],
        )
        get_controller().call(pos, vel, eff, tag="preclose-left-search")
        time.sleep(0.6)

    step = float(params["cap_xy_refine_step_m"])
    half = int(params["cap_xy_refine_half_steps"])
    probe_drop = float(params["cap_xy_probe_z_m"])
    best_x, best_y, best_eff = cap_x, cap_y, -1.0
    q_curr = seed_14
    total = (2 * half + 1) ** 2
    rospy.loginfo(
        "🔍 XY 精搜: 中心(%.3f,%.3f) 步长%.0fmm ±%.0fmm (%d点, 限时%.0fs)",
        cap_x, cap_y, step * 1000, half * step * 1000, total, max_sec,
    )

    idx = 0
    for ix in range(-half, half + 1):
        for iy in range(-half, half + 1):
            if rospy.is_shutdown() or time.time() - t0 > max_sec:
                rospy.logwarn("⚠️ XY精搜中断 (Ctrl+C / 超时 / shutdown)")
                break
            idx += 1
            cx = cap_x + ix * step
            cy = cap_y + iy * step
            pose_h = _left_cap_pose(cx, cy, hover_z)
            q_try = solve_left_ik_holding_right(
                ik_client, pose_h, q_curr, q_right_hold,
                f"[左手] XY精搜 {idx}/{total}",
                contact_phase=True,  # 加 l5/l6 偏置，让夹爪水平
            )
            if q_try is None:
                continue
            execute_hold_right(arm_pub, q_try, 0.3, q_right_hold, "")
            q_curr = np.copy(mag.last_commanded_joints_rad)
            pose_p = _left_cap_pose(cx, cy, hover_z - probe_drop)
            q_probe = solve_left_ik_holding_right(
                ik_client, pose_p, q_curr, q_right_hold, "", quiet=True,
                contact_phase=True,
            )
            if q_probe is not None:
                execute_hold_right(arm_pub, q_probe, 0.2, q_right_hold, "")
                q_curr = np.copy(mag.last_commanded_joints_rad)
            time.sleep(0.08)
            eff = _read_left_claw_effort()
            if eff > best_eff:
                best_eff = eff
                best_x, best_y = cx, cy
        else:
            continue
        break

    rospy.loginfo(
        "🎯 XY精搜结果: (%.3f,%.3f)→(%.3f,%.3f) effort=%.2f",
        cap_x, cap_y, best_x, best_y, best_eff,
    )
    if best_eff < 0.05:
        rospy.logwarn("⚠️ XY精搜 effort 无有效反馈，保持原 cap_xy（建议先标定 cap_offset）")

    pose_best = _left_cap_pose(best_x, best_y, hover_z)
    q_best = solve_left_ik_holding_right(
        ik_client, pose_best, q_curr, q_right_hold, "[左手] XY精搜落点",
        contact_phase=True,
    )
    if q_best is not None:
        execute_hold_right(arm_pub, q_best, 0.5, q_right_hold, "左手移至精搜最佳点")
        q_curr = np.copy(mag.last_commanded_joints_rad)
    return best_x, best_y, q_curr


def _joints_or_last(q):
    """numpy 关节向量不能用 `or` 判空，须显式 is None。"""
    return mag.last_commanded_joints_rad if q is None else q


def _emergency_safe_return(reason=""):
    """Ctrl+C 或进程退出前尽力大鹏展翅收手。"""
    ctx = _EMERGENCY_CTX
    if ctx.get("done") or not ctx.get("armed") or ctx.get("arm_pub") is None:
        return
    rospy.logwarn("🛑 紧急收手: %s", reason)
    try:
        q_live = getattr(mag, "current_joints_rad", None)
        if q_live is None or len(q_live) < 14:
            q_live = mag.last_commanded_joints_rad
        q_live = np.copy(q_live)
        rospy.loginfo("🛑 使用Ctrl+C瞬间实际关节姿态作为安全收手起点")
        safe_abort(
            ctx["arm_pub"],
            q_right_hold=ctx.get("q_right_hold"),
            q_left_last=q_live,
        )
    except Exception as exc:
        rospy.logerr("🛑 紧急收手失败: %s — 请手动急停或另开终端 execute_dual_arm_init_home", exc)
    ctx["done"] = True


def _on_sigint(signum, frame):
    _emergency_safe_return("SIGINT (Ctrl+C)")
    raise KeyboardInterrupt


def execute_hold_right(arm_pub, q_14_rad, time_sec, q_right_hold, step_name=""):
    """下发 14 轴轨迹，右手强制保持在 q_right_hold（左手动时用）。"""
    _execute_hold_frozen(arm_pub, q_14_rad, time_sec, q_right_hold, freeze_right=True, step_name=step_name)


def execute_hold_left(arm_pub, q_14_rad, time_sec, q_left_hold, step_name=""):
    """下发 14 轴轨迹，左手强制保持在 q_left_hold（右手动时用）。"""
    _execute_hold_frozen(arm_pub, q_14_rad, time_sec, q_left_hold, freeze_right=False, step_name=step_name)


def _execute_hold_right_waypoints(arm_pub, waypoints, point_dt, q_right_hold, step_name=""):
    """一次发布整段密集关节路点，让底层连续插值，避免逐点启停。"""
    if not waypoints:
        return None
    times = []
    values = []
    for i, waypoint in enumerate(waypoints):
        q = np.copy(waypoint)
        q[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        times.append(float(i + 1) * float(point_dt))
        values.extend(mag._clamp_elbow_deg([math.degrees(v) for v in q]))
    total_sec = times[-1]
    if step_name:
        rospy.loginfo("▶️ %s（%d路点连续轨迹，%.2fs）", step_name, len(waypoints), total_sec)
    arm_pub.publish(armTargetPoses(times=times, values=values))
    q_last = np.copy(waypoints[-1])
    q_last[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
    mag.last_commanded_joints_rad = q_last
    if _EMERGENCY_CTX.get("armed"):
        _EMERGENCY_CTX["q_left"] = np.copy(q_last)
    rospy.sleep(total_sec)
    post_sleep = float((_ACTIVE_ARM_PARAMS or {}).get("arm_trajectory_post_sleep_sec", 0.12))
    hold_hz = float((_ACTIVE_ARM_PARAMS or {}).get("arm_hold_republish_hz", 20.0))
    _republish_arm_hold(arm_pub, q_last, post_sleep, hold_hz)
    return q_last


def _republish_arm_hold(arm_pub, q_14_rad, duration_sec, hz):
    """段末按同一关节角重复下发，减轻 WBC 在轨迹间隔下垂（右手抓瓶时尤为重要）。"""
    if duration_sec <= 0 or hz <= 0:
        return
    target_deg = mag._clamp_elbow_deg([math.degrees(r) for r in q_14_rad])
    period = 1.0 / float(hz)
    t_end = time.time() + duration_sec
    while time.time() < t_end and not rospy.is_shutdown():
        arm_pub.publish(armTargetPoses(times=[max(period, 0.05)], values=target_deg))
        rospy.sleep(period)


def _execute_hold_frozen(arm_pub, q_14_rad, time_sec, q_frozen_ref, freeze_right, step_name=""):
    q = np.copy(q_14_rad)
    if freeze_right:
        q[RIGHT_SLICE] = q_frozen_ref[RIGHT_SLICE]
    else:
        q[LEFT_SLICE] = q_frozen_ref[LEFT_SLICE]
    if step_name:
        rospy.loginfo("▶️ %s (%.1fs)", step_name, time_sec)
    target_deg = mag._clamp_elbow_deg([math.degrees(r) for r in q])
    arm_pub.publish(armTargetPoses(times=[time_sec], values=target_deg))
    mag.last_commanded_joints_rad = np.copy(q)
    if _EMERGENCY_CTX.get("armed"):
        _EMERGENCY_CTX["q_left"] = np.copy(q)
    post_sleep = 0.12
    hold_hz = 20.0
    if _ACTIVE_ARM_PARAMS is not None:
        post_sleep = float(_ACTIVE_ARM_PARAMS.get("arm_trajectory_post_sleep_sec", 0.12))
        hold_hz = float(_ACTIVE_ARM_PARAMS.get("arm_hold_republish_hz", 20.0))
    rospy.sleep(time_sec)
    _republish_arm_hold(arm_pub, q, post_sleep, hold_hz)



def _solve_left_ik_nearest(ik_client, pose_stamped, seed_14, q_ref, q_right_hold,
                           ee_link="zarm_l7_end_effector"):
    """固定TCP轨迹专用：指定末端链路，并选择最接近上一关节姿态的解。"""
    group_name, _ = mag._ik_group_profile(True)
    seeds = [np.copy(q_ref), np.copy(seed_14)]
    # 7自由度在腕部奇异位形附近可能跳分支；小扰动种子用于寻找连续分支。
    for idx in (LEFT_L5_LEVEL_IDX, LEFT_L6_LEVEL_IDX, LEFT_L7_TWIST_IDX):
        for delta_deg in (-2.0, -0.5, 0.5, 2.0):
            candidate = np.copy(q_ref)
            candidate[idx] += math.radians(delta_deg)
            seeds.append(candidate)

    best = None
    best_jump = float("inf")
    best_rms = float("inf")
    for seed in seeds:
        seed[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        req = GetPositionIKRequest()
        req.ik_request.group_name = group_name
        req.ik_request.ik_link_name = ee_link
        ps = PoseStamped()
        ps.header.frame_id = pose_stamped.header.frame_id
        ps.header.stamp = rospy.Time(0)
        ps.pose = pose_stamped.pose
        req.ik_request.pose_stamped = ps
        req.ik_request.robot_state = mag._build_robot_state_seed(seed)
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = rospy.Duration(0.25)
        try:
            resp = ik_client(req)
        except rospy.ServiceException:
            continue
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            continue
        merged = np.copy(q_ref)
        for j, name in enumerate(resp.solution.joint_state.name):
            if name in mag.joint_names_14:
                merged[mag.joint_names_14.index(name)] = resp.solution.joint_state.position[j]
        merged[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        delta = np.abs(merged[LEFT_SLICE] - q_ref[LEFT_SLICE])
        jump = float(np.max(delta))
        rms = float(np.sqrt(np.mean(delta * delta)))
        if (jump, rms) < (best_jump, best_rms):
            best = merged
            best_jump = jump
            best_rms = rms
        if jump <= math.radians(2.0):
            break
    if best is not None:
        rospy.loginfo(
            "🧭 连续IK候选最小跳变: max=%.2f° rms=%.2f°",
            math.degrees(best_jump), math.degrees(best_rms),
        )
    return best


def solve_left_ik_holding_right(ik_client, pose_stamped, seed_14, q_right_hold, step_name,
                                quiet=False, contact_phase=False):
    """
    左手 IK；seed 与结果中右手始终为抓握构型。
    链式 seed + official seed 双通道，提高路点切换成功率。
    """
    group_name, ee_link = mag._ik_group_profile(True)
    if step_name and not quiet:
        rospy.loginfo("⏳ IK: %s ...", step_name)

    seed_variants = [np.copy(seed_14)]
    seed_off = mag._build_ik_seed_for_pose(True, seed_14, use_official_active=True)
    seed_off[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
    seed_variants.append(seed_off)
    q_ready_seed = _left_ready_merged(q_right_hold)
    if float(np.max(np.abs(seed_14[LEFT_SLICE] - q_ready_seed[LEFT_SLICE]))) > 0.08:
        seed_variants.append(q_ready_seed)

    for seed in seed_variants:
        seed[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
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
                if not quiet:
                    rospy.logwarn("⚠️ IK 服务异常: %s", exc)
                continue
            if resp.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            merged = np.copy(seed)
            for j, name in enumerate(resp.solution.joint_state.name):
                if name in mag.joint_names_14:
                    merged[mag.joint_names_14.index(name)] = resp.solution.joint_state.position[j]
            merged[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
            merged = _apply_left_wrist_level_bias(merged, contact_phase=contact_phase)

            # FK 指尖补偿只用于旧 end_effector TCP；URDF 虚拟指尖时不能再二次补偿。
            if (
                _ACTIVE_LEFT_CAP_PARAMS is not None
                and _ACTIVE_LEFT_CAP_PARAMS.get("left_claw_tip_enable", False)
                and link != "left_gripper_tip"
            ):
                fk_adj = _fk_adjust_ik_target(
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y,
                    pose_stamped.pose.position.z,
                    merged, _ACTIVE_LEFT_CAP_PARAMS,
                )
                adj_x, adj_y, adj_z, err_x, err_y, err_z = fk_adj
                if abs(err_x) > 0.001 or abs(err_y) > 0.001 or abs(err_z) > 0.001:
                    # FK 检测到偏差，用修正后目标重新解 IK
                    ps_adj = PoseStamped()
                    ps_adj.header.frame_id = pose_stamped.header.frame_id
                    ps_adj.header.stamp = rospy.Time(0)
                    ps_adj.pose = pose_stamped.pose
                    ps_adj.pose.position.x = adj_x
                    ps_adj.pose.position.y = adj_y
                    ps_adj.pose.position.z = adj_z
                    req_adj = GetPositionIKRequest()
                    req_adj.ik_request = req.ik_request
                    req_adj.ik_request.pose_stamped = ps_adj
                    try:
                        resp_adj = ik_client(req_adj)
                        if resp_adj.error_code.val == MoveItErrorCodes.SUCCESS:
                            merged_adj = np.copy(seed)
                            for j, name in enumerate(resp_adj.solution.joint_state.name):
                                if name in mag.joint_names_14:
                                    merged_adj[mag.joint_names_14.index(name)] = resp_adj.solution.joint_state.position[j]
                            merged_adj[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
                            merged_adj = _apply_left_wrist_level_bias(merged_adj, contact_phase=contact_phase)
                            if step_name and not quiet:
                                rospy.loginfo("✅ %s IK+FK 成功 (%s)", step_name, link)
                            return merged_adj
                    except Exception:
                        pass

            if step_name and not quiet:
                rospy.loginfo("✅ %s IK 成功 (%s)", step_name, link)
            return merged

    if step_name and not quiet:
        rospy.logerr("❌ %s IK 无解", step_name)
    return None


# ── FK 正运动学指尖补偿 ──
_FK_CLIENT = None

def _get_fk_client():
    global _FK_CLIENT
    if _FK_CLIENT is None:
        try:
            _FK_CLIENT = rospy.ServiceProxy('/compute_fk', GetPositionFK)
            _FK_CLIENT.wait_for_service(timeout=rospy.Duration(2.0))
        except Exception:
            rospy.logwarn("⚠️ /compute_fk 服务不可用，FK 指尖补偿跳过")
            _FK_CLIENT = False
    return _FK_CLIENT if _FK_CLIENT is not False else None


def _fk_link_pose(q_14, link_name="left_gripper_tip"):
    """返回指定链路在 base_link 下的 FK Pose；固定TCP拧盖必须有该反馈。"""
    fk = _get_fk_client()
    if fk is None:
        return None
    try:
        req = GetPositionFKRequest()
        req.header.frame_id = "base_link"
        req.fk_link_names = [link_name]
        req.robot_state.joint_state.name = list(mag.joint_names_14)
        req.robot_state.joint_state.position = [float(v) for v in q_14]
        resp = fk(req)
        if resp.error_code.val != MoveItErrorCodes.SUCCESS or not resp.pose_stamped:
            return None
        return resp.pose_stamped[0].pose
    except Exception as exc:
        rospy.logwarn("⚠️ %s FK 失败: %s", link_name, exc)
        return None


def _fk_adjust_ik_target(finger_x, finger_y, finger_z, q_14, params):
    """
    用 MoveIt FK 计算夹爪指尖真实位置，返回修正后的 IK 目标 (x, y, z)。
    q_14: 14轴关节角 (rad)，用于 FK 输入。
    返回 (adjusted_x, adjusted_y, adjusted_z, delta_x, delta_y, delta_z)。
    """
    fk = _get_fk_client()
    if fk is None:
        return finger_x, finger_y, finger_z, 0.0, 0.0, 0.0

    dx = float(params.get("gripper_tip_dx_m", 0.0))
    dy = float(params.get("gripper_tip_dy_m", 0.0))
    dz = float(params.get("gripper_tip_dz_m", 0.0))
    if abs(dx) < 1e-5 and abs(dy) < 1e-5 and abs(dz) < 1e-5:
        return finger_x, finger_y, finger_z, 0.0, 0.0, 0.0

    try:
        req = GetPositionFKRequest()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["zarm_l7_end_effector"]
        req.robot_state.joint_state.name = list(mag.joint_names_14)
        req.robot_state.joint_state.position = [float(v) for v in q_14]
        resp = fk(req)
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            rospy.logwarn("⚠️ FK 求解失败 (error=%d), 跳过指尖补偿", resp.error_code.val)
            return finger_x, finger_y, finger_z, 0.0, 0.0, 0.0
        if len(resp.pose_stamped) < 1:
            return finger_x, finger_y, finger_z, 0.0, 0.0, 0.0

        ps = resp.pose_stamped[0]
        l7_pos = np.array([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z])
        q_l7 = ps.pose.orientation

        # FK 给出的 l7 姿态 + CLAW_ROLL 已在 IK 目标四元数中
        # 夹爪指尖 = l7_pos + R_l7 @ [dx, dy, dz]
        tip_offset = np.array([dx, dy, dz])
        delta = _quat_rotate_vec(q_l7, tip_offset)
        tip_pos = l7_pos + delta

        # 修正：指尖应该到 (finger_x, finger_y, finger_z)
        error = np.array([finger_x, finger_y, finger_z]) - tip_pos
        adjusted = np.array([finger_x, finger_y, finger_z]) + error

        rospy.loginfo(
            "🔧 FK指尖补偿: l7=(%.3f,%.3f,%.3f) tip=(%.3f,%.3f,%.3f) "
            "error=(%.3f,%.3f,%.3f) → IK目标调整=(%.3f,%.3f,%.3f)",
            l7_pos[0], l7_pos[1], l7_pos[2],
            tip_pos[0], tip_pos[1], tip_pos[2],
            error[0], error[1], error[2],
            adjusted[0], adjusted[1], adjusted[2],
        )
        return float(adjusted[0]), float(adjusted[1]), float(adjusted[2]), float(error[0]), float(error[1]), float(error[2])

    except Exception as exc:
        rospy.logwarn("⚠️ FK 服务调用失败: %s", exc)
        return finger_x, finger_y, finger_z, 0.0, 0.0, 0.0


def _right_claw_hold_cmd():
    """
    左手阶段操作左爪时，右爪必须保持夹紧，禁止发 CLAW_OPEN（否则会松开瓶身）。
    """
    claw = get_controller()
    st = claw.last_state
    if st is not None and len(st.data.position) >= 2:
        pos = float(st.data.position[1])
        eff = float(st.data.effort[1])
        if pos >= 45.0:
            return pos, max(eff, CLAW_EFFORT_CLOSE_RIGHT * 0.4)
    return CLAW_CLOSE_RIGHT, CLAW_EFFORT_CLOSE_RIGHT


def build_left_claw_cmd_hold_right(left_pos, left_effort):
    """仅动左爪；右爪保持当前夹紧（或默认闭合力矩）。"""
    r_pos, r_eff = _right_claw_hold_cmd()
    return (
        [float(left_pos), r_pos],
        list(CLAW_VEL),
        [float(left_effort), r_eff],
    )


def _left_claw_fb_metrics(claw):
    st = claw.last_state
    if st is None or len(st.data.position) < 1:
        return None, None
    return float(st.data.position[0]), float(st.data.effort[0])


def _call_left_claw_close_step(claw, pos_cmd, effort, tag, params):
    """
    发左爪闭合指令并等到 Reached(2) 再返回。
    驱动在 Moving 中拒收新指令（日志 service failed: b''），必须逐步等到位。
    """
    settle_sec = max(float(params.get("left_cap_close_settle_sec", 2.5)), 0.5)
    retries = max(int(params.get("left_cap_close_retries", 4)), 1)
    wait_sec = float(params.get("left_cap_close_wait_sec", 0.25))
    ramp_sleep = max(float(params.get("left_cap_close_ramp_sleep_sec", 0.0)), 0.0)

    for attempt in range(retries):
        pos, vel, eff = build_left_claw_cmd_hold_right(pos_cmd, effort)
        attempt_tag = tag if attempt == 0 else "%s-r%d" % (tag, attempt + 1)
        ok = claw.call(
            pos, vel, eff, tag=attempt_tag,
            wait_sec=wait_sec, abort_on_stall=False,
        )
        if not ok:
            rospy.logwarn(
                "⚠️ 左爪 %s 服务失败 %d/%d（可能仍在 Moving），等待后重试",
                attempt_tag, attempt + 1, retries,
            )
            claw.wait_side_settled("left", timeout=0.8, require_saw_moving=False)
            time.sleep(0.35)
            continue
        if claw.wait_side_settled("left", timeout=settle_sec, require_saw_moving=True):
            time.sleep(ramp_sleep)
            return True
        fb_pos, _ = _left_claw_fb_metrics(claw)
        st = claw.last_state
        if st is not None and len(st.state) >= 1 and int(st.state[0]) == 2:
            time.sleep(ramp_sleep)
            return True
        rospy.logwarn(
            "⚠️ 左爪 %s 未在 %.1fs 内 Reached (fb pos=%s)，重试",
            attempt_tag, settle_sec, "?" if fb_pos is None else "%.1f" % fb_pos,
        )
        time.sleep(0.25)
    return False


def _recover_left_claw_close(claw, target, effort, params, tag_prefix):
    """渐进结束后若反馈远低于目标，补发直达闭合。"""
    fb_pos, _ = _left_claw_fb_metrics(claw)
    if fb_pos is None or fb_pos >= target - 4.0:
        return True
    rospy.logwarn(
        "⚠️ 左爪仅到 %.1f，补压至 %.0f (差 %.0f)",
        fb_pos, target, target - fb_pos,
    )
    return _call_left_claw_close_step(
        claw, target, effort, "%s-recover" % tag_prefix, params,
    )


def _left_cap_final_squeeze(claw, target, effort, params, tag_prefix="close-left-cap"):
    """渐进闭合后终生成目标 pos/effort 再压紧一段时间。"""
    if not bool(params.get("left_cap_final_squeeze_enable", True)):
        return True
    hold_sec = max(float(params.get("left_cap_final_squeeze_sec", 0.7)), 0.0)
    passes = max(int(params.get("left_cap_final_squeeze_passes", 1)), 1)
    if hold_sec <= 1e-6:
        return True
    ok = True
    for i in range(passes):
        tag = "%s-squeeze" % tag_prefix if passes == 1 else "%s-squeeze-%d" % (tag_prefix, i + 1)
        ok = _call_left_claw_close_step(claw, target, effort, tag, params) and ok
        rospy.loginfo(
            "🔧 左爪终压 %d/%d: pos=%.0f effort=%.2f 保持 %.1fs",
            i + 1, passes, target, effort, hold_sec,
        )
        time.sleep(hold_sec)
    return ok


def close_left_cap_gradual(params, tag_prefix="close-left-cap"):
    """
    左爪渐进夹紧瓶盖：从 preclose 分步增至 left_cap_close_pos。
    参考 20.gripper_issue.md 力位混合：触到盖（effort 升高）或达目标即停。
    claw_safe 会将 pos 限在 max_close_pos、effort 限在 max_close_effort（见 apply_claw_safe_limits）。
    """
    claw = get_controller()
    target = float(params["left_cap_close_pos"])
    start = float(params.get("left_contact_preclose_pos", 35.0))
    step = max(float(params.get("left_cap_close_ramp_step", 5.0)), 1.0)
    effort = float(params["left_cap_effort"])
    stop_eff = float(params.get("left_cap_close_effort_stop", 0.32))
    min_pos = float(params.get("left_cap_close_min_pos", 0.0))
    ramp = bool(params.get("left_cap_close_ramp_enable", True))

    if not ramp:
        ok = _call_left_claw_close_step(claw, target, effort, tag_prefix, params)
        ok = _left_cap_final_squeeze(claw, target, effort, params, tag_prefix) and ok
        _log_left_claw_close_result(claw, target, effort, ok)
        return ok

    pos_cmd = start
    last_ok = True
    while pos_cmd <= target + 1e-6:
        pos_cmd = min(pos_cmd, target)
        tag = "%s-%.0f" % (tag_prefix, pos_cmd)
        last_ok = _call_left_claw_close_step(claw, pos_cmd, effort, tag, params) and last_ok
        fb_pos, fb_eff = _left_claw_fb_metrics(claw)
        if fb_pos is not None:
            rospy.loginfo(
                "🖐️ 左爪渐进闭合 %.0f → fb pos=%.1f effort=%.2f",
                pos_cmd, fb_pos, fb_eff if fb_eff is not None else 0.0,
            )
            if (
                fb_eff is not None
                and fb_eff >= stop_eff
                and pos_cmd >= start + step
                and fb_pos >= min_pos - 1e-6
            ):
                rospy.loginfo(
                    "✅ 左爪触阻停止 @ pos=%.1f effort=%.2f (阈值 %.2f, min_pos=%.0f)",
                    fb_pos, fb_eff, stop_eff, min_pos,
                )
                break
            if abs(fb_pos - pos_cmd) <= 3.0 and pos_cmd >= target - 1e-6:
                break
        if pos_cmd >= target - 1e-6:
            break
        pos_cmd += step

    last_ok = _left_cap_final_squeeze(claw, target, effort, params, tag_prefix) and last_ok
    _log_left_claw_close_result(claw, target, effort, last_ok)
    return last_ok


def _log_left_claw_close_result(claw, target_pos, target_eff, ok):
    st = claw.last_state
    if st is None or len(st.data.position) < 1:
        rospy.logwarn("⚠️ 左爪闭合后无状态反馈")
        return
    fb_pos = float(st.data.position[0])
    fb_eff = float(st.data.effort[0])
    if ok:
        rospy.loginfo(
            "✅ 左爪闭合完成: 目标≈%.0f → 反馈 pos=%.1f effort=%.2f",
            target_pos, fb_pos, fb_eff,
        )


def detect_left_contact(claw, effort_threshold):
    st = claw.last_state
    if st is None or len(st.data.effort) < 1:
        return False
    return float(st.data.effort[0]) >= effort_threshold


def descend_until_contact(arm_pub, ik_client, cap_x, cap_y, cap_z, quat, seed_14,
                          q_right_hold, params, start_z=None,
                          skip_preclose=False, skip_wrist_level=False):
    """
    悬停后小步纯 Z 下降触顶。
    effort_first（默认）：先半夹 → effort 触顶闭环；几何 cap_z 仅作软参考+硬下限。
    geometry_only：cap_z+stop_above 处硬停（旧行为）。
    """
    step_m = float(params["contact_step_m"])
    max_steps = int(params["contact_max_steps"])
    threshold = float(params["contact_effort_threshold"])
    stop_above = float(params["contact_z_stop_above_m"])
    z_soft = float(cap_z) + stop_above
    z_hard = float(cap_z) - float(params.get("contact_z_max_below_cap_m", 0.006))
    # world_z 模式下 EE 比指尖高 world_z_m，硬下限须相应下调
    if params.get("left_claw_tip_enable") and str(params.get("left_claw_tip_mode", "")).lower() == "world_z":
        z_hard -= float(params.get("left_claw_tip_world_z_m", 0.0))
    mode = str(params.get("contact_mode", "effort_first")).lower()
    effort_first = mode != "geometry_only"
    extra_steps = int(params.get("contact_extra_descend_steps", 6)) if effort_first else 0
    claw = get_controller()

    hover_z = float(cap_z) + float(params["cap_hover_m"])
    z = start_z if start_z is not None else (hover_z - step_m)
    q_curr = seed_14
    moved_steps = 0
    touched = False
    preclosed = False
    z_last_move = z

    preclose = float(params["left_contact_preclose_pos"])
    if (
        preclose > 0 and not skip_preclose
        and params.get("contact_preclose_before_descend", True)
    ):
        pos, vel, eff = build_left_claw_cmd_hold_right(
            preclose, params["left_contact_preclose_effort"],
        )
        claw.call(pos, vel, eff, tag="preclose-before-descend")
        time.sleep(0.55)
        preclosed = True
        rospy.loginfo(
            "🖐️ 触顶搜索前左爪半开 (pos=%.0f)，effort 触顶闭环已启用", preclose,
        )

    rospy.loginfo(
        "🔽 触顶搜索 mode=%s | z %.3f→硬下限%.3f (cap_z=%.3f 软参考+%.3f=%.3f)",
        mode, z, z_hard, cap_z, stop_above, z_soft,
    )

    steps_past_soft = 0
    for i in range(max_steps):
        if rospy.is_shutdown():
            rospy.logwarn("⚠️ 触顶搜索中断 (shutdown)")
            break
        if z < z_hard - 1e-6:
            rospy.logwarn("⚠️ 触达硬安全下限 z=%.3f (cap_z-%.3f)，停止下降", z_hard, z_hard - cap_z)
            break

        if not effort_first and z < z_soft - 1e-6:
            rospy.loginfo("✅ [geometry_only] 已降至 z=%.3f (≤ 软参考 %.3f)", z + step_m, z_soft)
            break

        if effort_first and z < z_soft - 1e-6:
            steps_past_soft += 1
            if steps_past_soft > extra_steps:
                rospy.logwarn(
                    "⚠️ 过软参考 %.3f 后再降 %d 步仍无 effort，停止 (z≈%.3f)",
                    z_soft, extra_steps, z + step_m,
                )
                break

        hover_pose = _left_cap_pose(cap_x, cap_y, z)
        q_try = solve_left_ik_holding_right(
            ik_client, hover_pose, q_curr, q_right_hold,
            f"[左手] 触顶搜索 {i + 1}/{max_steps} z={z:.3f}",
            contact_phase=False,
            quiet=(i > 0),
        )
        if q_try is None:
            rospy.logwarn("⚠️ 触顶 z=%.3f IK 无解，跳过本步", z)
            z -= step_m
            continue
        execute_hold_right(
            arm_pub, q_try, _left_descend_step_sec(params), q_right_hold,
            f"左手垂直下降 {i + 1}/{max_steps}",
        )
        q_curr = np.copy(mag.last_commanded_joints_rad)
        z_last_move = z
        moved_steps += 1
        _lock_left_approach_pitch(_ACTIVE_LEFT_IK_CTX.get("approach_pitch"), z)
        hold_hz = 20.0
        if _ACTIVE_ARM_PARAMS is not None:
            hold_hz = float(_ACTIVE_ARM_PARAMS.get("arm_hold_republish_hz", 20.0))
        q_steady = np.copy(q_curr)
        q_steady[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        _republish_arm_hold(arm_pub, q_steady, 0.15, hold_hz)
        if detect_left_contact(claw, threshold):
            touched = True
            rospy.loginfo(
                "✅ 左爪触顶 (effort≥%.2f) @ z=%.3f (cap_z=%.3f Δz=%+.3f)",
                threshold, z, cap_z, z - cap_z,
            )
            break
        z -= step_m

    if moved_steps == 0:
        rospy.logerr("❌ 触顶搜索未执行任何下降步（IK 全失败），中止左爪闭合")
        return q_curr, False, float(hover_z)

    if not skip_wrist_level:
        w5 = float(params.get("left_cap_wrist5_bias_rad", 0.0))
        w6 = float(params.get("left_cap_wrist6_bias_rad", 0.0))
        if abs(w5) > 1e-9 or abs(w6) > 1e-9:
            q_curr = _apply_left_claw_level_bias(q_curr, contact_phase=True)
            rospy.loginfo(
                "📐 触顶调平 l5=%.3f(%.1f°) l6=%.3f(%.1f°)",
                w5, math.degrees(w5), w6, math.degrees(w6),
            )
            execute_hold_right(
                arm_pub, q_curr, 0.35, q_right_hold, "左手触顶调平(l5主/l6辅)",
            )

    if preclose > 0 and not skip_preclose and not preclosed:
        pos, vel, eff = build_left_claw_cmd_hold_right(
            preclose, params["left_contact_preclose_effort"],
        )
        claw.call(pos, vel, eff, tag="preclose-left-descend")
        time.sleep(0.6)
        rospy.loginfo("🖐️ 触顶后左爪半开 (pos=%.0f)，右爪保持夹瓶", preclose)

    if not touched and not detect_left_contact(claw, threshold):
        rospy.logwarn(
            "⚠️ 未检测到触顶 effort，已在 z≈%.3f 继续（cap_z=%.3f 软参考=%.3f）",
            z_last_move if moved_steps else hover_z, cap_z, z_soft,
        )
    return q_curr, True, float(z_last_move if moved_steps else hover_z)


def _quat_mul(a, b):
    """Hamilton 积：组合旋转 (a * b)。"""
    q = mag.Quaternion()
    q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
    q.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
    q.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    return q


def _quat_yaw_rad(q):
    """从四元数提取绕世界 z 的 yaw（用于日志）。"""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _quat_world_z(angle_rad):
    q = mag.Quaternion()
    q.w = math.cos(angle_rad * 0.5)
    q.z = math.sin(angle_rad * 0.5)
    return q


def _quat_local_x(angle_rad):
    q = mag.Quaternion()
    q.w = math.cos(angle_rad * 0.5)
    q.x = math.sin(angle_rad * 0.5)
    return q


def _quat_local_y(angle_rad):
    q = mag.Quaternion()
    q.w = math.cos(angle_rad * 0.5)
    q.y = math.sin(angle_rad * 0.5)
    return q


def _quat_world_horizontal(angle_rad, axis_yaw_rad):
    """绕世界 XY 平面内指定方向的轴旋转。"""
    half = angle_rad * 0.5
    q = mag.Quaternion()
    q.w = math.cos(half)
    q.x = math.sin(half) * math.cos(axis_yaw_rad)
    q.y = math.sin(half) * math.sin(axis_yaw_rad)
    return q


def _left_level_axis_local(params):
    if str(params.get("left_cap_twist_level_axis", "local_x")).lower() == "local_y":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _quat_left_level(angle_rad, params):
    if str(params.get("left_cap_twist_level_axis", "local_x")).lower() == "local_y":
        return _quat_local_y(angle_rad)
    return _quat_local_x(angle_rad)


def _contact_ref_quat(cap_x, cap_y, params):
    """触顶/夹紧时的水平姿态（与接近阶段一致）。"""
    kw = _left_cap_orientation_kwargs()
    pitch = _locked_approach_pitch(params)
    yaw_x, yaw_y = _left_cap_yaw_xy(cap_x, cap_y, params)
    return get_topdown_left_quat(
        yaw_x, yaw_y, approach_pitch_override=pitch, **kw,
    )


def _twist_ref_quat(cap_x, cap_y, params):
    """
    拧盖参考姿态。默认 use_contact_yaw=True：与触顶一致，避免夹紧后再偏航导致 IK 无解。
    """
    if bool(params.get("left_cap_twist_use_contact_yaw", True)):
        return _contact_ref_quat(cap_x, cap_y, params)
    kw = _left_cap_orientation_kwargs()
    kw["yaw_extra"] = float(kw.get("yaw_extra", 0.0)) + float(
        params.get("left_cap_twist_face_right_yaw_rad", -1.57079633)
    )
    pitch = _locked_approach_pitch(params)
    yaw_x, yaw_y = _left_cap_yaw_xy(cap_x, cap_y, params)
    return get_topdown_left_quat(
        yaw_x, yaw_y, approach_pitch_override=pitch, **kw,
    )


def _cap_touch_z(cap_z, params):
    return float(cap_z) + float(params.get("contact_z_stop_above_m", 0.0))


def _move_left_to_cap_pose(arm_pub, ik_client, x, y, z, quat, q_seed, q_right_hold,
                           ik_label, move_label, duration, apply_level_bias=True,
                           tip_phase="close", q_continuity_ref=None, validate_twist_fk=False,
                           execute=True, twist_pivot_offset_ee=None):
    """固定 TCP 位姿 IK；(x,y,z) 为指尖落点，内部换算 EE 目标；可选 l5/l6 触顶偏置。"""
    params = _ACTIVE_LEFT_CAP_PARAMS or {}
    if twist_pivot_offset_ee is not None:
        pivot_delta = _quat_rotate_vec(quat, twist_pivot_offset_ee)
        ee_x = float(x) - float(pivot_delta[0])
        ee_y = float(y) - float(pivot_delta[1])
        ee_z = float(z) - float(pivot_delta[2])
    else:
        ee_x, ee_y, ee_z = _fingertip_target_to_ee_xyz(
            x, y, z, quat, params, tip_phase=tip_phase,
        )
    pose = mag._build_pose_stamped(ee_x, ee_y, ee_z, quat)
    if q_continuity_ref is not None:
        q = _solve_left_ik_nearest(
            ik_client, pose, q_seed, q_continuity_ref, q_right_hold,
            ee_link="zarm_l7_end_effector" if twist_pivot_offset_ee is not None else "left_gripper_tip",
        )
    else:
        q = solve_left_ik_holding_right(
            ik_client, pose, q_seed, q_right_hold, ik_label,
            contact_phase=False, quiet=True,
        )
    if q is None:
        rospy.logerr("❌ %s IK 无解", ik_label)
        return None
    if apply_level_bias:
        q = _apply_left_claw_level_bias(q, contact_phase=True)
    if q_continuity_ref is not None:
        jump_deg = math.degrees(float(np.max(np.abs(q[LEFT_SLICE] - q_continuity_ref[LEFT_SLICE]))))
        jump_limit = float(params.get("left_cap_twist_joint_jump_deg", 8.0))
        if jump_deg > jump_limit:
            rospy.logerr("❌ %s 关节解跳变 %.1f° > %.1f°，拒绝执行", ik_label, jump_deg, jump_limit)
            return None
    if validate_twist_fk:
        fk_link = "zarm_l7_end_effector" if twist_pivot_offset_ee is not None else "left_gripper_tip"
        fk_pose = _fk_link_pose(q, fk_link)
        if fk_pose is None:
            rospy.logerr("❌ %s 无法做 %s FK 校验，拒绝执行", ik_label, fk_link)
            return None
        actual_x = float(fk_pose.position.x)
        actual_y = float(fk_pose.position.y)
        actual_z = float(fk_pose.position.z)
        if twist_pivot_offset_ee is not None:
            actual_delta = _quat_rotate_vec(fk_pose.orientation, twist_pivot_offset_ee)
            actual_x += float(actual_delta[0])
            actual_y += float(actual_delta[1])
            actual_z += float(actual_delta[2])
        dx = actual_x - float(x)
        dy = actual_y - float(y)
        dz = actual_z - float(z)
        xy_err = math.hypot(dx, dy)
        xy_tol = float(params.get("left_cap_twist_fk_xy_tolerance_m", 0.0025))
        z_tol = float(params.get("left_cap_twist_fk_z_tolerance_m", 0.0015))
        if xy_err > xy_tol or abs(dz) > z_tol:
            rospy.logerr(
                "❌ %s FK漂移 xy=%.1fmm z=%.1fmm (限 %.1f/%.1fmm)，拒绝执行",
                ik_label, xy_err * 1000.0, dz * 1000.0, xy_tol * 1000.0, z_tol * 1000.0,
            )
            return None
    if execute:
        execute_hold_right(arm_pub, q, duration, q_right_hold, move_label)
    return q


def _twist_ik_to_angle(arm_pub, ik_client, cap_x, cap_y, touch_z, ref_quat, twist_rad,
                       q_seed, q_right_hold, params, tag, start_ang=0.0,
                       apply_level_bias=False):
    """预计算并校验固定TCP密集路点，再作为一整段连续轨迹发布。"""
    requested_step_deg = max(float(params.get("left_cap_twist_step_deg", 15.0)), 0.25)
    max_cart_step_deg = max(float(params.get("left_cap_twist_cartesian_max_step_deg", 1.0)), 0.25)
    step_deg = min(requested_step_deg, max_cart_step_deg)
    n = max(1, int(math.ceil(abs(math.degrees(twist_rad)) / step_deg)))
    actual_step_deg = abs(math.degrees(twist_rad)) / float(n)
    deg_per_sec = max(float(params.get("left_cap_twist_cartesian_deg_per_sec", 6.0)), 0.5)
    dt = max(actual_step_deg / deg_per_sec, 0.08)
    q = np.copy(q_seed)
    q_cycle_start = np.copy(q_seed)
    pivot_offset = np.asarray(
        params.get("left_cap_twist_pivot_ee_m", [0.04, -0.03, 0.05]), dtype=float,
    )
    waypoints = []
    done_rad = 0.0

    for i in range(1, n + 1):
        frac = float(i) / float(n)
        step_rad = twist_rad * frac
        ang = start_ang + step_rad
        q_ori = _quat_mul(_quat_world_z(ang), ref_quat)
        pitch_comp_deg = abs(math.degrees(ang)) * float(
            params.get("left_cap_twist_pitch_comp_deg_per_deg", 0.0)
        )
        pitch_comp_rad = math.radians(pitch_comp_deg)
        if str(params.get("left_cap_twist_pitch_comp_frame", "world_horizontal")).lower() == "local_y":
            q_ori = _quat_mul(q_ori, _quat_left_level(pitch_comp_rad, params))
        else:
            # 将实机前后俯仰轴投影到世界水平面，并在整段旋转中固定该补偿轴。
            ref_axis_world = _quat_rotate_vec(ref_quat, _left_level_axis_local(params))
            axis_yaw = math.atan2(float(ref_axis_world[1]), float(ref_axis_world[0]))
            q_ori = _quat_mul(_quat_world_horizontal(pitch_comp_rad, axis_yaw), q_ori)
        abs_ang_deg = abs(math.degrees(ang))
        x_comp = abs_ang_deg * float(
            params.get("left_cap_twist_x_comp_per_deg_m", 0.0)
        )
        y_comp = abs_ang_deg * float(
            params.get("left_cap_twist_y_comp_per_deg_m", 0.0)
        )
        z_comp = abs_ang_deg * float(
            params.get("left_cap_twist_z_comp_per_deg_m", 0.0)
        )
        waypoint_x = float(cap_x) + x_comp
        waypoint_y = float(cap_y) + y_comp
        waypoint_z = float(touch_z) + z_comp
        ik_seed = np.copy(q)
        if bool(params.get("left_cap_twist_seed_l7_enable", True)):
            l7_min = math.radians(float(params.get("left_cap_twist_l7_seed_min_deg", -38.0)))
            l7_max = math.radians(float(params.get("left_cap_twist_l7_seed_max_deg", 38.0)))
            l7_des = q_cycle_start[LEFT_L7_TWIST_IDX] + step_rad
            ik_seed[LEFT_L7_TWIST_IDX] = min(max(l7_des, l7_min), l7_max)
            ik_seed[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        q_next = _move_left_to_cap_pose(
            arm_pub, ik_client, waypoint_x, waypoint_y, waypoint_z, q_ori, ik_seed, q_right_hold,
            "[%s] 固定点 IK z=%.1f° seed_l7=%.1f°" % (
                tag, math.degrees(ang), math.degrees(ik_seed[LEFT_L7_TWIST_IDX])
            ),
            "", dt,
            apply_level_bias=apply_level_bias,
            q_continuity_ref=q,
            validate_twist_fk=True,
            execute=False,
            twist_pivot_offset_ee=pivot_offset,
        )
        if q_next is None:
            break
        q = q_next
        waypoints.append(np.copy(q))
        done_rad = step_rad

    if not waypoints:
        return None, 0.0
    if len(waypoints) < n:
        rospy.logwarn(
            "⚠️ %s 仅规划成功 %d/%d 路点，执行已校验的 %.1f°",
            tag, len(waypoints), n, math.degrees(done_rad),
        )
    q_executed = _execute_hold_right_waypoints(
        arm_pub, waypoints, dt, q_right_hold,
        "%s 固定TCP水平旋转" % tag,
    )
    return q_executed, done_rad


def _twist_in_place_direct_l7(arm_pub, q_start, q_right_hold, total_rad, params, tag,
                              start_ang=0.0):
    """固定当前整臂构型，只转 l7；避免 IK 在位置优先时用肩肘抵消 yaw。"""
    step_deg = max(float(params.get("left_cap_twist_step_deg", 15.0)), 1.0)
    n = max(1, int(math.ceil(abs(math.degrees(total_rad)) / step_deg)))
    dt = float(params.get("left_cap_twist_step_sec", 0.45))
    q = np.copy(q_start)
    for i in range(1, n + 1):
        frac = float(i) / float(n)
        ang = start_ang + total_rad * frac
        q_next = np.copy(q_start)
        q_next[LEFT_L7_TWIST_IDX] = q_start[LEFT_L7_TWIST_IDX] + ang
        q_next[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        execute_hold_right(
            arm_pub, q_next, dt, q_right_hold,
            "%s l7 %.1f°" % (tag, math.degrees(ang)),
        )
        q = q_next
    return q


def _twist_cap_in_place_ik(arm_pub, ik_client, cap_x, cap_y, cap_z, q_start, q_right_hold, params,
                           contact_z=None):
    """
    原地 IK 拧盖：固定 left_gripper_tip 的 XYZ，仅旋转姿态绕世界 Z。
    不修改左爪抓取落点逻辑，也不复用 left_tcp_extra 作为轨道半径。
    """
    cycles = int(params.get("left_cap_twist_cycles", 4))
    cycle_deg = float(params.get("left_cap_twist_cycle_deg", 90.0))
    if cycles <= 0:
        rospy.logwarn("⏭️ left_cap_twist_cycles=0：跳过 in_place_ik 拧盖")
        return q_start, False

    touch_z = float(contact_z) if contact_z is not None else _cap_touch_z(cap_z, params)
    reclose = bool(params.get("left_cap_twist_reclose", False))
    release_pos = float(params.get("left_cap_twist_release_pos", 12.0))
    cycle_rad = math.radians(cycle_deg)
    ref_quat = _twist_ref_quat(cap_x, cap_y, params)
    rospy.loginfo(
        "🔄 拧盖 in_place_ik（固定指尖XYZ）: %d×%.0f° | touch_z=%.3f | reclose=%s | direct_l7=%s seed_l7=%s",
        cycles, cycle_deg, touch_z, reclose,
        params.get("left_cap_twist_direct_l7_enable", True),
        params.get("left_cap_twist_seed_l7_enable", True),
    )

    q = np.copy(q_start)
    twisted_any = False
    current_ang = 0.0
    total_done = 0.0
    target_total = abs(cycle_rad) * cycles
    use_direct_l7 = bool(params.get("left_cap_twist_direct_l7_enable", True))
    if not use_direct_l7:
        anchor_pose = _fk_link_pose(q_start, "zarm_l7_end_effector")
        if anchor_pose is None:
            rospy.logerr("❌ 固定TCP拧盖需要 /compute_fk 的 zarm_l7_end_effector，停止拧盖")
            return q_start, False
        pivot_offset = np.asarray(
            params.get("left_cap_twist_pivot_ee_m", [0.0, 0.0, -0.07]), dtype=float,
        )
        model_delta = _quat_rotate_vec(anchor_pose.orientation, pivot_offset)
        model_x = float(anchor_pose.position.x) + float(model_delta[0])
        model_y = float(anchor_pose.position.y) + float(model_delta[1])
        model_z = float(anchor_pose.position.z) + float(model_delta[2])
        configured_cap = (float(cap_x), float(cap_y), float(touch_z))
        if bool(params.get("left_cap_twist_skip_align", True)):
            # Preserve the already-grasped pivot instead of correcting perception/TCP mismatch
            # in the first twist waypoint. Rotation geometry and dynamic compensation stay unchanged.
            cap_x, cap_y, touch_z = model_x, model_y, model_z
        ref_quat = anchor_pose.orientation
        rospy.loginfo(
            "🎯 旋转中心: 使用=(%.4f, %.4f, %.4f) 配置=(%.4f, %.4f, %.4f) | "
            "EE→tip=[%.3f,%.3f,%.3f] | 对齐跳过=%s 配置中心误差=(%.1f,%.1f,%.1f)mm | "
            "XYZ补偿=(%.2f,%.2f,%.2f)mm/deg | "
            "pitch初值=%.2fdeg 斜率=%.3fdeg/deg frame=%s axis=%s",
            cap_x, cap_y, touch_z,
            configured_cap[0], configured_cap[1], configured_cap[2],
            pivot_offset[0], pivot_offset[1], pivot_offset[2],
            bool(params.get("left_cap_twist_skip_align", True)),
            (model_x - configured_cap[0]) * 1000.0,
            (model_y - configured_cap[1]) * 1000.0,
            (model_z - configured_cap[2]) * 1000.0,
            float(params.get("left_cap_twist_x_comp_per_deg_m", 0.0)) * 1000.0,
            float(params.get("left_cap_twist_y_comp_per_deg_m", 0.0)) * 1000.0,
            float(params.get("left_cap_twist_z_comp_per_deg_m", 0.0)) * 1000.0,
            float(params.get("left_cap_twist_pitch_offset_deg", 0.0)),
            float(params.get("left_cap_twist_pitch_comp_deg_per_deg", 0.0)),
            str(params.get("left_cap_twist_pitch_comp_frame", "world_horizontal")),
            str(params.get("left_cap_twist_level_axis", "local_x")),
        )
    max_attempts = cycles if use_direct_l7 else max(cycles, cycles * 4)
    c = 0
    while c < max_attempts and (use_direct_l7 or abs(total_done) < target_total - math.radians(0.5)):
        c += 1
        remain_rad = cycle_rad
        if not use_direct_l7:
            remain_abs = max(target_total - abs(total_done), 0.0)
            remain_rad = math.copysign(min(abs(cycle_rad), remain_abs), cycle_rad)
        rospy.loginfo(
            "🔩 原地拧盖尝试 %d/%d → 固定XYZ +%.0f° (累计 %.1f°/%.1f°)",
            c, max_attempts, math.degrees(remain_rad), math.degrees(abs(total_done)), math.degrees(target_total),
        )
        q_attempt_home = np.copy(q)
        if use_direct_l7:
            q_next = _twist_in_place_direct_l7(
                arm_pub, q, q_right_hold, remain_rad, params,
                tag="周期%d 原地旋转" % c,
                start_ang=0.0,
            )
            done_rad = remain_rad
        else:
            q_next, done_rad = _twist_ik_to_angle(
                arm_pub, ik_client, cap_x, cap_y, touch_z, ref_quat,
                remain_rad, q, q_right_hold, params,
                tag="周期%d 原地旋转" % c,
                start_ang=current_ang,
                apply_level_bias=False,
            )
        if q_next is None or abs(done_rad) < math.radians(0.5):
            rospy.logwarn("⚠️ 尝试 %d 原地旋转 IK 失败且无有效进展，停止", c)
            break
        q = q_next
        current_ang += done_rad
        total_done += abs(done_rad)
        twisted_any = True
        partial_attempt = abs(done_rad) < abs(remain_rad) - math.radians(0.5)

        if reclose and (use_direct_l7 and c < cycles or (not use_direct_l7 and abs(total_done) < target_total - math.radians(0.5))):
            pos, vel, eff = build_left_claw_cmd_hold_right(
                release_pos, params.get("left_contact_preclose_effort", 0.22),
            )
            get_controller().call(pos, vel, eff, tag="twist-inplace-release-%d" % c)
            time.sleep(0.5)
            if use_direct_l7:
                q_return = np.copy(q_attempt_home)
                q_return[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
                execute_hold_right(
                    arm_pub, q_return,
                    float(params.get("left_cap_twist_step_sec", 0.45)),
                    q_right_hold,
                    "周期%d 关节回放回正" % c,
                )
                return_done = -done_rad
            else:
                q_return, return_done = _twist_ik_to_angle(
                    arm_pub, ik_client, cap_x, cap_y, touch_z, ref_quat,
                    -done_rad, q, q_right_hold, params,
                    tag="周期%d 固定TCP反向回正" % c,
                    start_ang=current_ang,
                    apply_level_bias=False,
                )
            if q_return is None or abs(return_done + done_rad) > math.radians(0.5):
                rospy.logwarn("⚠️ 尝试 %d 固定TCP回正未完成，停止后续重夹", c)
                break
            q = q_return
            current_ang += return_done
            close_left_cap_gradual(params, tag_prefix="twist-inplace-reclose-%d" % c)
            time.sleep(0.2)
            hold_hz = float((_ACTIVE_ARM_PARAMS or {}).get("arm_hold_republish_hz", 20.0))
            q_steady = np.copy(q)
            q_steady[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
            _republish_arm_hold(arm_pub, q_steady, 0.8, hold_hz)
            if partial_attempt:
                rospy.logwarn("⚠️ 本轮只完成 %.1f°，已回正；不再把重复小动作累计为拧盖进度", math.degrees(done_rad))
                break

    return q, twisted_any


def _twist_orbital_ik(arm_pub, ik_client, cap_x, cap_y, touch_z, ref_quat, total_rad,
                      q_seed, q_right_hold, params, tag, start_ang=0.0):
    """
    轨道式 IK 拧盖：TCP 绕瓶盖中心 (cap_x, cap_y) 做小圆轨道运动，同时姿态绕 Z 旋转。
    轨道半径 = |left_tcp_extra + tcp_off|，即 TCP 当前位置到 cap 中心的距离。

    start_ang: 起始角度（前转=0；回正=cycle_rad，逐步回到0）
    total_rad: 总转角（前转=+cycle_rad；回正=-cycle_rad）
    """
    step_deg = float(params.get("left_cap_twist_step_deg", 15.0))
    step_sec = float(params.get("left_cap_twist_step_sec", 0.45))
    n = max(1, int(math.ceil(abs(math.degrees(total_rad)) / step_deg)))
    # TCP 当前位置距 cap 中心的完整偏移 = tcp_offset + left_extra
    off_x, off_y = mag.tcp_offsets_for_arm(True)
    full_off_x = float(off_x) + float(params.get("left_tcp_extra_x_m", 0.0))
    full_off_y = float(off_y) + float(params.get("left_tcp_extra_y_m", 0.0))
    rospy.logdebug(
        "轨道偏移: base=(%.3f,%.3f) extra=(%.3f,%.3f) full=(%.3f,%.3f)",
        off_x, off_y, params.get("left_tcp_extra_x_m", 0.0),
        params.get("left_tcp_extra_y_m", 0.0), full_off_x, full_off_y,
    )
    q = np.copy(q_seed)
    for i in range(1, n + 1):
        frac = float(i) / float(n)
        ang = start_ang + total_rad * frac  # 绝对角度
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        # TCP = C + Rz(θ) * full_off  → 从起始位置开始平滑绕 C 旋转
        r_x = cos_a * full_off_x - sin_a * full_off_y
        r_y = sin_a * full_off_x + cos_a * full_off_y
        orbit_x = cap_x + r_x
        orbit_y = cap_y + r_y
        orbit_z = touch_z
        # 姿态：绕世界 Z 转 ang
        q_ori = _quat_mul(_quat_world_z(ang), ref_quat)
        pose = mag._build_pose_stamped(orbit_x, orbit_y, orbit_z, q_ori)
        q_next = solve_left_ik_holding_right(
            ik_client, pose, q, q_right_hold,
            "[%s] 轨道 θ=%.1f°" % (tag, math.degrees(ang)),
            quiet=(i > 1),
        )
        if q_next is None:
            rospy.logerr("❌ [%s] 轨道 IK 无解 @ θ=%.0f°", tag, math.degrees(ang))
            return None
        execute_hold_right(
            arm_pub, q_next, step_sec, q_right_hold,
            "%s %.0f%%" % (tag, frac * 100.0),
        )
        q = q_next
    return q


def _twist_cap_cycle_ik(arm_pub, ik_client, cap_x, cap_y, cap_z, q_start, q_right_hold, params,
                        contact_z=None):
    """
    循环拧盖（轨道棘轮式）：
    TCP 绕瓶盖中心轨道旋转 → 松爪 → 轨道返回 → 再夹 → 重复。
    基于 IK 的 task-space 轨道运动，比 l7 关节旋转更精准。
    """
    cycles = int(params.get("left_cap_twist_cycles", 4))
    cycle_deg = float(params.get("left_cap_twist_cycle_deg", 90.0))
    if cycles <= 0:
        rospy.logwarn("⏭️ left_cap_twist_cycles=0：跳过拧盖")
        return q_start, False

    touch_z = float(contact_z) if contact_z is not None else _cap_touch_z(cap_z, params)
    reclose = bool(params.get("left_cap_twist_reclose", True))
    release_pos = float(params.get("left_cap_twist_release_pos", 12.0))
    cycle_rad = math.radians(cycle_deg)

    # 拧盖参考姿态（沿用触顶姿态）
    ref_quat = _twist_ref_quat(cap_x, cap_y, params)
    rospy.loginfo(
        "🔄 拧盖 cycle_ik（轨道棘轮式）: %d×%.0f° | touch_z=%.3f | reclose=%s",
        cycles, cycle_deg, touch_z, reclose,
    )

    q = np.copy(q_start)
    twisted_any = False
    for c in range(cycles):
        rospy.loginfo("🔩 拧盖周期 %d/%d → 绕 cap 轨道 +%.0f°", c + 1, cycles, cycle_deg)

        # --- Step 1: 旋转 — TCP 绕 cap 轨道前进 cycle_deg ---
        q = _twist_orbital_ik(
            arm_pub, ik_client, cap_x, cap_y, touch_z, ref_quat,
            cycle_rad, q, q_right_hold, params,
            tag="周期%d 旋转" % (c + 1),
        )
        if q is None:
            rospy.logwarn("⚠️ 周期 %d 轨道旋转 IK 失败，停止", c + 1)
            break
        twisted_any = True

        # --- Step 2: 松开夹爪（防回拖瓶盖） ---
        pos, vel, eff = build_left_claw_cmd_hold_right(
            release_pos, params.get("left_contact_preclose_effort", 0.22),
        )
        get_controller().call(pos, vel, eff, tag="twist-release-%d" % (c + 1))
        time.sleep(0.5)

        # --- Step 3: 夹爪返回开始 — TCP 反向轨道回到 cap 中心 ---
        q = _twist_orbital_ik(
            arm_pub, ik_client, cap_x, cap_y, touch_z, ref_quat,
            -cycle_rad, q, q_right_hold, params,
            tag="周期%d 回正" % (c + 1),
            start_ang=cycle_rad,  # 从 cycle_rad 平滑回到 0
        )
        if q is None:
            rospy.logwarn("⚠️ 周期 %d 轨道回正 IK 失败", c + 1)
            break

        # --- Step 4: 闭合夹爪再旋转 (非末周期) ---
        if reclose and c < cycles - 1:
            close_left_cap_gradual(params, tag_prefix="twist-reclose-%d" % (c + 1))
            time.sleep(0.2)
            hold_hz = float((_ACTIVE_ARM_PARAMS or {}).get("arm_hold_republish_hz", 20.0))
            q_steady = np.copy(q)
            q_steady[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
            _republish_arm_hold(arm_pub, q_steady, 0.8, hold_hz)

    return q, twisted_any


def _twist_cap_joint_l7(arm_pub, q_start, q_right_hold, params):
    """旧版：仅累加 l7 关节角，TCP 可能漂移。"""
    steps = int(params["twist_steps"])
    if steps <= 0:
        rospy.logwarn(
            "⏭️ twist_steps=0：已触顶+轻夹，但跳过 l7 拧盖旋转。"
            " 要真拧盖请去掉 _twist_steps:=0 或设 _twist_steps:=15"
        )
        return q_start
    deg = params["twist_deg_per_step"]
    q = np.copy(q_start)
    for i in range(steps):
        q[LEFT_L7_TWIST_IDX] += math.radians(deg)
        q[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        execute_hold_right(
            arm_pub, q, 0.45, q_right_hold,
            f"拧盖 {i + 1}/{steps} (+{deg:.0f}°)",
        )
        hold_hz = 20.0
        if _ACTIVE_ARM_PARAMS is not None:
            hold_hz = float(_ACTIVE_ARM_PARAMS.get("arm_hold_republish_hz", 20.0))
        q_steady = np.copy(q)
        q_steady[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
        _republish_arm_hold(arm_pub, q_steady, 0.15, hold_hz)
    return q


def twist_cap(arm_pub, ik_client, cap_x, cap_y, cap_z, q_start, q_right_hold, params,
              contact_z=None):
    """
    拧盖入口。
    cycle_ik（默认）：轨道棘轮式 — TCP 绕瓶盖中心轨道旋转→松爪→轨道返回→再夹→重复。
    joint_l7（legacy）：旧版仅加 l7 关节（无轨道，不推荐）。
    """
    mode = str(params.get("left_cap_twist_mode", "cycle_ik")).lower()
    if mode in ("in_place_ik", "inplace_ik", "fixed_xyz"):
        if int(params.get("left_cap_twist_cycles", 4)) <= 0:
            rospy.logwarn("⏭️ left_cap_twist_cycles=0：跳过 in_place_ik 拧盖")
            return q_start
        q_out, twisted = _twist_cap_in_place_ik(
            arm_pub, ik_client, cap_x, cap_y, cap_z, q_start, q_right_hold, params,
            contact_z=contact_z,
        )
        return q_out
    if mode == "cycle_ik":
        if int(params.get("left_cap_twist_cycles", 4)) <= 0:
            rospy.logwarn("⏭️ left_cap_twist_cycles=0：跳过 cycle_ik 拧盖")
            return q_start
        q_out, twisted = _twist_cap_cycle_ik(
            arm_pub, ik_client, cap_x, cap_y, cap_z, q_start, q_right_hold, params,
            contact_z=contact_z,
        )
        return q_out
    if int(params.get("twist_steps", 15)) <= 0:
        rospy.logwarn("⏭️ twist_steps=0：跳过 joint_l7 拧盖")
        return q_start
    return _twist_cap_joint_l7(arm_pub, q_start, q_right_hold, params)


def run_right_grasp_hold(left_arm, right_arm, arm_pub, ik_client, x_hist, y_hist, params):
    """
    阶段 A：右手抓瓶（复用 moveit_auto_grasp 流程，不抬升不收手）。
    返回 (success, q_right_hold, grasp_x, grasp_y, grasp_z)。
    """
    is_left_arm = False
    arm = right_arm
    locked_x, locked_y = right_locked_xy(float(np.median(x_hist)), float(np.median(y_hist)), params)
    locked_z = mag.SAFE_LOCKED_Z + float(params["right_grasp_z_offset_m"])
    rospy.loginfo(
        "🎯 右手 TCP 打击点: X=%.3f Y=%.3f Z=%.3f (extra_x=%.3f extra_y=%.3f z_off=%.3f)",
        locked_x, locked_y, locked_z,
        params["right_tcp_extra_x_m"], params["right_tcp_extra_y_m"], params["right_grasp_z_offset_m"],
    )

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
        return False, None, None, None, None

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
        return False, None, None, None, None

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
    get_controller().call(
        pos, vel, effort, tag="close-right-body", abort_on_stall=False,
    )
    time.sleep(2.0)

    ok_grasp, grasp_reason = verify_right_grasp(params)
    if not ok_grasp:
        rospy.logerr("❌ 右手抓握验收失败：%s", grasp_reason)
        return False, np.copy(mag.last_commanded_joints_rad), locked_x, locked_y, locked_z

    q_hold = np.copy(mag.last_commanded_joints_rad)
    grasp_z = locked_z

    if params["right_hold_after_grasp"]:
        lift_z = locked_z + params["right_micro_lift_m"]
        lift_pose = mag._build_pose_stamped(locked_x, locked_y, lift_z, quat)
        q_lift = mag._solve_pose_ik(
            ik_client, arm, is_left_arm, lift_pose, q_hold, "[右手] 微抬腾空间",
        )
        if q_lift is not None:
            mag.execute_single_pose(arm_pub, q_lift, 1.8, "右手微抬", is_left_arm)
            q_hold = np.copy(mag.last_commanded_joints_rad)
            grasp_z = lift_z

    rospy.loginfo("✅ 阶段 A 完成：右手抓稳，关节已锁定")
    return True, q_hold, locked_x, locked_y, grasp_z


def _left_cap_pose(x, y, z, approach_pitch_override=None, tip_phase="preclose"):
    """
    左手瓶盖路点；(x,y,z) 为指尖/瓶盖接触目标，IK 用 EE 补偿后位姿。
    接近/触顶用 preclose 偏移；拧盖阶段由 _move_left_to_cap_pose 用 close。
    """
    params = _ACTIVE_LEFT_CAP_PARAMS or {}
    kw = _left_cap_orientation_kwargs()
    pitch = _locked_approach_pitch(params, approach_pitch_override)
    yaw_x, yaw_y = _left_cap_yaw_xy(x, y, params)
    quat = get_topdown_left_quat(
        yaw_x, yaw_y, approach_pitch_override=pitch, **kw,
    )
    ee_x, ee_y, ee_z = _fingertip_target_to_ee_xyz(x, y, z, quat, params, tip_phase=tip_phase)
    return mag._build_pose_stamped(ee_x, ee_y, ee_z, quat)


def _left_ik_pitch_candidates(params):
    """主 pitch + 可选 fallback（右手侧夹 -π/2）。"""
    primary = float(params["left_cap_approach_pitch_rad"])
    cands = [primary]
    if params.get("left_ik_enable_pitch_fallback", True):
        fb = float(params.get("left_ik_pitch_fallback_rad", -1.57079633))
        if abs(fb - primary) > 1e-5:
            cands.append(fb)
    return cands


def _left_overhead_z_candidates(hover_z, cap_z, params):
    """瓶正上方候选 Z：从低到高探测（低处更易 IK）。"""
    high_z = hover_z + float(params["left_high_approach_m"])
    min_z = cap_z + float(params["cap_hover_m"]) + 0.015
    z_set = []
    for z in (hover_z, hover_z + 0.015, high_z):
        if z >= min_z - 1e-6:
            z_set.append(z)
    for d in params.get("left_ik_high_z_drop_m", [0.0, -0.03, -0.05]):
        z = high_z + float(d)
        if z >= min_z - 1e-6 and all(abs(z - e) > 1e-4 for e in z_set):
            z_set.append(z)
    z_set.sort()
    return z_set


def _probe_left_ik(ik_client, pose, q_right_hold, label, contact_phase=False, quiet=False):
    """仅 IK 探测，不下发轨迹。"""
    return solve_left_ik_holding_right(
        ik_client, pose, mag.last_commanded_joints_rad, q_right_hold, label,
        contact_phase=contact_phase, quiet=quiet,
    )


def _commit_left_ik(arm_pub, q, q_right_hold, label, duration):
    execute_hold_right(arm_pub, q, duration, q_right_hold, label)
    return q


def _move_left_pose_candidates(arm_pub, ik_client, poses, q_right_hold, ik_label, move_label,
                               duration, contact_phase=False, approach_pitch=None):
    """按序 IK 探测，首个成功者才动臂。"""
    for i, pose in enumerate(poses):
        label = ik_label if i == 0 else "%s 备选%d" % (ik_label, i)
        move = move_label if i == 0 else "%s (备选)" % move_label
        q = _probe_left_ik(ik_client, pose, q_right_hold, label, contact_phase, quiet=(i > 0))
        if q is None:
            continue
        z = pose.pose.position.z
        pitch = approach_pitch if approach_pitch is not None else _locked_approach_pitch(
            _ACTIVE_LEFT_CAP_PARAMS or {},
        )
        _lock_left_approach_pitch(pitch, z)
        _commit_left_ik(arm_pub, q, q_right_hold, move, duration)
        return q
    return None


def _try_left_pre_lift(arm_pub, ik_client, cap_x, cap_y, hover_z, q_right_hold, params):
    """
    曲肘后中间路点：朝肩 retreat + 额外抬 Z，再前往瓶盖，避免低扫蹭桌。
    IK 失败时仅告警，不阻断后续路径。
    """
    if not bool(params.get("left_pre_lift_enable", True)):
        return True
    retreat_m = float(params.get("left_pre_lift_retreat_m", 0.10))
    wx, wy = _retreat_xy_toward_shoulder(cap_x, cap_y, True, retreat_m)
    base_z = (
        hover_z + float(params["left_high_approach_m"])
        + float(params.get("left_pre_lift_extra_z_m", 0.06))
    )
    move_sec = _left_move_sec(params, "left_pre_lift_move_sec", 3.5, approach_only=True)
    z_cands = []
    for dz in (0.0, 0.03, -0.03, -0.05):
        z = base_z + dz
        if z >= hover_z + 0.015 and all(abs(z - e) > 1e-4 for e in z_cands):
            z_cands.append(z)
    z_cands.sort(reverse=True)
    for i, z_try in enumerate(z_cands):
        pose = _left_cap_pose(wx, wy, z_try)
        q = _probe_left_ik(
            ik_client, pose, q_right_hold,
            "[左手] 预抬升避障 z=%.3f" % z_try,
            contact_phase=False, quiet=(i > 0),
        )
        if q is None:
            continue
        pitch = _locked_approach_pitch(_ACTIVE_LEFT_CAP_PARAMS or {})
        _lock_left_approach_pitch(pitch, z_try)
        _commit_left_ik(arm_pub, q, q_right_hold, "左手预抬升避障", move_sec)
        rospy.loginfo(
            "✅ 预抬升路点 (%.3f,%.3f,%.3f) retreat=%.2fm",
            wx, wy, z_try, retreat_m,
        )
        return True
    rospy.logwarn("⚠️ 预抬升路点 IK 均失败，仍继续后续路径")
    return False


def _try_left_direct_overhead(arm_pub, ik_client, cap_x, cap_y, hover_z, cap_z,
                              q_right_hold, params, reason="", move_sec=None):
    """
    直连瓶盖正上方：IK 探测（不动臂）→ 成功才执行。
    锁定成功的 approach_pitch，供悬停/触顶沿用。
    """
    if reason:
        rospy.logwarn("⚠️ %s → 直连探测（pitch×Z 网格，探测成功才动）", reason)
    if move_sec is None:
        move_sec = _left_move_sec(params, "left_ik_probe_move_sec", 4.0, approach_only=True)
    for pitch in _left_ik_pitch_candidates(params):
        pitch_tag = "pitch=%.2f" % pitch
        for z_try in _left_overhead_z_candidates(hover_z, cap_z, params):
            pose = _left_cap_pose(cap_x, cap_y, z_try, approach_pitch_override=pitch)
            q = _probe_left_ik(
                ik_client, pose, q_right_hold,
                "[左手] 探测 %s z=%.3f" % (pitch_tag, z_try),
                quiet=True, contact_phase=False,
            )
            if q is None:
                continue
            _lock_left_approach_pitch(pitch, z_try)
            _commit_left_ik(
                arm_pub, q, q_right_hold,
                "左手至瓶盖正上方(%s z=%.3f pitch=%.2f)" % (pitch_tag, z_try, pitch),
                move_sec,
            )
            rospy.loginfo(
                "✅ 直连成功: z=%.3f pitch=%.3f（已锁定，悬停/触顶沿用同姿态）",
                z_try, pitch,
            )
            return True
    rospy.logerr("❌ 直连探测均失败")
    return False


def _move_left_holding_right(arm_pub, ik_client, pose, q_right_hold, ik_label, move_label,
                             duration, contact_phase=False, approach_pitch=None):
    q = _probe_left_ik(ik_client, pose, q_right_hold, ik_label, contact_phase)
    if q is None:
        return None
    z = pose.pose.position.z
    pitch = approach_pitch if approach_pitch is not None else _locked_approach_pitch(
        _ACTIVE_LEFT_CAP_PARAMS or {},
    )
    _lock_left_approach_pitch(pitch, z)
    _commit_left_ik(arm_pub, q, q_right_hold, move_label, duration)
    return q


def _left_vertical_to_z(arm_pub, ik_client, cap_x, cap_y, target_z, q_right_hold, params,
                        ik_label, move_label, duration=0.8, contact_phase=False):
    """从当前构型纯 Z 移到 target_z（姿态 pitch 已锁定）。"""
    pose = _left_cap_pose(cap_x, cap_y, target_z)
    q = _probe_left_ik(ik_client, pose, q_right_hold, ik_label, contact_phase)
    if q is None:
        return None
    _lock_left_approach_pitch(_ACTIVE_LEFT_IK_CTX.get("approach_pitch"), target_z)
    _commit_left_ik(arm_pub, q, q_right_hold, move_label, duration)
    return q


def _run_left_classic_approach(arm_pub, ik_client, cap_x, cap_y, cap_z, hover_z, high_z,
                               q_right_hold, params, lat, pre_x, fin_x, move_sec):
    """经典绕障：侧向高位 → 瓶前 → 切入 → 瓶正上方（IK 探测后动臂）。"""
    lat_candidates = [lat, max(lat * 0.5, 0.03)]
    side_ok = False
    for lat_try in lat_candidates:
        for z_side in _left_overhead_z_candidates(hover_z, cap_z, params):
            if z_side < high_z - 0.02:
                continue
            pose_try = _left_cap_pose(cap_x, cap_y + lat_try, z_side)
            if _move_left_holding_right(
                arm_pub, ik_client, pose_try, q_right_hold,
                "[左手] 侧向高位探测", "左手侧向高位", move_sec,
            ) is not None:
                side_ok = True
                high_z = z_side
                _lock_left_approach_pitch(_ACTIVE_LEFT_IK_CTX.get("approach_pitch"), z_side)
                break
        if side_ok:
            break

    pre_candidates = [pre_x, max(pre_x * 0.67, 0.04)]
    front_ok = False
    for z_front in _left_overhead_z_candidates(hover_z, cap_z, params):
        if z_front > high_z + 0.01:
            continue
        front_poses = [_left_cap_pose(cap_x - px, cap_y, z_front) for px in pre_candidates]
        if _move_left_pose_candidates(
            arm_pub, ik_client, front_poses, q_right_hold,
            "[左手] 瓶前高位探测", "左手瓶前高位", move_sec,
        ) is not None:
            front_ok = True
            high_z = z_front
            break

    if front_ok or side_ok:
        fin_candidates = [fin_x, max(fin_x * 0.67, 0.03)]
        cap_poses = [_left_cap_pose(cap_x - fx, cap_y, high_z) for fx in fin_candidates]
        cap_poses.append(_left_cap_pose(cap_x, cap_y, high_z))
        if _move_left_pose_candidates(
            arm_pub, ik_client, cap_poses, q_right_hold,
            "[左手] 切入/瓶上探测", "左手至瓶正上方", move_sec,
        ) is not None:
            return True
    return False


def run_left_approach_only(arm_pub, ik_client, vision_x, vision_y, bottle_z,
                           q_right_hold, params, grasp_x=None, grasp_y=None, grasp_z=None,
                           stop_at="contact", tune_mode=False,
                           cap_override=None):
    """
    左手至瓶盖悬停或触顶高度（不闭爪、不拧盖）。
    tune_mode=True：触顶时不自动腕偏置、不 preclose，便于人工调平。
    cap_override: (cap_x, cap_y, cap_z) 可选，来自 HSV 瓶盖检测，跳过几何推算。
    返回 (ok, q_left, meta)；meta 含 cap_x/y/z 与 ik_baseline（14轴快照）。
    """
    global _ACTIVE_LEFT_CAP_PARAMS, _ACTIVE_LEFT_CAP_XY
    _ACTIVE_LEFT_CAP_PARAMS = params
    _ACTIVE_LEFT_CAP_XY = {"yaw_x": None, "yaw_y": None}
    _reset_left_ik_ctx()
    if cap_override is not None and all(v is not None for v in cap_override):
        cap_x = float(cap_override[0]) + float(params.get("left_tcp_extra_x_m", 0.0))
        cap_y = float(cap_override[1]) + float(params.get("left_tcp_extra_y_m", 0.0))
        cap_z = float(cap_override[2]) + float(params.get("left_tcp_extra_z_m", 0.0))
        rospy.loginfo(
            "🔵 使用 HSV 瓶盖坐标: raw=(%.3f,%.3f,%.3f) + extra=(%.3f,%.3f,%.3f) → IK=(%.3f,%.3f,%.3f)",
            cap_override[0], cap_override[1], cap_override[2],
            params.get("left_tcp_extra_x_m", 0.0),
            params.get("left_tcp_extra_y_m", 0.0),
            params.get("left_tcp_extra_z_m", 0.0),
            cap_x, cap_y, cap_z,
        )
    else:
        cap_x, cap_y, cap_z = compute_cap_target(
            vision_x, vision_y, bottle_z, params,
            grasp_x=grasp_x, grasp_y=grasp_y, grasp_z=grasp_z,
        )
    hover_z = cap_z + params["cap_hover_m"]
    high_z = hover_z + params["left_high_approach_m"]
    lat = params["left_lateral_m"]
    pre_x = params["left_pre_forward_m"]
    fin_x = params["left_final_forward_m"]
    quat = get_topdown_left_quat(cap_x, cap_y, **_left_cap_orientation_kwargs())
    rospy.loginfo(
        "🎯 瓶盖推算 Z=%.3f | hover=%.3f high=%.3f (high 裕度 %.0fcm)",
        cap_z, hover_z, high_z, params["left_high_approach_m"] * 100,
    )

    classic_sec = _left_move_sec(params, "left_ik_classic_move_sec", 4.0, approach_only=True)
    direct_first = bool(params.get("left_approach_direct_first", True))

    # 🔧 自动标定：从 base_link extra + 当前抓取四元数反算 EE 系偏移
    _ee_cal = _calibrate_ee_offset(cap_x, cap_y, params)
    if _ee_cal is not None:
        rospy.loginfo(
            "🔧 [标定] 基于 extra=(%.3f,%.3f,%.3f) → EE系偏移=[%.4f, %.4f, %.4f]",
            params.get("left_tcp_extra_x_m", 0.0),
            params.get("left_tcp_extra_y_m", 0.0),
            params.get("left_tcp_extra_z_m", 0.0),
            float(_ee_cal[0]), float(_ee_cal[1]), float(_ee_cal[2]),
        )
        rospy.loginfo(
            "🔧 [标定] 下次用: _left_claw_tip_enable:=true _left_claw_tip_mode:=ee "
            "_left_tcp_extra_x_m:=0 _left_tcp_extra_y_m:=0 _left_tcp_extra_z_m:=0 "
            "_left_claw_tip_ee_close_m:=\"[%.4f, %.4f, %.4f]\"",
            float(_ee_cal[0]), float(_ee_cal[1]), float(_ee_cal[2]),
        )

    if not _left_at_ready_pose(mag.last_commanded_joints_rad, q_right_hold):
        execute_hold_right(
            arm_pub, _left_ready_merged(q_right_hold),
            _left_move_sec(params, "left_ready_move_sec", 4.4, approach_only=True), q_right_hold,
            "左手曲肘护胸",
        )

    _try_left_pre_lift(arm_pub, ik_client, cap_x, cap_y, hover_z, q_right_hold, params)

    reached = False
    if direct_first:
        reached = _try_left_direct_overhead(
            arm_pub, ik_client, cap_x, cap_y, hover_z, cap_z, q_right_hold, params,
            reason="优先直连",
        )
    if not reached:
        rospy.logwarn("⚠️ 直连未达，尝试 classic 绕障 ...")
        reached = _run_left_classic_approach(
            arm_pub, ik_client, cap_x, cap_y, cap_z, hover_z, high_z,
            q_right_hold, params, lat, pre_x, fin_x, classic_sec,
        )
    if not reached:
        reached = _try_left_direct_overhead(
            arm_pub, ik_client, cap_x, cap_y, hover_z, cap_z, q_right_hold, params,
            reason="classic 失败后最后直连",
        )
    if not reached:
        rospy.logerr("❌ 左手无法到达瓶盖上方")
        return False, np.copy(mag.last_commanded_joints_rad), {}

    reach_z = _ACTIVE_LEFT_IK_CTX.get("reach_z") or hover_z
    if reach_z > hover_z + 0.012:
        # 梯度降级：hover_z → hover_z+0.01 → hover_z+0.02 ... 逐一尝试
        z_candidates = [hover_z]
        step = 0.01
        while z_candidates[-1] + step <= reach_z + 0.005:
            z_candidates.append(z_candidates[0] + step * len(z_candidates))
        z_candidates.reverse()  # 从高到低试
        descended = False
        for z_try in z_candidates:
            q_try = _left_vertical_to_z(
                arm_pub, ik_client, cap_x, cap_y, z_try, q_right_hold, params,
                "[左手] 垂直降至 z=%.3f" % z_try,
                "左手垂直降至 z=%.3f" % z_try,
                duration=1.2,
                contact_phase=False,
            )
            if q_try is not None:
                reach_z = z_try
                descended = True
                break
        if not descended:
            rospy.logwarn("⚠️ 垂直降级全部失败，从当前高度 z=%.3f 继续", reach_z)
            hover_z = reach_z

    cap_x, cap_y, _ = refine_cap_xy_search(
        arm_pub, ik_client, cap_x, cap_y, hover_z, quat,
        mag.last_commanded_joints_rad, q_right_hold, params,
    )

    meta = {
        "cap_x": cap_x, "cap_y": cap_y, "cap_z": cap_z,
        "hover_z": hover_z,
    }

    if stop_at == "hover":
        q = np.copy(mag.last_commanded_joints_rad)
        meta["ik_baseline"] = np.copy(q)
        rospy.loginfo("✅ 左手已至悬停，进入调平前高度 z≈%.3f", hover_z)
        return True, q, meta

    descend_start = min(
        (_ACTIVE_LEFT_IK_CTX.get("reach_z") or hover_z) - params["contact_step_m"],
        hover_z - params["contact_step_m"],
    )
    q_contact, descended_ok, contact_z_actual = descend_until_contact(
        arm_pub, ik_client, cap_x, cap_y, cap_z, quat,
        mag.last_commanded_joints_rad, q_right_hold, params,
        start_z=descend_start,
        skip_preclose=tune_mode,
        skip_wrist_level=tune_mode,
    )
    if not descended_ok:
        return False, np.copy(mag.last_commanded_joints_rad), meta

    meta["ik_baseline"] = np.copy(q_contact)
    meta["contact_z"] = contact_z_actual
    rospy.loginfo(
        "✅ 左手已至触顶高度（stop_at=contact）z=%.3f，可人工调平", contact_z_actual,
    )
    return True, q_contact, meta


def run_left_unscrew(arm_pub, ik_client, vision_x, vision_y, bottle_z,
                     q_right_hold, params, grasp_x=None, grasp_y=None, grasp_z=None,
                     cap_override=None):
    """
    阶段 B~D：左手侧向高位 → 平移到瓶前 → 水平切入 → 垂直悬停
    → (可选) XY 精搜 → 触顶 → 轻夹 → (可选)拧盖。
    cap_override: (cap_x, cap_y, cap_z) 可选，来自 HSV 瓶盖检测。
    """
    ok, q_contact, meta = run_left_approach_only(
        arm_pub, ik_client, vision_x, vision_y, bottle_z, q_right_hold, params,
        grasp_x=grasp_x, grasp_y=grasp_y, grasp_z=grasp_z,
        stop_at="contact", tune_mode=False,
        cap_override=cap_override,
    )
    if not ok:
        return False, q_contact

    if params.get("save_left_tune_pose"):
        save_left_cap_tune_pose(q_contact, q_right_hold, meta, params=params)

    rospy.loginfo(
        "🖐️ 左爪夹紧瓶盖 (close=%.0f effort=%.2f ramp=%s)...",
        params["left_cap_close_pos"], params["left_cap_effort"],
        params.get("left_cap_close_ramp_enable", True),
    )
    close_left_cap_gradual(params, tag_prefix="close-left-cap")
    hold_hz = 20.0
    if _ACTIVE_ARM_PARAMS is not None:
        hold_hz = float(_ACTIVE_ARM_PARAMS.get("arm_hold_republish_hz", 20.0))
    q_steady = np.copy(q_contact)
    q_steady[RIGHT_SLICE] = q_right_hold[RIGHT_SLICE]
    _republish_arm_hold(arm_pub, q_steady, 2.0, hold_hz)

    q_last = twist_cap(
        arm_pub, ik_client,
        meta.get("cap_x"), meta.get("cap_y"), meta.get("cap_z"),
        q_steady, q_right_hold, params,
        contact_z=meta.get("contact_z"),
    )
    rospy.loginfo("✅ 阶段 B~D 完成（是否拧开需目视确认）")
    return True, q_last


def verify_right_grasp(params):
    """闭爪后检查右爪是否真正夹到瓶身（防空抓仍进左手阶段）。"""
    claw = get_controller()
    st = claw.last_state
    if st is None or len(st.data.position) < 2:
        return False, "无 /leju_claw_state"
    pos = float(st.data.position[1])
    eff = float(st.data.effort[1])
    min_pos = params["right_grasp_min_close_pos"]
    min_eff = params["right_grasp_min_effort"]
    accept_pos = params.get("right_grasp_accept_close_pos", 70.0)
    if pos < min_pos:
        return False, "右爪闭合不足 pos=%.1f (需≥%.0f)" % (pos, min_pos)
    if eff < min_eff and pos < accept_pos:
        return False, "右爪 effort 过低 %.2f (需≥%.2f，pos %.1f<%.0f，可能空抓)" % (
            eff, min_eff, pos, accept_pos,
        )
    if eff < min_eff:
        rospy.logwarn(
            "⚠️ 右手抓握 effort 偏低 %.2f，但闭合反馈 pos=%.1f ≥ %.0f，放行",
            eff, pos, accept_pos,
        )
    rospy.loginfo("✅ 右手抓握验收: pos=%.1f effort=%.2f", pos, eff)
    return True, ""


def _arms_near_init(tolerance_rad=0.15):
    """判断当前指令构型是否接近 init（用于 abort 是否可直跳 home）。"""
    init = mag._init_joints_rad()
    q = mag.last_commanded_joints_rad
    return float(np.max(np.abs(q - init))) < tolerance_rad


def _claws_confirmed_open(max_pos=25.0):
    claw = get_controller()
    st = claw.last_state
    if st is None or len(st.data.position) < 2:
        return False
    return all(float(v) <= max_pos for v in st.data.position[:2])


def _open_claw_best_effort(tag="release"):
    """重复发松爪命令，并以状态反馈确认双爪已打开。"""
    claw = get_controller()
    pos, vel, eff = build_open_cmd()
    max_attempts = 6
    for i in range(max_attempts):
        if _claws_confirmed_open():
            rospy.loginfo("✅ %s 双爪已确认打开", tag)
            return True

        # 上一个夹爪动作仍在Moving时服务会拒绝，先给它时间停稳。
        settle_deadline = time.time() + 1.5
        while time.time() < settle_deadline:
            st = claw.last_state
            if st is not None and len(st.state) >= 2 and all(int(v) != 1 for v in st.state[:2]):
                break
            time.sleep(0.1)

        attempt_tag = tag if i == 0 else "%s-r%d" % (tag, i + 1)
        ok = claw.call(pos, vel, eff, tag=attempt_tag, abort_on_stall=False, wait_sec=0.25)
        if not ok:
            rospy.logwarn("⚠️ %s 松爪指令失败 %d/%d，等待后重试", tag, i + 1, max_attempts)
            time.sleep(0.6)
            continue

        confirm_deadline = time.time() + 2.5
        while time.time() < confirm_deadline:
            if _claws_confirmed_open():
                rospy.loginfo("✅ %s 双爪已确认打开", tag)
                return True
            time.sleep(0.1)
        rospy.logwarn("⚠️ %s 松爪未到位 %d/%d，继续重试", tag, i + 1, max_attempts)

    rospy.logerr("❌ %s 双爪未确认打开，禁止执行大鹏展翅", tag)
    return False


def safe_abort(arm_pub, q_right_hold=None, q_left_last=None):
    """异常中止。双臂前伸时必须走 vla 大鹏展翅，禁止直跳 init 扫桌。"""
    rospy.logwarn("⬅️ 异常中止：先确认松爪，再安全收手 ...")
    if not _open_claw_best_effort(tag="release-abort"):
        rospy.logerr("🛑 夹爪未松开，保持当前手臂姿态；请人工处理或急停")
        return False

    need_vla_return = q_right_hold is not None or not _arms_near_init()
    if need_vla_return:
        q_r = np.copy(q_right_hold if q_right_hold is not None else mag.last_commanded_joints_rad)
        q_l = np.copy(q_left_last if q_left_last is not None else mag.last_commanded_joints_rad)
        rospy.loginfo("⬅️ 双臂前伸态 → vla 大鹏展翅安全收手（禁止直跳 init）")
        execute_bimanual_safe_return(arm_pub, q_l, q_r)
    else:
        rospy.loginfo("⬅️ 构型近 init → 直接归位")
        mag.execute_dual_arm_init_home(arm_pub)
    _open_claw_best_effort(tag="release-abort-after-return")
    _EMERGENCY_CTX["done"] = True
    return True


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("bimanual_unscrew")
    signal.signal(signal.SIGINT, _on_sigint)
    params = load_params()
    apply_claw_safe_limits(params)
    global _ACTIVE_ARM_PARAMS
    _ACTIVE_ARM_PARAMS = params
    _log_key_params(params)

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
    _EMERGENCY_CTX["arm_pub"] = arm_pub

    print("=" * 60)
    print("🤝 双臂协同 v1：右手抓瓶 + 左手拧盖")
    action = str(params.get("left_cap_twist_action", "tighten"))
    action_label = "拧紧" if action == "tighten" else "拧松（待标定）"
    print("   动作模式: %s [%s]" % (action_label, action))
    if params.get("use_hsv_cap", True):
        print("   分阶段执行 | 瓶盖=HSV检测(/vla/cap_target) 回退几何推算 | 实机务必有人监护")
    else:
        print("   分阶段执行 | 瓶盖=几何推算 | 实机务必有人监护")
    twist_mode = str(params.get("left_cap_twist_mode", "cycle_ik"))
    if twist_mode == "cycle_ik":
        nc = int(params.get("left_cap_twist_cycles", 4))
        if nc <= 0:
            print("   ⚠️ left_cap_twist_cycles=0 → 只测接近+触顶+轻夹")
        else:
            print("   拧盖 cycle_ik（轨道棘轮式）: %d×%.0f° → TCP绕cap轨道 → 松爪 → 轨道返回 → 再夹" % (
                nc, params.get("left_cap_twist_cycle_deg", 90.0),
            ))
    elif twist_mode in ("in_place_ik", "inplace_ik", "fixed_xyz"):
        nc = int(params.get("left_cap_twist_cycles", 4))
        if nc <= 0:
            print("   ⚠️ left_cap_twist_cycles=0 → 只测接近+触顶+轻夹")
        else:
            print("   拧盖 in_place_ik（固定指尖XYZ）: %d×%.0f°" % (
                nc, params.get("left_cap_twist_cycle_deg", 90.0),
            ))
    elif int(params["twist_steps"]) <= 0:
        print("   ⚠️ twist_steps=0 → 只测接近+触顶+轻夹，不旋转拧盖")
    else:
        print("   拧盖 joint_l7: %d 步 × %.0f°" % (
            int(params["twist_steps"]), params["twist_deg_per_step"],
        ))
    squeeze_sec = (
        params.get("left_cap_final_squeeze_sec", 0.7)
        if params.get("left_cap_final_squeeze_enable", True) else 0.0
    )
    squeeze_passes = max(int(params.get("left_cap_final_squeeze_passes", 1)), 1)
    print("   左爪夹盖: close=%.0f effort=%.2f 终压=%dx%.1fs (上限 %.0f/%.1fA)" % (
        params["left_cap_close_pos"], params["left_cap_effort"],
        squeeze_passes, squeeze_sec,
        params.get("claw_max_close_pos", 95.0),
        params.get("claw_max_close_effort", 1.2),
    ))
    print("=" * 60)

    mag.call_leju_claw(*build_open_cmd(), tag="open")
    time.sleep(1.0)
    mag.execute_dual_arm_init_home(arm_pub)

    x_hist, y_hist = mag._collect_vision_targets_tf2_style()
    if len(x_hist) < 10:
        rospy.logerr("❌ 视觉采集失败")
        safe_abort(arm_pub)
        return

    raw_x = float(np.median(x_hist))
    raw_y = float(np.median(y_hist))
    bottle_x, bottle_y = right_locked_xy(raw_x, raw_y, params)
    bottle_z = mag.SAFE_LOCKED_Z

    ok_ws, ws_reason = validate_bottle_workspace(raw_x, raw_y, params)
    if not ok_ws:
        rospy.logerr("❌ 工作空间检查未通过：%s", ws_reason)
        safe_abort(arm_pub)
        return
    rospy.loginfo(
        "✅ 工作空间 OK: 视觉(%.3f, %.3f) | 右手抓点(%.3f, %.3f) extra=(%.3f,%.3f)",
        raw_x, raw_y, bottle_x, bottle_y,
        params["right_tcp_extra_x_m"], params["right_tcp_extra_y_m"],
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

    # 🔵 右手抓瓶之前采集 HSV 瓶盖坐标
    hsv_cx, hsv_cy, hsv_cz = _collect_hsv_cap_target(params, timeout_sec=3.0)
    cap_override = (hsv_cx, hsv_cy, hsv_cz) if hsv_cx is not None else None

    q_right_hold = None
    try:
        ok, q_right_hold, grasp_x, grasp_y, grasp_z = run_right_grasp_hold(
            left_arm, right_arm, arm_pub, ik_client, x_hist, y_hist, params,
        )
        if not ok:
            safe_abort(arm_pub, q_right_hold=q_right_hold,
                       q_left_last=mag.last_commanded_joints_rad)
            return

        _EMERGENCY_CTX["armed"] = True
        _EMERGENCY_CTX["q_right_hold"] = np.copy(q_right_hold)
        _EMERGENCY_CTX["q_left"] = np.copy(mag.last_commanded_joints_rad)

        # 🔵 抓后再读一次瓶盖
        hsv_cx2, hsv_cy2, hsv_cz2 = _collect_hsv_cap_target(params, timeout_sec=2.0)
        if hsv_cx2 is not None:
            cap_override = (hsv_cx2, hsv_cy2, hsv_cz2)
            rospy.loginfo("🔵 抓后重采瓶盖成功: (%.3f, %.3f, %.3f)", hsv_cx2, hsv_cy2, hsv_cz2)
        else:
            rospy.loginfo("🔵 抓后重采失败，沿用抓前坐标")

        if params["cap_re_vision_after_grasp"]:
            rx, ry = _quick_re_vision(params)
            if rx is not None:
                raw_x, raw_y = rx, ry
                rospy.loginfo("👁️ 使用抓后视觉: (%.3f, %.3f) 替换抓前 (%.3f, %.3f)",
                              raw_x, raw_y, float(np.median(x_hist)), float(np.median(y_hist)))

        ok, q_left_last = run_left_unscrew(
            arm_pub, ik_client, raw_x, raw_y, bottle_z, q_right_hold, params,
            grasp_x=grasp_x, grasp_y=grasp_y, grasp_z=grasp_z,
            cap_override=cap_override,
        )
        _EMERGENCY_CTX["q_left"] = np.copy(
            q_left_last if q_left_last is not None else mag.last_commanded_joints_rad)
        if not ok:
            safe_abort(arm_pub, q_right_hold=q_right_hold,
                       q_left_last=_EMERGENCY_CTX["q_left"])
            return
        if q_left_last is None:
            safe_abort(arm_pub, q_right_hold=q_right_hold)
            return

        rospy.loginfo("⬅️ 收手：松爪 + 双臂 vla 大鹏展翅 ...")
        mag.call_leju_claw(*build_open_cmd(), tag="release")
        time.sleep(0.8)
        execute_bimanual_safe_return(arm_pub, q_left_last, q_right_hold)
        _EMERGENCY_CTX["done"] = True
        try:
            rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
                changeArmCtrlModeRequest(control_mode=0))
        except Exception:
            pass
        print("🎉 双臂协同流程结束（请确认瓶盖是否已拧松）")

    except KeyboardInterrupt:
        rospy.logwarn("⚠️ 用户 Ctrl+C")
        _emergency_safe_return("KeyboardInterrupt")
    except Exception as exc:
        rospy.logerr("❌ 异常: %s", exc)
        _emergency_safe_return("异常: %s" % exc)
        raise
    finally:
        if _EMERGENCY_CTX.get("armed") and not _EMERGENCY_CTX.get("done"):
            _emergency_safe_return("finally 兜底")

if __name__ == "__main__":
    main()