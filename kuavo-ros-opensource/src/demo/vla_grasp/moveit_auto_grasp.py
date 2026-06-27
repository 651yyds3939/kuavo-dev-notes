#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from claw_safe import build_close_cmd, build_open_cmd, get_controller

import rospy
import moveit_commander
import geometry_msgs.msg
import math
import numpy as np
import time
from geometry_msgs.msg import PointStamped, PoseStamped
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetCartesianPath, GetCartesianPathRequest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# 🚨【极致精简】：只保留最基础的话题和夹爪服务，彻底断绝一切 NameError 隐患！
try:
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
    from kuavo_msgs.msg import armTargetPoses
except ImportError:
    rospy.logerr("❌ 无法导入 kuavo_msgs，请检查工作空间环境！")
    exit(1)

# =================================================================
# 🎯 TCP (工具中心点) 与 避障通道参数配置区
# =================================================================
# 左右手分别标定（Y>0 选左手，Y<0 选右手）
TCP_OFFSET_X_LEFT = -0.018      # 左手 X 基准
TCP_OFFSET_X_RIGHT = -0.024     # 右手仍略偏前，再多回拉 6mm
TCP_OFFSET_Y_LEFT = 0.020       # 左手偏左 → 减小往右修（原 0.03）
TCP_OFFSET_Y_RIGHT = 0.065      # 右手往左修
SAFE_LOCKED_Z = 0.385         # 与 vla_auto_grasp_daemon 一致
PRE_GRASP_DIST = 0.12        
LIFT_HEIGHT = 0.22            # 与 vla_auto_grasp_daemon 一致（22cm 拔高避桌）
SHOULDER_SWING_AVOID_DEG = 75.0  # vla 同款肩膀关节外摆避障
CLAW_ROLL_RIGHT = 1.5708  
CLAW_ROLL_LEFT = -1.5708  

# 笛卡尔直线切入引擎参数（12cm / 25 段 ≈ 4.8mm 步长）
CARTESIAN_NUM_WAYPOINTS = 25
IK_SERVICE_CANDIDATES = ['/compute_ik', '/move_group/compute_ik']
CARTESIAN_SERVICE_CANDIDATES = ['/compute_cartesian_path', '/move_group/compute_cartesian_path']
IK_CALL_TIMEOUT_SEC = 2.0
OFFICIAL_IK_SEED_7 = [0.0, 0.0, 0.0, -1.57079633, 0.0, 0.0, 0.0]  # auto_grasp_TF2 同款
MAX_ARM_JOINT_JUMP_DEG = 30.0   # 相邻航点单关节最大允许跳变（防冗余臂肘甩尾）
MIN_IK_SUCCESS_FRACTION = 0.70    # 与旧版 compute_cartesian_path fraction 阈值对齐

# 流式泵送安全参数（修复「手臂过快 / 抓不准」）
MIN_SEGMENT_DT = 0.08             # 每段最少 80ms，防止 MPC 收到过短指令猛冲
MAX_STREAM_POINTS = 35            # 单条轨迹最多航点数，避免 OMPL 上百点导致 dt 极小
SETTLE_TIME_SEC = 0.6             # 每步运动结束后等待关节到位
VELOCITY_SCALING = 0.35           # MoveIt 规划降速（1.0 会导致路径过激）
ACCELERATION_SCALING = 0.35

OCTOMAP_CLEAR_SERVICES = [
    '/move_group/clear_octomap',
    '/clear_octomap',
]

# 与 auto_grasp_TF2.py 完全一致的双臂初始归位角度（度）
DUAL_ARM_INIT_DEG = [
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
]

joint_names_14 = [
    'zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 'zarm_l5_joint', 'zarm_l6_joint', 'zarm_l7_joint',
    'zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint'
]
current_joints_rad = np.zeros(14)
last_commanded_joints_rad = np.zeros(14)
has_joint_states = False

def tcp_offsets_for_arm(is_left_arm):
    """返回 (offset_x, offset_y) 米，左右手独立标定。"""
    if is_left_arm:
        return TCP_OFFSET_X_LEFT, TCP_OFFSET_Y_LEFT
    return TCP_OFFSET_X_RIGHT, TCP_OFFSET_Y_RIGHT

def joint_states_callback(msg):
    global current_joints_rad, has_joint_states
    for i, name in enumerate(msg.name):
        if name in joint_names_14:
            idx = joint_names_14.index(name)
            current_joints_rad[idx] = msg.position[i]
    has_joint_states = True

class Quaternion:
    def __init__(self): self.w=0; self.x=0; self.y=0; self.z=0

def euler_to_rotation_matrix(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    q = Quaternion()
    if trace > 0:
        q.w = math.sqrt(trace + 1.0) / 2
        q.x = (R[2, 1] - R[1, 2]) / (4 * q.w); q.y = (R[0, 2] - R[2, 0]) / (4 * q.w); q.z = (R[1, 0] - R[0, 1]) / (4 * q.w)
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        j, k = (i + 1) % 3, (i + 2) % 3
        t = np.zeros(4)
        t[i] = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) / 2
        t[j] = (R[i, j] + R[j, i]) / (4 * t[i])
        t[k] = (R[k, i] + R[i, k]) / (4 * t[i])
        t[3] = (R[k, j] - R[j, k]) / (4 * t[i])
        q.x, q.y, q.z, q.w = t
    norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    if norm > 0: q.w /= norm; q.x /= norm; q.y /= norm; q.z /= norm
    return q

def get_horizontal_claw_quat(target_x, target_y, is_left_arm):
    yaw = math.atan2((target_y - (0.292 if is_left_arm else -0.292)), (target_x - (-0.017)))
    R_base = euler_to_rotation_matrix(yaw, -1.57079633, 0)
    cr, sr = math.cos(CLAW_ROLL_LEFT if is_left_arm else CLAW_ROLL_RIGHT), math.sin(CLAW_ROLL_LEFT if is_left_arm else CLAW_ROLL_RIGHT)
    return rotation_matrix_to_quaternion(R_base @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]))

def call_leju_claw(pos, vel, effort, tag="cmd"):
    return get_controller().call(pos, vel, effort, tag=tag)

def clear_octomap_cache():
    """清除 OctoMap 伪障碍（瓶身点云等），兼容 MoveIt 不同命名空间。"""
    cleared = False
    for svc_name in OCTOMAP_CLEAR_SERVICES:
        try:
            rospy.wait_for_service(svc_name, timeout=0.5)
            rospy.ServiceProxy(svc_name, Empty)()
            rospy.loginfo(f"🧹 OctoMap 已清除: {svc_name}")
            cleared = True
            break
        except rospy.ROSException:
            continue
    if not cleared:
        rospy.logwarn("⚠️ 未找到 clear_octomap 服务，若规划报障碍可检查 move_group 是否加载传感器")
    rospy.sleep(0.15)

def wait_joints_settle(timeout_sec=SETTLE_TIME_SEC):
    """等待关节稳定，减少「视觉锁点 vs 真机位姿」偏差导致的抓不准。"""
    rospy.sleep(timeout_sec)

def _clamp_elbow_deg(target_deg):
    """肘关节死锁防护（左右臂第 4 轴）。"""
    if target_deg[3] > 0.0:
        target_deg[3] = 0.0
    if target_deg[10] > 0.0:
        target_deg[10] = 0.0
    return target_deg

def _downsample_joint_chain(joint_chain, max_points=MAX_STREAM_POINTS):
    """均匀降采样，防止 OMPL 输出上百点导致每段 dt 过小、手臂过快。"""
    if len(joint_chain) <= max_points:
        return list(joint_chain)
    indices = np.linspace(0, len(joint_chain) - 1, max_points, dtype=int)
    return [joint_chain[i] for i in indices]

def publish_arm_trajectory_batch(arm_pub, joint_chain_rad, total_time_sec, is_left_arm=None):
    """
    一次性发布完整 14 轴轨迹（官方推荐用法），避免逐条消息打断 MPC 导致过快/抖动。
    times 为相对当前时刻的累计到达时间；values 为 deg 扁平数组。
    """
    global last_commanded_joints_rad
    if not joint_chain_rad:
        return False

    if arm_pub.get_num_connections() == 0:
        rospy.logwarn("⚠️ /kuavo_arm_target_poses 暂无订阅者，仍尝试下发（与 auto_grasp 行为一致）")

    chain = _downsample_joint_chain(joint_chain_rad)
    if is_left_arm is not None:
        chain = [_freeze_inactive_arm(q, is_left_arm) for q in chain]
    n = len(chain)
    dt = max(total_time_sec / n, MIN_SEGMENT_DT)
    actual_total = dt * n

    msg = armTargetPoses()
    msg.times = [(i + 1) * dt for i in range(n)]
    for joints_rad in chain:
        target_deg = _clamp_elbow_deg([math.degrees(r) for r in joints_rad])
        msg.values.extend(target_deg)

    arm_pub.publish(msg)
    last_commanded_joints_rad = np.copy(chain[-1])
    rospy.loginfo(
        f"📡 批量下发 {n} 个航点，总时长 {actual_total:.2f}s，"
        f"订阅者 {arm_pub.get_num_connections()} 个"
    )
    rospy.sleep(actual_total + 0.2)
    return True

def _log_ros_network():
    rospy.loginfo(f"🌐 ROS_MASTER_URI = {os.environ.get('ROS_MASTER_URI', '(未设置)')}")
    rospy.loginfo(f"🌐 ROS_IP         = {os.environ.get('ROS_IP', '(未设置)')}")

def execute_single_pose(arm_pub, joints_14_rad, time_sec, step_name="", is_left_arm=None):
    """单段 14 轴轨迹；is_left_arm 传入时锁定非活动臂。"""
    global last_commanded_joints_rad
    if is_left_arm is not None:
        joints_14_rad = _freeze_inactive_arm(joints_14_rad, is_left_arm)
    if step_name:
        rospy.loginfo(f"▶️ {step_name} ({time_sec}s)")
    target_deg = _clamp_elbow_deg([math.degrees(r) for r in joints_14_rad])
    arm_pub.publish(armTargetPoses(times=[time_sec], values=target_deg))
    last_commanded_joints_rad = np.copy(joints_14_rad)
    time.sleep(time_sec + 0.5)

def _ik_group_profile(is_left_arm):
    """视觉/厂家 IK 目标在夹爪 TCP，对应 URDF 的 *_end_effector（非 wrist link）。"""
    if is_left_arm:
        return "left_arm", "zarm_l7_end_effector"
    return "right_arm", "zarm_r7_end_effector"

def _ee_link_candidates(primary_ee_link):
    """优先 TCP end_effector，失败再回退 wrist link。"""
    if primary_ee_link.endswith("_end_effector"):
        return [primary_ee_link, primary_ee_link.replace("_end_effector", "_link")]
    return [primary_ee_link]

def _inactive_arm_slice(is_left_arm):
    return slice(7, 14) if is_left_arm else slice(0, 7)

def _active_arm_slice(is_left_arm):
    return slice(0, 7) if is_left_arm else slice(7, 14)

def _init_joints_rad():
    return np.radians(DUAL_ARM_INIT_DEG)

def _freeze_inactive_arm(joints_14_rad, is_left_arm):
    """仅动活动臂：非活动臂始终锁在 init_angles，防止左手跟着前伸。"""
    frozen = np.copy(joints_14_rad)
    init = _init_joints_rad()
    frozen[_inactive_arm_slice(is_left_arm)] = init[_inactive_arm_slice(is_left_arm)]
    return frozen

def _build_high_safe_joints(q_lift, is_left_arm):
    """vla_auto_grasp_daemon 同款：抬升后纯关节空间肩膀外摆，清空桌面领空。"""
    q = np.copy(q_lift)
    if is_left_arm:
        q[1] += math.radians(SHOULDER_SWING_AVOID_DEG)
    else:
        q[8] -= math.radians(SHOULDER_SWING_AVOID_DEG)
    return _freeze_inactive_arm(q, is_left_arm)

def _build_ik_seed_for_pose(is_left_arm, active_seed_14, use_official_active=False):
    """非活动臂锁 init；活动臂用 official 或链式 seed。"""
    seed = _init_joints_rad()
    if active_seed_14 is not None:
        active = np.asarray(active_seed_14, dtype=float)
        arm_sl = _active_arm_slice(is_left_arm)
        if use_official_active:
            seed[arm_sl] = OFFICIAL_IK_SEED_7
        else:
            seed[arm_sl] = active[arm_sl]
    return seed

def _solve_single_ik_mg(arm_group, is_left_arm, pose_stamped, seed_joints_rad, ee_link):
    """MoveGroupCommander 内置 IK（与 /compute_ik 共用 TRAC-IK，但自动匹配规划组）。"""
    seed = _build_ik_seed_for_pose(is_left_arm, seed_joints_rad)
    ps = PoseStamped()
    ps.header.frame_id = pose_stamped.header.frame_id
    ps.header.stamp = rospy.Time(0)
    ps.pose = pose_stamped.pose
    try:
        arm_group.set_start_state(_build_robot_state_seed(seed))
        arm_group.set_end_effector_link(ee_link)
        arm_group.clear_pose_targets()
        if not arm_group.set_pose_target(ps):
            return None
        merged = np.copy(seed)
        for j, name in enumerate(arm_group.get_active_joints()):
            merged[joint_names_14.index(name)] = arm_group.get_joint_value_target()[j]
        return _freeze_inactive_arm(merged, is_left_arm)
    except Exception as exc:
        rospy.logwarn(f"⚠️ MoveGroup IK ({ee_link}) 异常: {exc}")
        return None

def _solve_pose_ik(ik_client, arm_group, is_left_arm, pose_stamped, seed_14, step_name):
    """单点 MoveIt IK：official seed + TCP ee_link + MoveGroup/服务双通道。"""
    group_name, ee_link = _ik_group_profile(is_left_arm)
    rospy.loginfo(f"⏳ IK 求解: {step_name} ...")
    seed_variants = [
        _build_ik_seed_for_pose(is_left_arm, seed_14, use_official_active=True),
        _build_ik_seed_for_pose(is_left_arm, seed_14, use_official_active=False),
    ]
    for seed in seed_variants:
        for link in _ee_link_candidates(ee_link):
            if arm_group is not None:
                result = _solve_single_ik_mg(arm_group, is_left_arm, pose_stamped, seed, link)
                if result is not None:
                    rospy.loginfo(f"✅ {step_name} IK 成功 (MoveGroup / {link})")
                    return result
            result = _solve_single_ik(ik_client, group_name, link, pose_stamped, seed, is_left_arm)
            if result is not None:
                rospy.loginfo(f"✅ {step_name} IK 成功 (service / {link})")
                return result
    rospy.logerr(f"❌ {step_name} IK 无解")
    return None

def _build_pose_stamped(x, y, z, quat):
    ps = PoseStamped()
    ps.header.frame_id = "base_link"
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.position.z = z
    ps.pose.orientation.x = quat.x
    ps.pose.orientation.y = quat.y
    ps.pose.orientation.z = quat.z
    ps.pose.orientation.w = quat.w
    return ps

def _auto_grasp_ready_deg(is_left_arm):
    """auto_grasp_TF2 曲肘护胸关节角（度）。"""
    if is_left_arm:
        return [40, 20, 0, -120, 0, 0, -20, 20, 0, 0, -30, 0, 0, 0]
    return [20, 0, 0, -30, 0, 0, 0, 40, -20, 0, -120, 0, 0, -20]

def execute_dual_arm_init_home(arm_pub):
    """与 auto_grasp_TF2.py 相同：双臂硬编码归位，不依赖 MoveIt。"""
    global last_commanded_joints_rad
    print("🤖 双臂初始归位（auto_grasp_TF2 同款 init_angles）...")
    arm_pub.publish(armTargetPoses(times=[2.5], values=list(DUAL_ARM_INIT_DEG)))
    last_commanded_joints_rad = np.radians(DUAL_ARM_INIT_DEG)
    time.sleep(3.0)

def _collect_vision_targets_tf2_style():
    """与 auto_grasp_TF2.py 完全相同的视觉采集逻辑。"""
    x_hist, y_hist = [], []
    print("\n👁️ 正在听取 TF2 绝对坐标 (无视头部晃动，速采 10 帧)...")
    while len(x_hist) < 10 and not rospy.is_shutdown():
        try:
            msg = rospy.wait_for_message('/vla/yolo_target', PointStamped, timeout=0.2)
            if 0.30 <= msg.point.x <= 0.65:
                x_hist.append(msg.point.x)
                y_hist.append(msg.point.y)
                print(f"  📥 录入 ({len(x_hist)}/10): TF2_X={msg.point.x:.3f}, TF2_Y={msg.point.y:.3f}")
        except Exception:
            pass
    return x_hist, y_hist

def _safe_return_both_arms(left_arm, right_arm, arm_pub, step_name="安全收手"):
    """异常路径：直接回 init，不走 MoveIt OMPL（易扫桌面）。"""
    rospy.loginfo(f"\n🛡️ {step_name}：双臂回初始下垂...")
    execute_dual_arm_init_home(arm_pub)

def execute_vla_style_return(arm_pub, q_lift, is_left_arm):
    """
    vla_auto_grasp_daemon 验证过的收手：抬升 → 肩膀外摆 → 曲肘护胸 → init。
    MoveIt 不参与此段（URDF 无桌面碰撞体，OMPL 无法避桌）。
    """
    q_high_safe = _build_high_safe_joints(q_lift, is_left_arm)
    ready_rad = np.radians(_auto_grasp_ready_deg(is_left_arm))

    print(f"⬅️ 肩膀关节外摆 {int(SHOULDER_SWING_AVOID_DEG)}°，清空桌面领空...")
    execute_single_pose(arm_pub, q_high_safe, 2.0, "肩膀外摆避障", is_left_arm)

    print("🛡️ 曲肘护胸收手...")
    execute_single_pose(arm_pub, ready_rad, 3.0, "曲肘护胸", is_left_arm)

    print("🤖 恢复自然下垂...")
    execute_dual_arm_init_home(arm_pub)

# =================================================================
# 🧮 手搓笛卡尔直线插值 + moveit_msgs/GetPositionIK 求解引擎
# （严禁 compute_cartesian_path / 外部厂家 IK 服务）
# =================================================================
_robot_cmd = None
_ik_client = None
_cartesian_client = None

def _resolve_cartesian_service():
    global _cartesian_client
    if _cartesian_client is not None:
        return _cartesian_client
    for svc_name in CARTESIAN_SERVICE_CANDIDATES:
        try:
            rospy.wait_for_service(svc_name, timeout=IK_CALL_TIMEOUT_SEC)
            _cartesian_client = rospy.ServiceProxy(svc_name, GetCartesianPath)
            rospy.loginfo(f"✅ MoveIt 笛卡尔服务已锁定: {svc_name}")
            return _cartesian_client
        except rospy.ROSException:
            rospy.logwarn(f"⚠️ 笛卡尔服务 {svc_name} 未响应，尝试下一路径...")
    return None

def _get_robot_commander():
    global _robot_cmd
    if _robot_cmd is None:
        _robot_cmd = moveit_commander.RobotCommander()
    return _robot_cmd

def _resolve_ik_service():
    global _ik_client
    if _ik_client is not None:
        return _ik_client
    for svc_name in IK_SERVICE_CANDIDATES:
        try:
            rospy.wait_for_service(svc_name, timeout=IK_CALL_TIMEOUT_SEC)
            _ik_client = rospy.ServiceProxy(svc_name, GetPositionIK)
            rospy.loginfo(f"✅ MoveIt IK 服务已锁定: {svc_name}")
            return _ik_client
        except rospy.ROSException:
            rospy.logdebug(f"IK 服务 {svc_name} 不可用，尝试下一路径...")
    rospy.logerr("❌ 所有候选 IK 服务均不可用，请确认 move_group.launch 已启动！")
    return None

def _build_robot_state_seed(seed_joints_rad):
    """
    构建完整 RobotState 种子：先取 MoveIt 当前全身状态，再覆盖 14 轴手臂关节。
    只传 14 轴会导致 TRAC-IK 全部无解。
    """
    state = _get_robot_commander().get_current_state()
    js = state.joint_state
    positions = list(js.position)
    name_to_idx = {n: i for i, n in enumerate(js.name)}
    for slot, name in enumerate(joint_names_14):
        if name in name_to_idx:
            positions[name_to_idx[name]] = float(seed_joints_rad[slot])
    js.position = positions
    state.joint_state = js
    return state

def _apply_ik_solution_to_14d(seed_joints_rad, solution_joint_state, is_left_arm):
    merged = np.copy(seed_joints_rad)
    for j, name in enumerate(solution_joint_state.name):
        if name in joint_names_14:
            merged[joint_names_14.index(name)] = solution_joint_state.position[j]
    return _freeze_inactive_arm(merged, is_left_arm)

def _solve_single_ik(ik_client, group_name, ee_link, pose_stamped, seed_joints_rad, is_left_arm):
    """调用 /compute_ik 求解单点；非活动臂锁 init。"""
    ps = PoseStamped()
    ps.header.frame_id = pose_stamped.header.frame_id
    ps.header.stamp = rospy.Time(0)
    ps.pose = pose_stamped.pose

    req = GetPositionIKRequest()
    req.ik_request.group_name = group_name
    req.ik_request.ik_link_name = ee_link
    req.ik_request.pose_stamped = ps
    req.ik_request.robot_state = _build_robot_state_seed(seed_joints_rad)
    req.ik_request.avoid_collisions = False
    req.ik_request.timeout = rospy.Duration(0.5)
    try:
        resp = ik_client(req)
    except rospy.ServiceException as exc:
        rospy.logwarn(f"⚠️ IK 服务异常: {exc}")
        return None
    if resp.error_code.val != MoveItErrorCodes.SUCCESS:
        return None
    return _apply_ik_solution_to_14d(seed_joints_rad, resp.solution.joint_state, is_left_arm)

def interpolate_cartesian_line(start_ps, end_ps, num_waypoints=CARTESIAN_NUM_WAYPOINTS):
    """
    在起点与终点之间做 xyz 均匀直线插值，四元数姿态全程锁定不变（纯平移）。
    返回 num_waypoints+1 个 PoseStamped（含起终点）。
    """
    p0 = np.array([start_ps.pose.position.x, start_ps.pose.position.y, start_ps.pose.position.z])
    p1 = np.array([end_ps.pose.position.x, end_ps.pose.position.y, end_ps.pose.position.z])
    locked_ori = start_ps.pose.orientation
    waypoints = []
    for i in range(num_waypoints + 1):
        t = float(i) / float(num_waypoints)
        wp = PoseStamped()
        wp.header.frame_id = start_ps.header.frame_id
        wp.header.stamp = rospy.Time.now()
        wp.pose.position.x = p0[0] + t * (p1[0] - p0[0])
        wp.pose.position.y = p0[1] + t * (p1[1] - p0[1])
        wp.pose.position.z = p0[2] + t * (p1[2] - p0[2])
        wp.pose.orientation = locked_ori
        waypoints.append(wp)
    return waypoints

def _max_arm_joint_jump_deg(prev_joints, new_joints, is_left_arm):
    """计算本臂 7 轴中相邻航点的最大关节跳变（度）。"""
    arm_slice = slice(0, 7) if is_left_arm else slice(7, 14)
    delta = np.abs(new_joints[arm_slice] - prev_joints[arm_slice])
    return float(np.max(np.degrees(delta)))

def solve_cartesian_ik_chain(group_name, ee_link, waypoints, seed_joints_rad, is_left_arm):
    """
    逐航点调用 moveit_msgs/GetPositionIK，上一解作为下一解的种子。
    返回: (success_joints_list, success_fraction)
    """
    ik_client = _resolve_ik_service()
    if ik_client is None:
        return [], 0.0

    chain = []
    current_seed = np.copy(seed_joints_rad)

    for idx, wp in enumerate(waypoints):
        # 起点航点：机械臂已在预瞄位，直接采信当前关节角，不必再 IK
        if idx == 0:
            chain.append(np.copy(current_seed))
            continue

        candidate = _solve_single_ik(ik_client, group_name, ee_link, wp, current_seed, is_left_arm)
        if candidate is None:
            rospy.logwarn(f"⚠️ 航点 {idx}/{len(waypoints)-1} IK 无解")
            continue

        if chain:
            jump = _max_arm_joint_jump_deg(chain[-1], candidate, is_left_arm)
            if jump > MAX_ARM_JOINT_JUMP_DEG:
                rospy.logwarn(
                    f"⚠️ 航点 {idx} 肘关节甩尾跳变 {jump:.1f}° > {MAX_ARM_JOINT_JUMP_DEG}°，丢弃该点"
                )
                continue

        chain.append(candidate)
        current_seed = candidate

    fraction = len(chain) / max(len(waypoints), 1)
    return chain, fraction

def pack_joint_trajectory(joint_chain, total_time_sec):
    """
    将连续 14 轴关节序列封装为 trajectory_msgs/JointTrajectory（带均匀时间戳）。
    可直接供后续控制器或流式泵送逻辑消费。
    """
    traj = JointTrajectory()
    traj.joint_names = list(joint_names_14)
    if not joint_chain:
        return traj
    dt = total_time_sec / len(joint_chain)
    for i, joints in enumerate(joint_chain):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in joints]
        pt.time_from_start = rospy.Duration.from_sec((i + 1) * dt)
        traj.points.append(pt)
    return traj

def stream_joint_chain(arm_pub, joint_chain, total_time_sec):
    """将 14 轴关节链批量泵送至 /kuavo_arm_target_poses。"""
    return publish_arm_trajectory_batch(arm_pub, joint_chain, total_time_sec)

def execute_cartesian_linear_ik(arm_group, arm_pub, step_name, start_ps, end_ps, is_left_arm,
                                total_time_sec=2.5, num_waypoints=CARTESIAN_NUM_WAYPOINTS):
    """
    完整笛卡尔直线切入管线：插值 → 链式 IK → 封装轨迹 → 流式执行。
    """
    rospy.loginfo(f"\n🧠 自定义笛卡尔 IK 引擎启动: [{step_name}]...")
    clear_octomap_cache()
    wait_joints_settle(0.4)
    global last_commanded_joints_rad
    last_commanded_joints_rad = np.copy(current_joints_rad)

    group_name = "left_arm" if is_left_arm else "right_arm"
    _, ee_link = _ik_group_profile(is_left_arm)

    waypoints = interpolate_cartesian_line(start_ps, end_ps, num_waypoints)
    dist_m = math.sqrt(
        (end_ps.pose.position.x - start_ps.pose.position.x) ** 2 +
        (end_ps.pose.position.y - start_ps.pose.position.y) ** 2 +
        (end_ps.pose.position.z - start_ps.pose.position.z) ** 2
    )
    rospy.loginfo(
        f"📐 直线航程 {dist_m*100:.1f}cm，切分 {len(waypoints)} 个微观航点，姿态全程锁定"
    )

    joint_chain, fraction = solve_cartesian_ik_chain(
        group_name, ee_link, waypoints, last_commanded_joints_rad, is_left_arm
    )
    rospy.loginfo(f"🔗 链式 IK 完成：{len(joint_chain)}/{len(waypoints)} 航点求解成功 ({fraction*100:.1f}%)")

    if fraction < MIN_IK_SUCCESS_FRACTION:
        rospy.logerr(f"❌ [{step_name}] IK 覆盖率 {fraction*100:.1f}% 不足 {MIN_IK_SUCCESS_FRACTION*100:.0f}%！")
        return False

    # 强制末点贴近目标笛卡尔位姿，避免链式丢点导致抓不准
    ik_client = _resolve_ik_service()
    if ik_client is not None:
        end_joints = _solve_single_ik(
            ik_client, group_name, ee_link, waypoints[-1],
            joint_chain[-1] if joint_chain else last_commanded_joints_rad,
            is_left_arm,
        )
        if end_joints is not None:
            if not joint_chain or _max_arm_joint_jump_deg(joint_chain[-1], end_joints, is_left_arm) <= MAX_ARM_JOINT_JUMP_DEG:
                if joint_chain:
                    joint_chain[-1] = end_joints
                else:
                    joint_chain.append(end_joints)

    traj = pack_joint_trajectory(joint_chain, total_time_sec)
    rospy.loginfo(f"✅ 轨迹封装完毕：{len(traj.points)} 点 / {total_time_sec}s")

    ok = stream_joint_chain(arm_pub, joint_chain, total_time_sec)
    if ok:
        wait_joints_settle()
    return ok

def execute_cartesian_path_service(arm_pub, start_joints_rad, end_pose_stamped, is_left_arm,
                                   total_time_sec, step_name, min_fraction=0.95):
    """
    MoveIt 原生 GetCartesianPath 服务（非 Python compute_cartesian_path 绑定）。
    从 start_joints_rad 对应位姿沿直线插值至 end_pose，姿态锁定。
    """
    cart_client = _resolve_cartesian_service()
    if cart_client is None:
        rospy.logwarn(f"⚠️ [{step_name}] 笛卡尔服务不可用")
        return False

    group_name, ee_link = _ik_group_profile(is_left_arm)
    rospy.loginfo(f"\n📏 MoveIt 笛卡尔直线: [{step_name}] ({ee_link})...")

    req = GetCartesianPathRequest()
    req.header.frame_id = "base_link"
    req.start_state = _build_robot_state_seed(start_joints_rad)
    req.group_name = group_name
    req.link_name = ee_link
    req.waypoints = [end_pose_stamped.pose]
    req.max_step = 0.008
    req.jump_threshold = 0.0
    req.prismatic_jump_threshold = 0.0
    req.revolute_jump_threshold = 0.0
    req.avoid_collisions = False

    try:
        resp = cart_client(req)
    except rospy.ServiceException as exc:
        rospy.logwarn(f"⚠️ [{step_name}] 笛卡尔服务异常: {exc}")
        return False

    fraction = float(resp.fraction)
    rospy.loginfo(f"🔗 笛卡尔路径覆盖率: {fraction*100:.1f}%")
    if fraction < min_fraction:
        rospy.logerr(f"❌ [{step_name}] 笛卡尔覆盖率 {fraction*100:.1f}% < {min_fraction*100:.0f}%")
        return False

    traj = resp.solution.joint_trajectory
    if not traj.points:
        rospy.logerr(f"❌ [{step_name}] 笛卡尔轨迹为空")
        return False

    joint_chain = _trajectory_msg_to_14d_chain(traj, start_joints_rad)
    ok = publish_arm_trajectory_batch(arm_pub, joint_chain, total_time_sec, is_left_arm=is_left_arm)
    if ok:
        wait_joints_settle()
        rospy.loginfo(f"✅ [{step_name}] 笛卡尔直线执行完成 ({len(joint_chain)} 点)")
    return ok

def _trajectory_msg_to_14d_chain(traj, base_rad):
    """将 JointTrajectory（可能仅含单臂 7 轴）合并进 14 轴链。"""
    chain = []
    target_rad = np.copy(base_rad)
    for point in traj.points:
        for j, name in enumerate(traj.joint_names):
            if name in joint_names_14:
                target_rad[joint_names_14.index(name)] = point.positions[j]
        chain.append(np.copy(target_rad))
    return chain

def _plan_to_joint_chain(plan, base_rad):
    """将单侧 MoveIt 规划结果合并进 14 轴关节链。"""
    chain = []
    target_rad = np.copy(base_rad)
    for point in plan.joint_trajectory.points:
        for j, name in enumerate(plan.joint_trajectory.joint_names):
            if name in joint_names_14:
                target_rad[joint_names_14.index(name)] = point.positions[j]
        chain.append(np.copy(target_rad))
    return chain

def _merge_dual_arm_chains(chain_l, chain_r, base_rad):
    """将左右臂关节链按时间对齐合并为 14 轴同步轨迹。"""
    n = max(len(chain_l), len(chain_r), 1)
    merged = []
    for i in range(n):
        combined = np.copy(base_rad)
        if chain_l:
            combined[0:7] = chain_l[min(i, len(chain_l) - 1)][0:7]
        if chain_r:
            combined[7:14] = chain_r[min(i, len(chain_r) - 1)][7:14]
        merged.append(combined)
    return merged

def return_both_arms_to_ready(left_arm, right_arm, arm_pub, total_time=4.0, step_name="双臂同步归位"):
    """
    左右手臂同时规划并下发至 left_ready / right_ready。
    程序启动、异常退出、抓取结束均须调用，防止单臂前伸导致重心前倾。
    """
    global last_commanded_joints_rad
    rospy.loginfo(f"\n🛡️ {step_name}：左右手同时退回曲肘护胸...")
    clear_octomap_cache()
    for grp in (left_arm, right_arm):
        grp.set_start_state_to_current_state()
        grp.set_max_velocity_scaling_factor(VELOCITY_SCALING)
        grp.set_max_acceleration_scaling_factor(ACCELERATION_SCALING)
        grp.set_planning_time(8.0)

    left_arm.set_named_target("left_ready")
    right_arm.set_named_target("right_ready")
    ok_l, plan_l, _, _ = left_arm.plan()
    ok_r, plan_r, _, _ = right_arm.plan()

    if (not ok_l or not ok_r or
            len(plan_l.joint_trajectory.points) == 0 or
            len(plan_r.joint_trajectory.points) == 0):
        rospy.logerr("❌ 双臂归位规划失败！请检查 move_group 或人工扶稳机器人！")
        return False

    base = np.copy(current_joints_rad)
    chain_l = _plan_to_joint_chain(plan_l, base)
    chain_r = _plan_to_joint_chain(plan_r, base)
    joint_chain = _merge_dual_arm_chains(chain_l, chain_r, base)
    rospy.loginfo(f"✅ 双臂归位：左 {len(chain_l)} 点 + 右 {len(chain_r)} 点 → 合并 {len(joint_chain)} 点")

    ok = publish_arm_trajectory_batch(arm_pub, joint_chain, total_time)
    if ok:
        wait_joints_settle()
    return ok

# =================================================================
# 🛡️ 终极安全流式泵送引擎 (OMPL 避障规划 + 流式泵送)
# =================================================================
def smart_execute(arm_group, arm_pub, step_name, total_safe_time=3.5):
    rospy.loginfo(f"\n🧠 MoveIt 避障推演开始: [{step_name}]...")
    clear_octomap_cache()
    
    arm_group.set_start_state_to_current_state()
    arm_group.set_max_velocity_scaling_factor(VELOCITY_SCALING)
    arm_group.set_max_acceleration_scaling_factor(ACCELERATION_SCALING)
    arm_group.set_planning_time(8.0)

    success, plan, _, _ = arm_group.plan()

    if not success or len(plan.joint_trajectory.points) == 0:
        rospy.logerr(f"❌ [{step_name}] 路线推演失败！前方路径被硬障碍物截断。")
        rospy.logerr("   提示：确认终端 3 move_group 已启动，且已清除 OctoMap；水瓶点云可能挡住预瞄路径。")
        return False
        
    points = plan.joint_trajectory.points
    rospy.loginfo(f"✅ 几何规划成功！原始 {len(points)} 点，降采样后下发。")

    joint_chain = []
    target_rad = np.copy(last_commanded_joints_rad)
    for point in points:
        for j, name in enumerate(plan.joint_trajectory.joint_names):
            if name in joint_names_14:
                slot = joint_names_14.index(name)
                target_rad[slot] = point.positions[j]
        joint_chain.append(np.copy(target_rad))

    ok = publish_arm_trajectory_batch(arm_pub, joint_chain, total_safe_time)
    if ok:
        wait_joints_settle()
    return ok

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_auto_grasp_master')
    
    rospy.Subscriber('/joint_states', JointState, joint_states_callback)
    rospy.loginfo("⏳ 正在等待真实下位机关节状态对齐...")
    while not has_joint_states and not rospy.is_shutdown():
        rospy.sleep(0.1)
    global last_commanded_joints_rad
    last_commanded_joints_rad = np.copy(current_joints_rad)

    _log_ros_network()

    # === 与 auto_grasp_TF2.py 完全对齐的启动段（先归位+视觉，后加载 MoveIt）===
    try:
        rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)(changeArmCtrlModeRequest(control_mode=2))
    except Exception:
        rospy.logwarn("⚠️ /arm_traj_change_mode 调用失败，继续尝试...")

    arm_pub = rospy.Publisher('/kuavo_arm_target_poses', armTargetPoses, queue_size=10)
    rospy.sleep(0.3)

    print("🖐️ 初始化双爪完全张开...")
    pos, vel, effort = build_open_cmd()
    call_leju_claw(pos, vel, effort, tag="open")
    time.sleep(1.0)

    execute_dual_arm_init_home(arm_pub)

    x_hist, y_hist = _collect_vision_targets_tf2_style()
    if len(x_hist) < 10:
        rospy.logerr("❌ 视觉坐标采集失败")
        execute_dual_arm_init_home(arm_pub)
        return

    rospy.loginfo("✅ 视觉就绪，开始加载 MoveIt 规划器...")
    left_arm = moveit_commander.MoveGroupCommander("left_arm")
    right_arm = moveit_commander.MoveGroupCommander("right_arm")
    for grp in (left_arm, right_arm):
        grp.set_pose_reference_frame("base_link")

    try:
        _run_grasp_sequence(left_arm, right_arm, arm_pub, x_hist, y_hist)
    except Exception as exc:
        rospy.logerr(f"❌ 抓取流程异常: {exc}")
        _safe_return_both_arms(left_arm, right_arm, arm_pub, step_name="异常紧急收手")
        raise

def _run_grasp_sequence(left_arm, right_arm, arm_pub, x_hist, y_hist):
    is_left_arm = np.median(y_hist) > 0.0
    arm = left_arm if is_left_arm else right_arm

    off_x, off_y = tcp_offsets_for_arm(is_left_arm)
    locked_x = np.median(x_hist) + off_x
    locked_y = np.median(y_hist) + off_y
    locked_z = SAFE_LOCKED_Z  

    side = "左" if is_left_arm else "右"
    print(f"\n🎯 TF2 融合打击点: X={locked_x:.3f}, Y={locked_y:.3f}, Z={locked_z:.3f} ({side}手)")

    quat = get_horizontal_claw_quat(locked_x, locked_y, is_left_arm)
    shoulder_x = -0.017
    shoulder_y = 0.292 if is_left_arm else -0.292
    dist = math.hypot(locked_x - shoulder_x, locked_y - shoulder_y)
    if dist <= PRE_GRASP_DIST + 0.01:
        rospy.logerr(f"❌ 目标距肩点仅 {dist*100:.1f}cm，小于预瞄距离 {PRE_GRASP_DIST*100:.0f}cm！")
        _safe_return_both_arms(left_arm, right_arm, arm_pub, step_name="目标过近收手")
        return
    ratio = (dist - PRE_GRASP_DIST) / dist 
    
    pre_x = shoulder_x + (locked_x - shoulder_x) * ratio
    pre_y = shoulder_y + (locked_y - shoulder_y) * ratio
    grasp_pose = _build_pose_stamped(locked_x, locked_y, locked_z, quat)
    pre_pose = _build_pose_stamped(pre_x, pre_y, locked_z, quat)
    lift_pose = _build_pose_stamped(locked_x, locked_y, locked_z + LIFT_HEIGHT, quat)

    _, ee_link = _ik_group_profile(is_left_arm)
    try:
        arm.set_end_effector_link(ee_link)
    except Exception:
        rospy.logwarn(f"⚠️ 无法设置 ee_link={ee_link}，继续尝试 IK")

    ik_client = _resolve_ik_service()
    if ik_client is None:
        _safe_return_both_arms(left_arm, right_arm, arm_pub, step_name="IK服务不可用")
        return

    # === 逆推 IK 验证航点（auto_grasp 同款顺序 + MoveIt TCP/seed）===
    q_grasp = _solve_pose_ik(ik_client, arm, is_left_arm, grasp_pose, last_commanded_joints_rad, f"[{side}手] 终点切入")
    if q_grasp is None:
        _safe_return_both_arms(left_arm, right_arm, arm_pub, step_name="抓握点IK失败")
        return

    q_pre = _solve_pose_ik(ik_client, arm, is_left_arm, pre_pose, q_grasp, f"[{side}手] 退后预瞄 12cm")
    if q_pre is None:
        rospy.logwarn("⚠️ 预瞄点 IK 失败，回退使用抓握点")
        q_pre = q_grasp

    q_lift = _solve_pose_ik(ik_client, arm, is_left_arm, lift_pose, q_grasp, f"[{side}手] 垂直抬升")
    if q_lift is None:
        rospy.logwarn("⚠️ 抬升点 IK 失败，回退使用抓握点")
        q_lift = q_grasp

    print("\n🚀 执行: 曲肘护胸 → 预瞄 → 单段水平插入 → 夹爪 → 抬升 → vla收手")
    print("   （与 auto_grasp_TF2 一致：禁止 init 直跳预瞄，否则关节空间先上抬扫瓶）")

    # auto_grasp_TF2 验证过的构型：从曲肘位再伸向预瞄，避免 init→预瞄 先抬高手肘碰倒瓶
    ready_rad = np.radians(_auto_grasp_ready_deg(is_left_arm))
    execute_single_pose(arm_pub, ready_rad, 2.5, "曲肘护胸（预瞄前安全构型）", is_left_arm)

    q_pre_exec = _solve_pose_ik(
        ik_client, arm, is_left_arm, pre_pose, last_commanded_joints_rad, f"[{side}手] 曲肘后退至预瞄"
    )
    if q_pre_exec is None:
        rospy.logwarn("⚠️ 曲肘后预瞄 IK 失败，回退使用规划阶段解")
        q_pre_exec = q_pre
    execute_single_pose(arm_pub, q_pre_exec, 2.5, "退至预瞄点（肩→瓶直线 12cm 外）", is_left_arm)

    # 预瞄与抓握共线同高：单段 14 轴插值 = auto_grasp 水平笔直插入（不用笛卡尔多点，防连杆上翘）
    execute_single_pose(arm_pub, q_grasp, 1.5, "水平笔直插入抓握点", is_left_arm)

    print("✊ 执行二指夹爪安全闭合...")
    pos, vel, effort = build_close_cmd(is_left_arm)
    call_leju_claw(pos, vel, effort, tag="close")
    time.sleep(2.0) 

    execute_single_pose(arm_pub, q_lift, 2.0, f"垂直拔高 {int(LIFT_HEIGHT*100)}cm", is_left_arm)

    execute_vla_style_return(arm_pub, q_lift, is_left_arm)

    print("👐 抵达安全空域，松开夹爪释放水瓶！大功告成！")
    pos, vel, effort = build_open_cmd()
    call_leju_claw(pos, vel, effort, tag="release")
    try:
        rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)(changeArmCtrlModeRequest(control_mode=0))
    except Exception:
        pass

if __name__ == '__main__':
    main()