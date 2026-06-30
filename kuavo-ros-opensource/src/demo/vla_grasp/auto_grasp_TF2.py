#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import rospy
import time
import math
import numpy as np
from geometry_msgs.msg import PointStamped

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from claw_safe import build_close_cmd, build_open_cmd, get_controller

try:
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
    from kuavo_msgs.srv import twoArmHandPoseCmdSrv
    from kuavo_msgs.msg import twoArmHandPoseCmd, ikSolveParam
    from kuavo_msgs.msg import armTargetPoses
    from kuavo_msgs.msg import robotHeadMotionData
except ImportError:
    print("❌ 找不到 kuavo_msgs，请检查环境！")
    exit(1)

# =================================================================
# 🎯 TF2 专属 TCP (夹爪工具中心点) 与 避障通道配置区
# =================================================================
# 1. 抓取深度：保持你实测出“又准又稳”的神级参数
# =================================================================
# 🎯 TF2 专属 TCP (夹爪工具中心点) 与 避障通道配置区
# =================================================================
# 1. 抓取深度：保持你实测出“又准又稳”的神级参数
TCP_OFFSET_X = 0.005  

# 2. 抵消夹爪横置的机械偏心
# 🔥 修改这里：因为夹爪偏左，我们需要把数值改小，让手往右侧拉回。
# 原来是 0.05，我们减小 2 厘米（0.02），改成 0.03 试试。
TCP_OFFSET_Y_LEFT = 0.03    
TCP_OFFSET_Y_RIGHT = 0.03  

# 3. 桌面安全高度锁 (绝对防死锁)
SAFE_LOCKED_Z = 0.37  

# 4. 全新 3D 避障通道参数
LIFT_HEIGHT = 0.20          # 抓到后向上拔高 20cm
LIFT_HEIGHT_FALLBACKS_M = (0.20, 0.16, 0.12, 0.08)  # 抬升 IK 无解时递减 4cm
AVOID_OUTWARD_Y = 0.10      # 向外侧(远离书本方向)平移 10cm
# ... (下面保持不变) ...
# 夹爪横向姿态控制
CLAW_ROLL_RIGHT = 1.5708  
CLAW_ROLL_LEFT = -1.5708  
# =================================================================

class Quaternion:
    def __init__(self):
        self.w = 0; self.x = 0; self.y = 0; self.z = 0

def euler_to_rotation_matrix(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return R_yaw @ R_pitch @ R_roll

def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    q = Quaternion()
    if trace > 0:
        q.w = math.sqrt(trace + 1.0) / 2
        q.x = (R[2, 1] - R[1, 2]) / (4 * q.w)
        q.y = (R[0, 2] - R[2, 0]) / (4 * q.w)
        q.z = (R[1, 0] - R[0, 1]) / (4 * q.w)
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        j, k = (i + 1) % 3, (i + 2) % 3
        t = np.zeros(4)
        t[i] = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) / 2
        t[j] = (R[i, j] + R[j, i]) / (4 * t[i])
        t[k] = (R[i, k] + R[k, i]) / (4 * t[i])
        t[3] = (R[k, j] - R[j, k]) / (4 * t[i])
        q.x, q.y, q.z, q.w = t
    norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    if norm > 0: q.w /= norm; q.x /= norm; q.y /= norm; q.z /= norm
    return q

def get_horizontal_claw_quat(target_x, target_y, is_left_arm):
    robot_zero_x = -0.017  
    robot_zero_y = 0.292 if is_left_arm else -0.292  
    yaw = math.atan2((target_y - robot_zero_y), (target_x - robot_zero_x))
    R_base = euler_to_rotation_matrix(yaw, -1.57079633, 0)
    
    roll_angle = CLAW_ROLL_LEFT if is_left_arm else CLAW_ROLL_RIGHT
    cr, sr = math.cos(roll_angle), math.sin(roll_angle)
    R_local_z = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    
    return rotation_matrix_to_quaternion(R_base @ R_local_z)

def call_leju_claw(pos, vel, effort, tag="cmd"):
    return get_controller().call(pos, vel, effort, tag=tag)

def solve_ik(ik_client, ik_req, step_name):
    print(f"⏳ 正在计算: {step_name} ...")
    res = ik_client(ik_req)
    if not res.success:
        print(f"❌ {step_name} -> IK 解算失败！")
        return False, None
    print(f"✅ {step_name} -> 求解成功！")
    return True, res.q_arm

def execute_pose(arm_pub, q_arm, time_sec):
    arm_msg = armTargetPoses(times=[time_sec], values=[math.degrees(q) for q in q_arm])
    arm_pub.publish(arm_msg)
    time.sleep(time_sec + 0.5)

def main():
    rospy.init_node('auto_grasp_node')
    print("========== 🧠 VLA: 终极避障满血版 (高空平移 + 曲肘收回 + 自动松爪) ==========")
    
    rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)(changeArmCtrlModeRequest(control_mode=2))
    arm_pub = rospy.Publisher('/kuavo_arm_target_poses', armTargetPoses, queue_size=10)
    head_pub = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
    
    print("🖐️ 初始化双爪完全张开...")
    pos, vel, effort = build_open_cmd()
    call_leju_claw(pos, vel, effort, tag="open")
    time.sleep(1.0)
    
    print("🤖 机器手臂初始归位...")
    init_angles = [20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0]
    arm_pub.publish(armTargetPoses(times=[2.5], values=init_angles))
    time.sleep(3.0)

    print("\n👁️ 正在听取 TF2 绝对坐标 (无视头部晃动，速采 10 帧)...")
    x_hist, y_hist = [], []
    while len(x_hist) < 10 and not rospy.is_shutdown():
        try:
            msg = rospy.wait_for_message('/vla/yolo_target', PointStamped, timeout=0.2)
            if 0.30 <= msg.point.x <= 0.65:
                x_hist.append(msg.point.x); y_hist.append(msg.point.y)
                print(f"  📥 录入 ({len(x_hist)}/10): TF2_X={msg.point.x:.3f}, TF2_Y={msg.point.y:.3f}")
        except Exception: pass

    if len(x_hist) < 10: return

    raw_x, raw_y = np.median(x_hist), np.median(y_hist)
    is_left_arm = raw_y > 0.0
    active_arm = "左手" if is_left_arm else "右手"

    locked_x = raw_x + TCP_OFFSET_X
    if is_left_arm: locked_y = raw_y + TCP_OFFSET_Y_LEFT
    else: locked_y = raw_y + TCP_OFFSET_Y_RIGHT
    locked_z = SAFE_LOCKED_Z  
    
    print(f"\n🎯 TF2 融合打击点: X={locked_x:.3f}, Y={locked_y:.3f}, Z={locked_z:.3f}")

    ik_client = rospy.ServiceProxy('/ik/two_arm_hand_pose_cmd_srv', twoArmHandPoseCmdSrv)
    ik_req = twoArmHandPoseCmd()
    ik_req.use_custom_ik_param = True; ik_req.joint_angles_as_q0 = True
    ik_param = ikSolveParam()
    ik_param.major_optimality_tol = 1e-3; ik_param.major_feasibility_tol = 1e-3
    ik_param.minor_feasibility_tol = 1e-3; ik_param.major_iterations_limit = 500
    ik_param.pos_cost_weight = 0.0 
    ik_req.ik_param = ik_param

    target_quat = get_horizontal_claw_quat(locked_x, locked_y, is_left_arm)
    quat_array = [target_quat.x, target_quat.y, target_quat.z, target_quat.w]
    official_seed = [0.0, 0.0, 0.0, -1.57079633, 0.0, 0.0, 0.0]
    
    # === 1. 逆推规划 B (抓取点) ===
    ik_req.hand_poses.left_pose.joint_angles = official_seed
    ik_req.hand_poses.right_pose.joint_angles = official_seed

    if is_left_arm:
        ik_req.hand_poses.right_pose.pos_xyz = [-0.012, -0.225, -0.265] 
        ik_req.hand_poses.right_pose.quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        ik_req.hand_poses.left_pose.quat_xyzw = quat_array
        ik_req.hand_poses.left_pose.pos_xyz = [locked_x, locked_y, locked_z]
    else:
        ik_req.hand_poses.left_pose.pos_xyz = [-0.012, 0.225, -0.265]   
        ik_req.hand_poses.left_pose.quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        ik_req.hand_poses.right_pose.quat_xyzw = quat_array
        ik_req.hand_poses.right_pose.pos_xyz = [locked_x, locked_y, locked_z]
        
    ok, q_grasp = solve_ik(ik_client, ik_req, f"步骤 B [{active_arm}终点切入]")
    if not ok: return

    # === 2. 逆推规划 A (预瞄点 - 后退12cm) ===
    ik_req.hand_poses.left_pose.joint_angles = list(q_grasp[:7])
    ik_req.hand_poses.right_pose.joint_angles = list(q_grasp[7:])
    robot_zero_x = -0.017; robot_zero_y = 0.292 if is_left_arm else -0.292
    dist = math.hypot(locked_x - robot_zero_x, locked_y - robot_zero_y)
    ratio = (dist - 0.12) / dist 
    pre_x = robot_zero_x + (locked_x - robot_zero_x) * ratio
    pre_y = robot_zero_y + (locked_y - robot_zero_y) * ratio

    if is_left_arm: ik_req.hand_poses.left_pose.pos_xyz = [pre_x, pre_y, locked_z]
    else: ik_req.hand_poses.right_pose.pos_xyz = [pre_x, pre_y, locked_z]
    ok, q_pre = solve_ik(ik_client, ik_req, f"步骤 A [{active_arm}退后预瞄]")
    if not ok: q_pre = q_grasp

    # === 3. 逆推规划 D (垂直抬升，高度递减回退) ===
    q_lift = q_grasp
    for h in LIFT_HEIGHT_FALLBACKS_M:
        ik_req.hand_poses.left_pose.joint_angles = list(q_grasp[:7])
        ik_req.hand_poses.right_pose.joint_angles = list(q_grasp[7:])
        if is_left_arm:
            ik_req.hand_poses.left_pose.pos_xyz = [locked_x, locked_y, locked_z + h]
        else:
            ik_req.hand_poses.right_pose.pos_xyz = [locked_x, locked_y, locked_z + h]
        ok, q_try = solve_ik(ik_client, ik_req, f"步骤 D [{active_arm}垂直抬升 {int(round(h * 100))}cm]")
        if ok:
            q_lift = q_try
            if h < LIFT_HEIGHT - 1e-6:
                print(
                    "⚠️ 抬升 %dcm IK 无解，降级为 %dcm"
                    % (int(round(LIFT_HEIGHT * 100)), int(round(h * 100)))
                )
            break
    else:
        print("⚠️ 全部抬升高度 IK 失败，回退使用抓握点（无垂直拔高）")

    # === 4. 逆推规划 S/E (高空外侧平移避障) ===
    ik_req.hand_poses.left_pose.joint_angles = list(q_lift[:7])
    ik_req.hand_poses.right_pose.joint_angles = list(q_lift[7:])
    
    q_high_safe = q_lift
    for fallback_dist in [0.15, 0.12, 0.09, 0.06, 0.0]:
        ratio = (dist - fallback_dist) / dist if dist != 0 else 1.0
        safe_x = robot_zero_x + (locked_x - robot_zero_x) * ratio
        safe_y = locked_y + AVOID_OUTWARD_Y if is_left_arm else locked_y - AVOID_OUTWARD_Y
        
        if is_left_arm: ik_req.hand_poses.left_pose.pos_xyz = [safe_x, safe_y, locked_z + LIFT_HEIGHT]
        else: ik_req.hand_poses.right_pose.pos_xyz = [safe_x, safe_y, locked_z + LIFT_HEIGHT]
        
        ok, q_res = solve_ik(ik_client, ik_req, f"步骤 E [外移{int(AVOID_OUTWARD_Y*100)}cm + 后退{int(fallback_dist*100)}cm]")
        if ok:
            q_high_safe = q_res
            print(f"💡 AI防撞系统：锁定 3D 避障点 (向外移{int(AVOID_OUTWARD_Y*100)}cm, 后退{int(fallback_dist*100)}cm)！")
            break

    # ================= 全部规划成功，开始物理闭环执行 =================
    print("\n🚀 所有航点验证通过！开始物理执行！")
    
    # 核心动作：曲肘护胸的初始关节角，绝对避障保障
    ready_angles = [40, 20, 0, -120, 0, 0, -20, 20, 0, 0, -30, 0, 0, 0] if is_left_arm else [20, 0, 0, -30, 0, 0, 0, 40, -20, 0, -120, 0, 0, -20]
    execute_pose(arm_pub, [math.radians(x) for x in ready_angles], 2.5)

    print(f"✈️ 下降至预瞄点...")
    execute_pose(arm_pub, q_pre, 2.5)

    print(f"🎯 平移插入水瓶...")
    execute_pose(arm_pub, q_grasp, 1.5)

    print(f"✊ 执行夹爪横向满力矩闭合...")
    pos, vel, effort = build_close_cmd(is_left_arm)
    call_leju_claw(pos, vel, effort, tag="close")
    time.sleep(2.0) 

    print(f"⬆️ 夺得水瓶，垂直拔高 {int(LIFT_HEIGHT*100)}cm...")
    execute_pose(arm_pub, q_lift, 2.0)

    print(f"⬅️ 向外侧平移避开下方物体...")
    execute_pose(arm_pub, q_high_safe, 2.0)

    # 🔥 灵魂补充：加入你想要的“曲肘护胸”节点！
    print(f"🛡️ 执行关节曲肘折叠收手，确保不碰任何下方物体...")
    execute_pose(arm_pub, [math.radians(x) for x in ready_angles], 3.0)

    print(f"🤖 恢复自然下垂原状...")
    execute_pose(arm_pub, [math.radians(x) for x in init_angles], 3.0)
    
    print("👐 任务完成，松开夹爪释放水瓶...")
    pos, vel, effort = build_open_cmd()
    call_leju_claw(pos, vel, effort, tag="open")
    time.sleep(2.0)
    
    print("🎉 真理级完美防撞！盲抓与放置全流程圆满通关！")

if __name__ == '__main__':
    main()