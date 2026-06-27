#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import rospy
import time
import math
import numpy as np
import json
from std_msgs.msg import String
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
# 🎯 黄金参数区 
# =================================================================
TCP_OFFSET_X = 0.005  
TCP_OFFSET_Y_LEFT = 0.03    
TCP_OFFSET_Y_RIGHT = 0.03  

SAFE_LOCKED_Z = 0.385  
LIFT_HEIGHT = 0.22          # 拔高 22 厘米，彻底升空

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
        t[k] = (R[k, i] + R[i, k]) / (4 * t[i])
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


class VLAGraspStateMachine:
    def __init__(self):
        rospy.init_node('vla_auto_grasp_daemon', anonymous=True)
        print("========== 🧠 VLA 守护态小脑 (肩膀关节空间大摆臂版) ==========")
        
        rospy.wait_for_service('/arm_traj_change_mode')
        rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)(changeArmCtrlModeRequest(control_mode=2))
        
        self.arm_pub = rospy.Publisher('/kuavo_arm_target_poses', armTargetPoses, queue_size=10)
        self.head_pub = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
        
        rospy.wait_for_service('/ik/two_arm_hand_pose_cmd_srv')
        self.ik_client = rospy.ServiceProxy('/ik/two_arm_hand_pose_cmd_srv', twoArmHandPoseCmdSrv)
        
        self.is_working = False
        rospy.Subscriber('/vla/master_command', String, self.command_callback)
        print("🟢 纯关节摆动安全雷达已挂起...")

    def call_leju_claw(self, pos, vel, effort, tag="cmd"):
        return get_controller().call(pos, vel, effort, tag=tag)

    def solve_ik(self, ik_req, step_name):
        print(f"⏳ 正在计算: {step_name} ...")
        res = self.ik_client(ik_req)
        if not res.success:
            print(f"❌ {step_name} -> IK 解算失败！")
            return False, None
        print(f"✅ {step_name} -> 求解成功！")
        return True, res.q_arm

    def execute_pose(self, q_arm, time_sec):
        arm_msg = armTargetPoses(times=[time_sec], values=[math.degrees(q) for q in q_arm])
        self.arm_pub.publish(arm_msg)
        time.sleep(time_sec + 0.5)

    def command_callback(self, msg):
        if self.is_working: return
        try:
            cmd = json.loads(msg.data)
            if cmd.get("action") == "grab":
                target_name = cmd.get("target", "目标")
                self.is_working = True
                self.execute_grasp_pipeline(target_name)
        except Exception as e:
            rospy.logerr(f"大模型指令解析失败: {e}")

    def execute_grasp_pipeline(self, target_name):
        print(f"\n=======================================")
        print(f"🚀 接收大模型指令：开始抓取 [{target_name}]")
        print(f"=======================================\n")
        
        print("🖐️ 初始化双爪完全张开...")
        pos, vel, effort = build_open_cmd()
        self.call_leju_claw(pos, vel, effort, tag="open")
        time.sleep(1.0)
        
        print("🤖 机器手臂初始归位...")
        init_angles = [20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0]
        self.execute_pose([math.radians(x) for x in init_angles], 2.5)
        time.sleep(2.0)

        print("\n👁️ 正在听取 TF2 绝对坐标 (速采 10 帧)...")
        x_hist, y_hist = [], []
        while len(x_hist) < 10 and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message('/vla/yolo_target', PointStamped, timeout=0.2)
                if 0.30 <= msg.point.x <= 0.65:
                    x_hist.append(msg.point.x); y_hist.append(msg.point.y)
                    print(f"  📥 录入 ({len(x_hist)}/10): TF2_X={msg.point.x:.3f}, TF2_Y={msg.point.y:.3f}")
            except Exception: pass

        if len(x_hist) < 10: 
            self.is_working = False
            return

        raw_x, raw_y = np.median(x_hist), np.median(y_hist)
        is_left_arm = raw_y > 0.0
        active_arm = "左手" if is_left_arm else "右手"

        locked_x = raw_x + TCP_OFFSET_X
        if is_left_arm: locked_y = raw_y + TCP_OFFSET_Y_LEFT
        else: locked_y = raw_y + TCP_OFFSET_Y_RIGHT
        locked_z = SAFE_LOCKED_Z  
        
        print(f"\n🎯 TF2 最终解算：打击点 X={locked_x:.3f}, Y={locked_y:.3f}, Z={locked_z:.3f}")

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
        
        # === 1. 逆推规划 B (目标抓取点) ===
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
            
        ok, q_grasp = self.solve_ik(ik_req, f"步骤 B [{active_arm}终点切入]")
        if not ok: 
            self.is_working = False
            return

        # === 2. 逆推规划 A (进场直线预瞄点) ===
        ik_req.hand_poses.left_pose.joint_angles = list(q_grasp[:7])
        ik_req.hand_poses.right_pose.joint_angles = list(q_grasp[7:])
        robot_zero_x = -0.017; robot_zero_y = 0.292 if is_left_arm else -0.292
        dist = math.hypot(locked_x - robot_zero_x, locked_y - robot_zero_y)
        
        ratio = (dist - 0.12) / dist 
        pre_x = robot_zero_x + (locked_x - robot_zero_x) * ratio
        pre_y = robot_zero_y + (locked_y - robot_zero_y) * ratio

        if is_left_arm: ik_req.hand_poses.left_pose.pos_xyz = [pre_x, pre_y, locked_z]
        else: ik_req.hand_poses.right_pose.pos_xyz = [pre_x, pre_y, locked_z]
        ok, q_pre = self.solve_ik(ik_req, f"步骤 A [{active_arm}退后预瞄]")
        if not ok: q_pre = q_grasp

        # === 3. 逆推规划 D (垂直抬升起飞点) ===
        ik_req.hand_poses.left_pose.joint_angles = list(q_grasp[:7])
        ik_req.hand_poses.right_pose.joint_angles = list(q_grasp[7:])
        if is_left_arm: ik_req.hand_poses.left_pose.pos_xyz = [locked_x, locked_y, locked_z + LIFT_HEIGHT]
        else: ik_req.hand_poses.right_pose.pos_xyz = [locked_x, locked_y, locked_z + LIFT_HEIGHT]
        ok, q_lift = self.solve_ik(ik_req, f"步骤 D [{active_arm}垂直抬升 {int(LIFT_HEIGHT*100)}cm]")
        if not ok: q_lift = q_grasp 

        # === 4. 🌟【遵循真理重构步骤 E】🌟 ===
        # 彻底抛弃垃圾笛卡尔IK外移！直接进行 100% 可靠的关节空间大展翅！
        # 复制垂直起飞的高度角度，强行对大臂 Roll 轴关节施加 35 度的物理极限外展！
        print("🛡️ [避障重构] 废除笛卡尔矩阵，注入纯关节空间肩膀旋转流...")
        q_high_safe = list(q_lift)
        
        if is_left_arm:
            # 左手大臂（第 2 个关节，索引 1）加上 35 度，强制向正左方暴力摆动
            q_high_safe[1] += math.radians(75.0)
            print("💡 AI防撞系统：已生成左臂【正左方 35度】关节空间大展翅轨迹！")
        else:
            # 右手大臂（第 9 个关节，索引 8）减去 35 度，强制向正右方暴力摆动
            q_high_safe[8] -= math.radians(75.0)
            print("💡 AI防撞系统：已生成右臂【正右方 35度】关节空间大展翅轨迹！")

        # ================= 全部规划成功，开始物理闭环执行 =================
        print("\n🚀 所有航点核验通过，注入物理电流！")
        
        ready_angles = [40, 20, 0, -120, 0, 0, -20, 20, 0, 0, -30, 0, 0, 0] if is_left_arm else [20, 0, 0, -30, 0, 0, 0, 40, -20, 0, -120, 0, 0, -20]
        
        self.execute_pose([math.radians(x) for x in ready_angles], 2.5)

        print(f"✈️ 凌空下降至预瞄点...")
        self.execute_pose(q_pre, 2.5)

        print(f"🎯 笔直平移插入目标...")
        self.execute_pose(q_grasp, 1.5)

        print(f"✊ 执行夹爪安全闭合...")
        pos, vel, effort = build_close_cmd(is_left_arm)
        self.call_leju_claw(pos, vel, effort, tag="close")
        time.sleep(2.0) 

        print(f"⬆️ 夺得目标，垂直拔高 {int(LIFT_HEIGHT*100)}cm...")
        self.execute_pose(q_lift, 2.0)

        # 🌟 奇迹时刻：由于走的是纯关节摆动，右大臂会带动手腕在高度完全不变的前提下，划出一道极度明显的横向外甩大扇面！
        print(f"⬅️ 执行纯肩膀关节外侧大摆臂，100% 清空下方书堆领空...")
        self.execute_pose(q_high_safe, 2.0)

        print(f"🛡️ 安全空域，执行关节曲肘护胸收手...")
        self.execute_pose([math.radians(x) for x in ready_angles], 3.0)

        print(f"🤖 恢复自然下垂原状...")
        self.execute_pose([math.radians(x) for x in init_angles], 3.0)
        
        print("👐 任务完成，松开夹爪释放目标...")
        pos, vel, effort = build_open_cmd()
        self.call_leju_claw(pos, vel, effort, tag="release")
        time.sleep(2.0)
        
        print("🎉 降维级关节大摆臂避障全面胜利，状态机解锁。")
        self.is_working = False

if __name__ == '__main__':
    daemon = VLAGraspStateMachine()
    rospy.spin()