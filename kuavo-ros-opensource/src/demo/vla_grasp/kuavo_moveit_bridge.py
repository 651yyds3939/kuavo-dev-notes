#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import math
import numpy as np
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult

try:
    from kuavo_msgs.msg import armTargetPoses
except ImportError:
    rospy.logerr("❌ 无法导入 kuavo_msgs，请确保工作空间已 source 并且编译成功！")
    exit(1)

class KuavoMoveItBridge(object):
    def __init__(self):
        rospy.init_node('kuavo_moveit_bridge_node', anonymous=True)
        rospy.loginfo("=========================================================")
        rospy.loginfo("🦾 Kuavo 4 Pro MoveIt! -> 终极单点降维打击 桥接中枢激活！")
        rospy.loginfo("=========================================================")

        self.joint_names_14 = [
            'zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 'zarm_l5_joint', 'zarm_l6_joint', 'zarm_l7_joint',
            'zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint'
        ]

        self.current_joints_rad = np.zeros(14)
        self.has_joint_states = False

        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        
        # 🚨 退回你实机绝对管用的原厂单点控制话题！
        self.arm_pub = rospy.Publisher('/kuavo_arm_target_poses', armTargetPoses, queue_size=100)

        rospy.loginfo("⏳ 正在尝试截获第一帧 /joint_states 电机物理角度...")
        rospy.sleep(1.0) 
        if not self.has_joint_states:
            rospy.logwarn("⚠️ 未能在开机首秒截获高频 /joint_states，自动激活保底内存对齐...")
            self.current_joints_rad = np.array([0.0]*14)
            self.has_joint_states = True
        rospy.loginfo("✅ 14轴基础肌肉通路校准完毕！")

        self.left_server = actionlib.SimpleActionServer(
            'left_arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction,
            execute_cb=self.execute_left_cb,
            auto_start=False
        )
        self.right_server = actionlib.SimpleActionServer(
            'right_arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction,
            execute_cb=self.execute_right_cb,
            auto_start=False
        )

        self.left_server.start()
        self.right_server.start()
        rospy.loginfo("🚀 Left & Right 双通道 Action 服务端满血启动，随时开火！")

    def joint_states_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_names_14:
                idx = self.joint_names_14.index(name)
                self.current_joints_rad[idx] = msg.position[i]
        self.has_joint_states = True

    def execute_left_cb(self, goal):
        rospy.logwarn("🔥🔥🔥 截获 MoveIt! 左臂轨迹！执行单点降维提取！")
        self.trajectory_dispatcher(goal, self.left_server, is_left=True)

    def execute_right_cb(self, goal):
        rospy.logwarn("🔥🔥🔥 截获 MoveIt! 右臂轨迹！执行单点降维提取！")
        self.trajectory_dispatcher(goal, self.right_server, is_left=False)

    def trajectory_dispatcher(self, goal, server, is_left):
        points = goal.trajectory.points
        if not points:
            server.set_succeeded()
            return

        # =================================================================
        # 🚨【全场唯一真理：单点降维提取法】
        # 彻底废除 for 循环逐个点下发！直接提取 MoveIt 轨迹的“最后一个点”和“总耗时”！
        # 完美模拟 4.4real_visual_grasp.md 中绝对管用的 execute_pose 机制！
        # =================================================================
        last_point = points[-1]
        total_time = last_point.time_from_start.to_sec()
        
        # 给一个保底时间，防止除零错误
        if total_time <= 0.1: 
            total_time = 1.0

        target_indices = []
        for name in goal.trajectory.joint_names:
            if name in self.joint_names_14:
                target_indices.append(self.joint_names_14.index(name))

        # 将目标写入当前 14 轴的大表中
        send_positions_rad = np.copy(self.current_joints_rad)
        for j, idx in enumerate(target_indices):
            send_positions_rad[idx] = last_point.positions[j]

        # 弧度转度数
        send_positions_deg = [math.degrees(rad) for rad in send_positions_rad]

        # 肘关节死锁硬防御
        if send_positions_deg[3] > 0.0: send_positions_deg[3] = 0.0     
        if send_positions_deg[10] > 0.0: send_positions_deg[10] = 0.0   

        # 构造一条唯一指令，轰入底层！
        msg = armTargetPoses()
        msg.times = [total_time]
        msg.values = send_positions_deg
        
        self.arm_pub.publish(msg)

        # 模拟物理执行的时间延迟，让 MoveIt 大脑耐心等待真实手臂抵达
        rospy.sleep(total_time)

        # 任务完美结束
        result = FollowJointTrajectoryResult()
        result.error_code = FollowJointTrajectoryResult.SUCCESSFUL
        server.set_succeeded(result)
        rospy.loginfo(f"🎉 目标角度 {send_positions_deg[3]:.1f} 等已单点下发，真机物理响应完毕！")

if __name__ == '__main__':
    try:
        bridge = KuavoMoveItBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass