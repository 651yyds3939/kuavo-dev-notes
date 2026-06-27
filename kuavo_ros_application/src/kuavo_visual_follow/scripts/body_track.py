#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project : Kuavo 3D Head-Body Synergistic Tracker
@File    : body_track.py
@Desc    : 夸父人形机器人终极头身协同大脑（已封入三维物理真理配平与动态安全锁）
@Warning : 部署点火前，请务必确认底盘已硬化进入 Walk 状态，且安全龙门架绳索已挂载微松！
"""

import rospy
from kuavo_msgs.msg import robotHeadMotionData
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist 

class HeadTracker:
    def __init__(self):
        # 声明系统注册节点名
        rospy.init_node('head_tracking_controller')
        
        # ========================================================
        # 👑 【核心安全锁 2】动作使能物理开关
        # 必须通过终端指令：rosparam set /head_tracking_controller/movement_enabled true 激活
        # 否则该节点只做纯净配平下发与云台转动，底盘对一切视觉追随保持绝对静止
        # ========================================================
        self.movement_enabled = rospy.get_param('~movement_enabled', False)
        
        # 1. 注册发布者与订阅者总线链路
        self.head_pub = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.subscriber = rospy.Subscriber('/visual_follow/target_2d', Detection2DArray, self.vision_callback)
        
        # 2. 图像坐标系绝对锚点与云台控制映射参数
        self.center_x = 640.0         
        self.center_y = 400.0         
        self.kp_yaw = 0.01            
        self.kp_pitch = -0.01         
        
        # 系统状态机：存储当前的位姿期望
        self.current_yaw = 0.0
        self.current_pitch = 0.0

        # ==================================
        # 🛡️ 底盘协同动力学参数 (极致柔和限速版)
        # ==================================
        self.kp_body_yaw = 0.01      # 极低转速系数，确保底盘发力时极度平缓不丢重
        self.body_deadzone = 15.0     # 头身协同惰性死区：头部偏转超过 15 度才触发底盘代偿
        self.max_angular_z = 0.2      # 物理绝对限幅阀：最大旋转速度被锁死在约 10.4度/秒

        # ==================================
        # 👑 实机三维物理极值配平 (Trim 真理库)
        # 将我们使用 trim_test.py 测出的真理值硬编码固化
        # ==================================
        self.trim_linear_x = -0.008   # (真理1) 抵消前冲滑移漂移
        self.trim_linear_y = 0.001    # (真理2) 抵消向右侧滑
        self.trim_angular_z = -0.002  # (真理3) 抵消向左内旋倾向

    def vision_callback(self, msg):
        """
        视觉流高频回调函数，心跳级别处理张量数据与动力学转化
        """
        # ---------------------------------------------------------
        # 【核心安全锁 1】：视觉空窗期的配平保活防宕机机制
        # 当镜头暂时无人时，立刻终止动态计算，仅下发纯净物理抗漂移速度
        # ---------------------------------------------------------
        if len(msg.detections) == 0:
            if self.movement_enabled:
                self.publish_trim_velocity()
            return
            
        target = msg.detections[0]
        error_x = target.bbox.center.x - self.center_x
        error_y = target.bbox.center.y - self.center_y
        
        # ====================
        # A. 头部云台独立解算层
        # ====================
        # 30 像素死区硬过滤
        if abs(error_x) >= 30: 
            self.current_yaw += error_x * self.kp_yaw
        if abs(error_y) >= 30: 
            self.current_pitch += error_y * self.kp_pitch
            
        # 云台软限幅刀砍截断
        self.current_yaw = max(-30.0, min(30.0, self.current_yaw))
        self.current_pitch = max(-25.0, min(25.0, self.current_pitch))
        
        # 发布云台指令
        head_msg = robotHeadMotionData()
        head_msg.joint_data = [self.current_yaw, self.current_pitch]
        self.head_pub.publish(head_msg)

        # ====================
        # B. 底盘动力学高级协同层
        # ====================
        # 仅当操作员显式解除使能锁时，才准许向下发散底盘扭矩计算指令
        if self.movement_enabled:
            cmd_msg = Twist()
            # 强行注入 X 和 Y 的防溜车三维真理配平值
            cmd_msg.linear.x = self.trim_linear_x
            cmd_msg.linear.y = self.trim_linear_y
            
            # 最终下发旋转角速度 = 代偿速度基数 + 自转偏置配平
            cmd_msg.angular.z = self.calculate_base_angular_z() + self.trim_angular_z
            
            # 底盘保命截断阀
            cmd_msg.angular.z = max(-self.max_angular_z, min(self.max_angular_z, cmd_msg.angular.z))
            
            self.cmd_pub.publish(cmd_msg)

    def calculate_base_angular_z(self):
        """
        阶梯状次级触发链算法：计算底盘的协同拖拽代偿角速度
        """
        if self.current_yaw > self.body_deadzone:
            return (self.current_yaw - self.body_deadzone) * self.kp_body_yaw
        elif self.current_yaw < -self.body_deadzone:
            return (self.current_yaw + self.body_deadzone) * self.kp_body_yaw
        
        # 目标仍处于中心舒适视场内，底盘计算零指令
        return 0.0

    def publish_trim_velocity(self):
        """
        纯净物理配平保活函数：用于视觉丢失时的心跳维持
        """
        msg = Twist()
        msg.linear.x = self.trim_linear_x
        msg.linear.y = self.trim_linear_y
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = self.trim_angular_z
        self.cmd_pub.publish(msg)

if __name__ == '__main__':
    try:
        tracker = HeadTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
