#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project : Kuavo Visual Tracker (Gen 1)
@File    : head_track.py
@Desc    : 第一代原型机：仅包含二维像素向三维颈部云台的物理映射闭环。底盘绝对锁死。
"""

import rospy
from kuavo_msgs.msg import robotHeadMotionData
from vision_msgs.msg import Detection2DArray

class HeadTrackerGen1:
    def __init__(self):
        rospy.init_node('head_tracking_controller_gen1')
        
        # 仅注册头部关节发布者
        self.head_pub = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
        # 订阅 YOLO 二维识别边界框
        rospy.Subscriber('/visual_follow/target_2d', Detection2DArray, self.vision_callback)
        
        # 绝对视觉中心锚点 (1280x800)
        self.center_x = 640.0
        self.center_y = 400.0
        
        # 头部 P 控制器增益 (kp_pitch 设负反转坐标系)
        self.kp_yaw = 0.01   
        self.kp_pitch = -0.01  
        
        # 内部状态机维护
        self.current_yaw = 0.0
        self.current_pitch = 0.0

    def vision_callback(self, msg):
        # 视野丢失直接抛弃，不做任何处理
        if len(msg.detections) == 0:
            return
            
        target = msg.detections[0]
        error_x = target.bbox.center.x - self.center_x
        error_y = target.bbox.center.y - self.center_y
        
        # 30 像素物理死区防高频抽搐过滤
        if abs(error_x) >= 30:
            self.current_yaw += error_x * self.kp_yaw
        if abs(error_y) >= 30:
            self.current_pitch += error_y * self.kp_pitch
        
        # 颈部碳纤维连杆机械防破坏：绝对软限幅
        self.current_yaw = max(-30.0, min(30.0, self.current_yaw))
        self.current_pitch = max(-25.0, min(25.0, self.current_pitch))
        
        # 封包下发硬件驱动
        head_msg = robotHeadMotionData()
        head_msg.joint_data = [self.current_yaw, self.current_pitch]
        self.head_pub.publish(head_msg)

if __name__ == '__main__':
    try:
        tracker = HeadTrackerGen1()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
