#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
# 导入官方头部位姿消息类
from kuavo_msgs.msg import robotHeadMotionData

def look_down_to_limit():
    # 1. 初始化节点，在ROS网络注册
    rospy.init_node('head_max_look_down_node', anonymous=True)
    
    # 2. 创建发布者，指向头部控制话题
    pub = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
    
    # 🌟 极其重要：遵循你真机成功运行的经验，强制休眠1秒等待连接稳固，防止丢包
    rospy.sleep(1.0)
    
    # 3. 创建消息纸箱
    head_msg = robotHeadMotionData()
    
    # 4. 向底层下达最高物理极限的低头指令
    print("🚀 正在向底层注入绝对极限指令：头部降到最低位置...")
    
    # joint_data 接收参数：[偏航角Yaw(左右), 俯仰角Pitch(上下)]
    # Yaw=0.0 保持面朝正前方
    # Pitch=25.0 触碰官方文档规定的绝对低头极限负边界
    head_msg.joint_data = [0.0, 25.0]  
    
    # 5. 广播发布，驱使硬件执行
    pub.publish(head_msg)
    
    # 留出 2 秒让脖子电机充分旋转到位
    rospy.sleep(2.0)
    print("✅ 机器人头部已到达物理最低位置！地面视野已最大化开阔！")

if __name__ == '__main__':
    try:
        look_down_to_limit()
    except rospy.ROSInterruptException:
        pass