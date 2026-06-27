#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import JointState
try:
    from kuavo_msgs.msg import sensorsData
except ImportError:
    rospy.logerr("❌ 无法导入 kuavo_msgs，请确保工作空间已 source 并且编译成功！")
    exit(1)

class KuavoStatePublisher(object):
    def __init__(self):
        rospy.init_node('kuavo_state_publisher_bridge', anonymous=True)
        rospy.loginfo("=========================================================")
        rospy.loginfo("👁️ Kuavo 全身 28 轴状态引渡神经元激活：/sensors_data_raw -> /joint_states")
        rospy.loginfo("=========================================================")

        # 🚨【满血升级】：完美对齐 URDF 里面涵盖全身的 28 个标准关节名称
        self.joint_names = [
            # 1. 左下肢 6 个关节 (0~5)
            'leg_l1_joint', 'leg_l2_joint', 'leg_l3_joint', 'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint',
            # 2. 右下肢 6 个关节 (6~11)
            'leg_r1_joint', 'leg_r2_joint', 'leg_r3_joint', 'leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint',
            # 3. 左手臂 7 个关节 (12~18)
            'zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 'zarm_l5_joint', 'zarm_l6_joint', 'zarm_l7_joint',
            # 4. 右手臂 7 个关节 (19~25)
            'zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint',
            # 5. 头部 2 个关节 (26~27)
            'zhead_1_joint', 'zhead_2_joint'
        ]

        # 建立标准状态发布者
        self.joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

        # 订阅乐聚真机 WBC 驱动原始传感器话题
        rospy.Subscriber('/sensors_data_raw', sensorsData, self.sensor_callback)

    def sensor_callback(self, data):
        # 🚨【安全防御】：乐聚官方数据总长必须为 28 轴，不够则直接拦截防止越界崩溃
        if not hasattr(data.joint_data, 'joint_q') or len(data.joint_data.joint_q) < 28:
            return

        # 实例化 ROS 官方标准关节消息
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = self.joint_names

        # 🚨【核心升级】：因为原厂的排布顺序（下肢->手臂->头部）与我们上面定义的完全 1:1 对齐
        # 咱们直接一把全量整包转换，免去切片碎裂的风险！
        joint_state_msg.position = list(data.joint_data.joint_q)
        joint_state_msg.velocity = list(data.joint_data.joint_v)

        # 喷涌推送
        self.joint_state_pub.publish(joint_state_msg)

if __name__ == '__main__':
    try:
        publisher = KuavoStatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass