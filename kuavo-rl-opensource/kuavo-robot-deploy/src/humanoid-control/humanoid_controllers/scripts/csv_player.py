#!/usr/bin/env python3
import rospy
import csv  # 纯原生组件，零依赖
from sensor_msgs.msg import JointState

def play_csv():
    rospy.init_node('csv_motion_player', anonymous=True)
    
    # 🟢 频道对齐：精准投递到官方底层接收的 /joint_cmd 接口
    pub = rospy.Publisher('/joint_cmd', JointState, queue_size=10)
    
    joint_data = []
    csv_path = "/root/kuavo_ws/kuavo_punch_ref.csv"
    
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                joint_data.append([float(x) for x in row[1:]])
    except FileNotFoundError:
        rospy.logerr(f"❌ 未找到参考轨迹文件，请确保它在容器根目录下: {csv_path}")
        return
        
    rate = rospy.Rate(50) # 50Hz 频率播放
    frame = 0
    num_frames = len(joint_data)

    # 🟢 终极破案核心：必须一字不差地注入夸父 26 自由度标准 URDF 关节链名字
    # 顺序严格契合 flat_csp.info 字典的 0 - 25 号索引排布
    joint_names = [
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint"
    ]

    rospy.loginfo(f"🎬 夸父 26 维基因对齐放映机点火！总计 {num_frames} 帧，正在高频推送控制流...")
    
    while not rospy.is_shutdown():
        current_joints = joint_data[frame]
        
        # 封装标准姿态包
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_names       # 🟢 注入关节神经名字映射！
        msg.position = current_joints # 注入 26 维弧度位置
        
        pub.publish(msg)
        
        frame = (frame + 1) % num_frames # 循环回放
        rate.sleep()

if __name__ == '__main__':
    try:
        play_csv()
    except rospy.ROSInterruptException:
        pass