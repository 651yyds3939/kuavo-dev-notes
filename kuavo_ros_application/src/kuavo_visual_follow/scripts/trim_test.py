#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project : Kuavo Dynamic Physics Calibrator
@File    : trim_test.py (Universal Template)
@Desc    : 夸父底盘三维防漂移降维探测工具 (当前代码状态：Y轴横向滑移专测版)
@Warning : 运行前必须通过状态机确保机器人已在终端切入 walk 踏步状态！
"""

import rospy
from geometry_msgs.msg import Twist
import threading
import sys

# 全局变量：存储通过终端控制台动态注入的当前测试速度值
current_test_val = 0.0

def publish_velocity():
    """
    后台保活守护线程：
    以 20Hz 的高频率向底层不断灌注 Twist 指令。
    (双足 MPC 看门狗机制要求必须有连续的速度心跳注入，否则会报通信超时并触发瘫痪急停)
    """
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(20) # 20Hz 刷新率
    
    while not rospy.is_shutdown():
        msg = Twist()
        
        # ==========================================
        # 👑 已锁死的完美真理配平值 (控制变量法)
        # ==========================================
        # [阶段1结果固化] 锁死前后前冲公差
        msg.linear.x = -0.008   
        
        # [阶段2结果固化] 锁死左转偏航公差
        msg.angular.z = -0.002  
        
        # ==========================================
        # 🎯 当前探测维度接入
        # [阶段3进行中] 动态接收控制台输入，探测横向侧滑
        # ==========================================
        msg.linear.y = current_test_val
        
        # 其余不相关维度强行置零，杜绝动力学干扰
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        
        pub.publish(msg)
        rate.sleep()

def main():
    global current_test_val
    rospy.init_node('trim_tester_node', anonymous=True)

    # 将高频发布挂载为底层守护线程启动，绝对不阻碍主线程的交互 IO
    pub_thread = threading.Thread(target=publish_velocity)
    pub_thread.daemon = True
    pub_thread.start()

    # 极客风格控制台 UI
    print("\n" + "="*55)
    print(" 🤖 夸父底盘防漂移（多轴微调）硬核探测仪 ")
    print("="*55)
    print(" ⚠️ 前置要求：务必确认底盘已处于 walk 均匀踏步状态！")
    print(" 🎯 已锁定先验变量: linear.x = -0.008, angular.z = -0.002")
    print(" 💡 当前探测 [Y轴]: 建议值 0.001, 0.003 (输入正数即施加向左的反拉力)")
    print(" 🛑 安全退出：随时盲打 'q' 切断指令并退出。")
    print("="*55 + "\n")

    while not rospy.is_shutdown():
        try:
            # 跨版本 Python IO 兼容处理 (适配 ROS1 内置的 Python 2/3 混用环境)
            if sys.version_id[0] == 2:
                user_input = raw_input("当前测试速度 [{:.3f}] -> 键入新测试值: ".format(current_test_val))
            else:
                user_input = input("当前测试速度 [{:.3f}] -> 键入新测试值: ".format(current_test_val))
            
            # 优雅退出捕捉
            if user_input.lower() == 'q':
                print(">>> 探测器终止，安全切断底层数据流...")
                break
            
            new_vel = float(user_input)
            
            # ==========================================
            # 🛡️ 工业级防爆输入强制截断阀
            # 彻底杜绝车间环境键盘多按(如敲成 1.0 m/s)导致的暴冲死亡指令
            # ==========================================
            if new_vel > 0.05 or new_vel < -0.05:
                print(" ❌ [高危告警] 配平速度过大！为防止物理劈叉，已强制实施 ±0.05 m/s 物理安全截断。")
                new_vel = max(-0.05, min(0.05, new_vel))
            
            # 原子级更新全局测试变量，立刻在下一帧心跳包中生效
            current_test_val = new_vel
            print(" ✅ 已向底层总线注入新指令: {:.3f}\n".format(current_test_val))

        except ValueError:
            print(" ❌ [语法错误] 无效的浮点数据类型！请明确输入数值（例：0.001）。\n")
        except KeyboardInterrupt:
            print("\n>>> 收到硬件中断信号，紧急挂起探测器...")
            break

if __name__ == '__main__':
    sys.version_id = sys.version_info
    main()
