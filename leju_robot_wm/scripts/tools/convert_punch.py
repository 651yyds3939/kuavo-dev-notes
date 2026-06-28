import numpy as np
import pandas as pd

# ==========================================
# 1. 基础参数定义
# ==========================================
target_hz = 50        # 训练环境的控制频率 (50Hz，即每步 0.02 秒)
duration = 10.0       # 动作总时长：10秒
num_frames = int(duration * target_hz) # 总帧数：500帧
num_joints = 26       # 夸父 4 Pro 的 26 个自由度

# 初始化一个全零的参考动作矩阵，形状为 (500, 26)
# 每一行代表 0.02 秒时，26个关节的期望弧度
reference_motions = np.zeros((num_frames, num_joints))

# ==========================================
# 2. 核心保命线：硬编码锁死下半身 (0 - 11 索引)
# 使用第4篇笔记中真机平稳直立/微蹲的黄金物理真值
# ==========================================
# 0-5 号为左腿，6-11 号为右腿
standing_legs = np.array([
    -0.01867,  # 0: leg_l1_joint (侧摆)
    -0.00196,  # 1: leg_l2_joint (航向)
    -0.376,    # 2: leg_l3_joint (大腿俯仰)
     0.676,    # 3: leg_l4_joint (膝关节俯仰)
    -0.352,    # 4: leg_l5_joint (踝关节俯仰)
    -0.0198,   # 5: leg_l6_joint (踝关节侧摆)
    -0.0160,   # 6: leg_r1_joint
    -0.0008,   # 7: leg_r2_joint
    -0.376,    # 8: leg_r3_joint
     0.676,    # 9: leg_r4_joint
    -0.352,    # 10: leg_r5_joint
     0.0160    # 11: leg_r6_joint
])

# 将 500 帧里腿部关节全部用静态直立角度填满，确保训练和运行中下半身稳如泰山
for f in range(num_frames):
    reference_motions[f, 0:12] = standing_legs

# ==========================================
# 3. 注入上半身打拳动作 (12 - 25 索引)
# 规律：12-18 是左臂，19-25 是右臂
# ==========================================
time_axis = np.linspace(0, duration, num_frames)

for f in range(num_frames):
    t = time_axis[f]
    
    # 设计一个优雅的交替出拳正弦波 (频率为 1Hz，即每秒打一拳)
    # 假设 zarm_l4_joint 和 zarm_r4_joint 是左右肘关节俯仰 (Elbow Pitch)
    # 假设 zarm_l1_joint 和 zarm_r1_joint 是左右肩关节俯仰 (Shoulder Pitch)
    
    # 左臂出拳逻辑 (0-5秒内出拳，后面收回)
    if t < 5.0:
        # 肩关节前举，肘关节伸直
        reference_motions[f, 12] = 0.5 * np.sin(np.pi * t)  # 左肩前举
        reference_motions[f, 15] = -0.6 * np.sin(np.pi * t) # 左肘伸直
    else:
        reference_motions[f, 12] = 0.0 # 收回待机
        reference_motions[f, 15] = -0.2
        
    # 右臂出拳逻辑 (5-10秒内出拳)
    if t >= 5.0:
        reference_motions[f, 19] = 0.5 * np.sin(np.pi * (t - 5.0))  # 右肩前举
        reference_motions[f, 22] = -0.6 * np.sin(np.pi * (t - 5.0)) # 右肘伸直
    else:
        reference_motions[f, 19] = 0.0 # 收回待机
        reference_motions[f, 22] = -0.2

# ==========================================
# 4. 导出为乐聚标准格式的参考轨迹文件
# ==========================================
# 如果官方最新格式需要 CSV，我们直接带时间戳导出
columns = ["time"] + [f"joint_{i}" for i in range(num_joints)]
time_col = time_axis.reshape(-1, 1)
data_to_save = np.hstack((time_col, reference_motions))

df = pd.DataFrame(data_to_save, columns=columns)
df.to_csv("kuavo_punch_ref.csv", index=False)
print("🔥 恭喜！高安全系数的『下肢锁死、上肢打拳』标准参考 CSV 已生成完毕！")
