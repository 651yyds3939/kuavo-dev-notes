# v16 修改总结：回归 v2 稳定基线 + 腿部跟踪站姿

## 修改时间
2026-07-13（基于 17 次训练历史的系统性分析）

## 核心诊断

### v15 崩溃的根本原因
通过对比 v2（最佳，mean_reward=83.6，摔倒率1%）和 v15（最新，mean_reward=0.05，摔倒率35.85%），发现：

**v15 一次性改了 5 个关键变量，每个都在削弱稳定性：**

| 变量 | v2（好） | v15（崩） | 影响 |
|------|---------|----------|------|
| **腿部奖励** | legs=5.0（upright门控exp） | legs=0.0 | ⚠️ 删掉了最大的站姿正向吸引子 |
| 手臂门控 | min_upright=0.88 | min_upright=0.75 | 门控放松→歪着也能拿分 |
| 腿约束 | 无 | joint_deviation_legs=-0.5（L1惩罚） | 弱惩罚替代强正奖励 |
| 防漂移 | 无root_xy | root_xy=-2.0 | 新增约束干扰平衡 |
| 阻尼 | rate=-0.004, smooth=-0.006 | rate=-0.01, smooth=-0.01 | 压制手臂动作 |

### 关键洞察

**`track_punch_legs=5.0` 不是"腿部跳舞奖励"，而是"稳定性支柱"**

- CSV 腿部幅度只有 ~0.28 rad，第1帧就是站姿（膝30°踝-17°）
- v2 的 `track_punch_legs`（exp+upright门控）实际上干的是**"奖励腿保持在接近站姿的位置"**
- 这是个**形状良好、有界[0,1]、被直立门控的正向吸引子**，PPO 特别吃这套
- v15 把它换成 `joint_deviation_legs`（L1 惩罚，-0.5）：
  - 无界、负向、无门控
  - 机器人失去"站好了就发钱"的正反馈
  - 只剩"动了就扣钱"→ PPO 找到的最优解崩了

## 解决方案

### 设计哲学
**一次只动一个自由度 + 保留成功的奖励结构**

v16 的策略：
1. **100% 复刻 v2 的稳定性配置**（flat=-12, squat=-15, height=-8, 阻尼-0.004/-0.006）
2. **保留 v2 的奖励结构**（exp + upright门控 + 权重5.0），不用负向惩罚
3. **唯一实质改动**：把"腿跟 CSV 跳舞"换成"腿跟固定站姿"

### 新增奖励函数

在 `rewards.py` 添加：

```python
def track_leg_standing_upright_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.30,
    min_upright: float = 0.85,
    min_height: float | None = 0.72,
    target_height: float = 0.87,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励腿部保持自然站姿(膝30°踝-17°)，结构同 v2 的 track_punch_legs
    但目标锁定站姿而非 CSV。exp+门控 → 正向吸引子，兼顾防跪/防勾脚。
    
    Critical insight from v2/v15 comparison: This is NOT just a constraint, it's the
    stability pillar. The exp+gate structure creates a bounded [0,1] attractor that
    PPO loves, identical to v2's track_punch_legs but targeting fixed standing pose
    instead of CSV dancing legs.
    
    This simultaneously:
    1. Inherits v2's stability (same reward structure)
    2. Keeps legs still (target = standing pose, not CSV)
    3. Prevents hook-foot (ankle target = -0.30 rad = natural flat)
    """
```

**为什么这个能同时解决站稳、腿不动、防勾脚？**

- **站稳**：继承 v2 的 exp+gate 结构，权重 5.0，和手臂奖励同级
- **腿不动**：目标是固定站姿（膝30°踝-17°），不是 CSV 的跳舞腿
- **防勾脚**：站姿向量里踝=-0.30（自然平放），正向奖励会把踝拉到 -0.30，比负向惩罚 `penalty_foot_pitch` 温和得多（v8/v9 用惩罚全崩了）

## 配置修改详情

### 奖励权重（punch_env_cfg.py）

| 项 | v15 | v16 | 原因 |
|----|-----|-----|------|
| **track_leg_standing** | 无 | **5.0** | 新增，接管 v2 的稳定性支柱位置 |
| track_punch_arms | 15.0, gate=0.75 | **12.0, gate=0.85** | 回 v2 权重，收紧门控 |
| track_punch_legs | 0.0 | 0.0 | 保持关闭 |
| joint_deviation_legs | -0.5 | **0.0** | 关掉负向惩罚，交给正向奖励 |
| flat_orientation_l2 | -12.0 | -12.0 | v2 值，不变 |
| base_height | -8.0 | -8.0 | v2 值，不变 |
| penalty_root_squat | -15.0 | -15.0 | v2 值，不变 |
| action_rate_l2 | -0.01 | **-0.004** | 回 v2 轻档，不压手臂 |
| action_smoothness_l2 | -0.01 | **-0.006** | 回 v2 轻档 |
| base_lin_vel_xy_stationary | -1.0 | **-1.3** | 回 v2 值 |
| penalty_root_xy_displacement | -2.0 | **0.0** | v2 没有，先关掉排除干扰 |

### track_leg_standing 参数解析

```python
track_leg_standing = RewTerm(
    func=local_rewards.track_leg_standing_upright_exp,
    weight=5.0,  # v2 的 track_punch_legs 权重，保持稳定性支柱强度
    params={
        "std": 0.30,          # 站姿容差（CSV腿幅 ~0.28 rad，std=0.30 匹配）
        "min_upright": 0.85,  # 门控阈值（站稳才拿分，防躺平刷分）
        "min_height": 0.72,   # 高度门控（防跪着站姿）
        "target_height": 0.87,
    },
)
```

## 预期效果

基于 v2 的表现（mean_reward=83.6，摔倒率1%，arms=8.68/12），v16 应该达到：

1. **站稳**：摔倒率 < 5%（v2 是 1%，允许一些回归但不能崩）
2. **手臂跟踪**：arms ≈ 8-10（v2 是 8.68，权重从 12→12 不变）
3. **腿保持站姿**：腿不跳舞，保持膝30°踝-17°
4. **勾脚缓解**：踝关节被正向吸引到 -0.30（自然平放），不再像 v2 那样勾到 -0.17 以上

### 如果 v16 还有勾脚

下一版（v17）单独试 `penalty_foot_pitch=-1.0`：
- 比失败的 v8 的 -2 更轻
- 此时已有 track_leg_standing 站姿正奖励托底
- 不会像 v9（-8）那样出生就死

## 吸取的核心教训

从 17 次训练历史中提炼的原则：

1. **门控型奖励需要极稳底座**  
   手臂奖励被 upright gate 锁住，必须先站起来才能拿分。flat/squat 必须强到能独立引导站立（v2 的 -12/-15 才行，-5/-8 不行）。

2. **正向吸引子 > 负向惩罚**  
   exp+gate 的有界 [0,1] 奖励比无界负向惩罚更利于 PPO 收敛。v2 用 track_punch_legs=5.0（正向），v15 换成 joint_deviation_legs=-0.5（负向）直接崩盘。

3. **奖励权重之间不是线性关系**  
   flat 从 -12 降到 -5 不是"宽松一点"，而是彻底改变最优解结构。一次改多个变量无法定位问题（v15 改了 5 个全崩）。

4. **腿部微调是平衡必需**  
   人形机器人手臂运动需要腿部配合平衡。腿锁太死（-5.0）阻止补偿动作，手臂一动就倒。

5. **脚下约束是双刃剑**  
   v2 无脚下约束能站稳但勾脚。加 foot_pitch/foot_link_flat 太重（v9，-4/-8）直接致死，太轻（v8，-2）不起作用。正确方式是用站姿正向奖励引导，而非硬约束。

## 训练命令

```bash
cd ~/kuavo_all/leju_robot_rl && conda activate isaaclab
bash scripts/tools/run_s49_lafan1_retrain.sh
```

训练 2000 iterations 后检查：
- 摔倒率是否 < 5%？
- 手臂是否跟踪 CSV？
- 腿是否保持站姿（膝30°）？
- 脚踝是否平放（不勾脚）？

## 修改文件清单

1. `exts/ext_template/ext_template/tasks/locomotion/velocity/mdp/rewards.py`
   - 新增 `track_leg_standing_upright_exp()` 函数

2. `exts/ext_template/ext_template/tasks/locomotion/velocity/config/s49/punch_env_cfg.py`
   - 修改 `RewardsCfg`：添加 track_leg_standing，调整所有权重回 v2 基线
   - 更新版本注释为 v16

3. `scripts/tools/run_s49_lafan1_retrain.sh`
   - 更新脚本头注释和 echo 输出为 v16

## 下一步（如果 v16 成功）

部署到真机验证：
1. 导出 policy（ONNX 或 JIT）
2. 在 Kuavo S49 实机上运行
3. 观察真实环境下的站稳、手臂跟踪、腿部表现
4. 如有 sim2real gap，回头微调奖励权重或域随机化参数

## 下一步（如果 v16 仍有问题）

按优先级排查：
1. **摔倒率高**（>10%）→ 加强稳定性（flat/squat 权重，或检查 Spawn 姿态）
2. **手臂不动**（arms < 5）→ 降低阻尼或提高手臂权重
3. **腿乱动**（偏离站姿）→ 提高 track_leg_standing 权重或降低 std
4. **仍然勾脚**（踝角 > -0.1）→ 单独加 penalty_foot_pitch=-1.0

**关键原则：一次只改一个变量，对照组清晰。**