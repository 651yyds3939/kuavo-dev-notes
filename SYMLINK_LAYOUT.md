# 工作空间软链接拓扑与维护手册

> **给 GitHub 访客：** 本文档描述的是**作者个人电脑**上的目录布局（`~/Notes/kuavo-dev-notes`、`~/kuavo_all/_training_logs` 等）。  
> **克隆本仓库阅读笔记与代码时，可以整篇跳过**；不必按此配置软链接。  
> 配套脚本 [`automove_and_link.sh`](./automove_and_link.sh) 同样仅服务于作者本机维护，**非必读；克隆本仓库时可跳过**。

> **作者自用一句话：** 源码在 `~/Notes/kuavo-dev-notes`，训练 log/video 在 `~/kuavo_all/_training_logs`，终端从 `~/kuavo_all` 进入。自动化请用脚本，不要手抄命令。


---

## 1. 三层拓扑（不要只挪其中一层）

```text
~/kuavo_all/                              ← 终端工作入口
├── leju_robot_rl ──────────链──→ ~/Notes/kuavo-dev-notes/leju_robot_rl/
├── leju_robot_wm ──────────链──→ ~/Notes/kuavo-dev-notes/leju_robot_wm/
├── kuavo-rl-opensource ────链──→ ~/Notes/kuavo-dev-notes/kuavo-rl-opensource/
│       └── kuavo-robot-train/
│               ├── logs   ──链──→ _training_logs/.../logs    (8G+)
│               └── videos ──链──→ _training_logs/.../videos  (1G+)
├── leju_robot_rl/logs ──链──→ _training_logs/leju_robot_rl/logs
├── leju_robot_wm/logs ──链──→ _training_logs/leju_robot_wm/logs
│
├── kuavo-ros-opensource/                 ← 实体（仅魔改子目录链到 Notes）
├── kuavo_ros_application/                ← 实体（仅魔改子目录链到 Notes）
└── _training_logs/                       ← log/video 肉身，不进 GitHub
```

| 层级 | 存什么 | 典型路径 |
|------|--------|----------|
| **L1 入口** | 整仓软链接 | `~/kuavo_all/leju_robot_rl` |
| **L2 源码** | 本仓库根目录 | `~/Notes/kuavo-dev-notes/` |
| **L3 训练产物** | logs / videos / checkpoint | `~/kuavo_all/_training_logs/` |

魔改 ROS 包（下位机/上位机）是 **L2 影子目录 + L1 单文件/子目录反向链**，见脚本 `shadow` 子命令。

---

## 2. 脚本子命令速查

```bash
cd ~/Notes/kuavo-dev-notes

./automove_and_link.sh help              # 帮助
./automove_and_link.sh status            # 看链接目标与体积
./automove_and_link.sh verify            # 检查链接是否断裂

./automove_and_link.sh link-repos        # 整仓: kuavo_all → Notes
./automove_and_link.sh shadow            # 魔改影子: ROS 工作空间 → Notes
./automove_and_link.sh archive-logs      # logs 肉身 → _training_logs
./automove_and_link.sh archive-videos    # train/videos → _training_logs
./automove_and_link.sh cleanup-train     # 删 egg-info / __pycache__ / play.txt
./automove_and_link.sh git-untrack-logs  # rl/wm 从 Git 索引移除 logs

./automove_and_link.sh setup-all         # 新机器推荐：link + archive + verify
```

---

## 3. 我们做过什么（时间线备忘）

| 步骤 | 操作 | 命令 |
|------|------|------|
| 1 | 整仓提升到 Notes | 手动 `mv` + `ln -s`（或 `link-repos`） |
| 2 | 魔改包迁入本仓库 | `shadow` |
| 3 | 16G logs 挪出 | `archive-logs` |
| 4 | 1.3G videos 挪出 | `archive-videos` |
| 5 | deploy 编译 log | `sudo rm -rf .../kuavo-robot-deploy/logs` |
| 6 | 删 kuavo_ws.tar.gz | 204M 旧快照，Docker 不用它 |
| 7 | Git 不跟踪 logs | `git-untrack-logs` + 各子仓 commit |

---

## 4. 训练产物 archive 目录结构（内部路径勿改）

```text
~/kuavo_all/_training_logs/
├── leju_robot_rl/logs/rsl_rl/Kuavo/
│   ├── s42/flat/          ← RL 行走
│   ├── s42/rough/
│   └── s49/dance/         ← RL 舞蹈
├── leju_robot_wm/logs/
│   ├── tdmpc2/Kuavo/s42/tdmpc2_dance/     ← 世界模型·舞蹈
│   └── tdmpc2/Kuavo/s42/tdmpc2_velocity/  ← 世界模型·行走
└── kuavo-rl-opensource/kuavo-robot-train/
    ├── logs/Kuavo_s42_sk_ppo/   ← 旧 PPO 行走
    └── videos/Kuavo_s42_sk_ppo/ ← 录屏
```

Isaac Lab / RSL-RL 认 `./logs/rsl_rl/...` 路径，**归档时必须整棵 `mv`，不能扁平化重命名**。

---

## 5. 三条铁律（防止套娃乱）

1. **挪整个仓库目录** → 只动 `kuavo-dev-notes` 整包，再 `link-repos` 改入口链。  
2. **挪 log/video** → 只动 `_training_logs`，再 `archive-logs` / `archive-videos` 重挂链。  
3. **不要**只拖中间某一层（例如只挪 `leju_robot_rl` 不更新 `kuavo_all` 的链）。

挪之前先看链：

```bash
readlink ~/kuavo_all/leju_robot_rl
readlink ~/Notes/kuavo-dev-notes/leju_robot_rl/logs
```

---

## 6. 体积参考（不含 .git）

| 位置 | 约 |
|------|-----|
| `kuavo-dev-notes`（工作文件） | ~1.5G |
| 各子仓 `.git` 历史 | +~2G |
| `_training_logs` | ~17G |

GitHub push 不应包含 `_training_logs`；子仓 `.gitignore` 已含 `logs/`、`logs`。

---

## 7. 故障排查

| 现象 | 处理 |
|------|------|
| `mv logs 权限不够` (deploy) | `sudo rm -rf kuavo-robot-deploy/logs` |
| `git rm --cached logs` 无效 (rl) | logs 已是软链接时用 `git ls-files -z logs \| xargs -0 git rm --cached -f` |
| 训练找不到 checkpoint | `./automove_and_link.sh verify` |
| 文件管理器 2.8G vs du 4.9G | 属性不含隐藏 `.git` |

---

*最后更新：2026-06-28 · 与当前本机布局一致*
