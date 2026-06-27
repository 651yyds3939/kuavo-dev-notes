# VLA 案例 22 — Python 行为树（py_trees）

位置：`src/demo/vla_grasp/bt/`，与 `vla_auto_grasp_daemon.py` 同包。

## 文件

| 文件 | 作用 |
|------|------|
| `blackboard.py` | 黑板键与读写 |
| `config.py` | ROS 参数（话题、超时、tick 频率） |
| `grasp_skills.py` | IK 规划 + 分步动作 |
| `nodes.py` | BT 叶子/装饰器节点 |
| `trees.py` | 拼树 |
| `../vla_bt_daemon.py` | 入口 |

## 树结构

```
Selector [VLA_ROOT]
├── Sequence [HandleGrab]
│   ├── HasAction[grab]
│   ├── EnsureNotBusy
│   ├── MarkBusy
│   └── BusyScope [GrabBusyScope]     ← 失败/成功都自动解锁
│       └── Sequence [GrabWork]
│           ├── GraspPipeline
│           │   ├── ParseTarget
│           │   ├── CheckYoloAlive
│           │   ├── CollectTF2_median  ← 增量采点，RUNNING
│           │   ├── PlanIK_ABDE
│           │   └── RunMotions         ← 11 个分步动作节点
│           └── ClearCommand
├── Sequence [HandleChat]
│   ├── HasAction[chat]
│   ├── EnsureNotBusy
│   ├── MarkBusy
│   └── BusyScope [ChatBusyScope]
│       └── Sequence [ChatWork]
│           ├── ParseChatFields
│           ├── SpeakTTS               ← POST tts_server:5000
│           └── ClearCommand
└── Idle
```

常开进程（WBC/IK/YOLO/Whisper/TTS server）不进树，只通过话题/HTTP 交互。

## 依赖

```bash
pip install py_trees
source ~/kuavo_all/kuavo-ros-opensource/devel/setup.bash
```

## 启动

```bash
cd ~/kuavo_all/kuavo-ros-opensource/src/demo/vla_grasp
python3 vla_bt_daemon.py
```

**不要**与 `vla_auto_grasp_daemon.py` 同时运行。

### 可调 ROS 参数（`~` 私有参数）

| 参数 | 默认 | 说明 |
|------|------|------|
| `~tick_hz` | 10 | BT tick 频率 |
| `~yolo_topic` | `/vla/yolo_target` | YOLO 目标话题 |
| `~yolo_sample_count` | 10 | 采集中值帧数 |
| `~yolo_collect_timeout` | 8.0 | 采点总超时（秒） |
| `~tts_url` | `http://127.0.0.1:5000/tts` | TTS HTTP 地址 |
| `~log_status_change` | true | 根节点状态变化打日志 |

示例：

```bash
rosrun demo_pkg vla_bt_daemon.py _yolo_sample_count:=15 _tick_hz:=20
# 或直接 python3：
python3 vla_bt_daemon.py _yolo_collect_timeout:=12.0
```

## 手动测试

抓取：

```bash
rostopic pub /vla/master_command std_msgs/String \
  '{"data": "{\"action\": \"grab\", \"target\": \"可乐\"}"}' -1
```

对话（需先起 `tts_server.py`）：

```bash
rostopic pub /vla/master_command std_msgs/String \
  '{"data": "{\"action\": \"chat\", \"text\": \"好的，我来帮你拿\"}"}' -1
```

## 相对上一版的优化点

1. **动作拆成 11 步**：失败时日志能精确定位到哪一步（如 `CloseClaw`）。
2. **增量采点**：`CollectYoloTarget` 用 `RUNNING` 分 tick 采集，不阻塞整棵树一帧收完。
3. **BusyScope**：抓取/对话中途失败也会自动 `working=False`，不会死锁。
4. **chat 分支**：对接现有 `tts_server.py` HTTP 接口。
5. **CheckYoloAlive**：执行前确认 YOLO 话题可用。
6. **ROS 参数化**：话题、超时、tick 频率可在线调。

## 二次开发

- 在 `RunMotions` 某步外包 `py_trees.decorators.Retry` 做重试。
- 在 `HandleGrab` 前加 `look_down` 触发节点。
- 参考官方：`kuavo_humanoid_sdk/.../kuavo_strategy_pytree/`。
