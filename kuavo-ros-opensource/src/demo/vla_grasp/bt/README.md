# VLA 案例 22 — Python 行为树（py_trees）

位置：`kuavo-ros-opensource/src/demo/vla_grasp/bt/`（本仓库内路径；真机 NUC 同相对路径）  
入口：`../vla_bt_daemon.py`  
文档：[`kuavo_notes/22.2.tree_VLA_grasp.md`](../../../../../../kuavo_notes/22.2.tree_VLA_grasp.md)

---

## 文件

| 文件 | 作用 |
|------|------|
| `blackboard.py` | 黑板键与读写 |
| `config.py` | ROS 参数（话题、超时、tick 频率） |
| `grasp_skills.py` | IK 规划 + 分步动作 |
| `nodes.py` | BT 叶子/装饰器节点 |
| `trees.py` | 拼树 |
| `../vla_bt_daemon.py` | 进程入口 |

## 树结构（摘要）

```
Selector [VLA_ROOT]
├── Sequence [HandleGrab]  → ParseTarget → CollectTF2 → PlanIK → RunMotions
├── Sequence [HandleChat]  → ParseChat → SpeakTTS
└── Idle
```

常开进程（WBC / IK / YOLO / ASR / TTS）不进树，只通过话题 / HTTP 交互。

---

## 依赖

```bash
pip install py_trees
source ~/kuavo-ros-opensource/devel/setup.bash   # 或 ~/kuavo_all/kuavo-ros-opensource
```

## 启动

```bash
cd ~/kuavo-ros-opensource/src/demo/vla_grasp
python3 vla_bt_daemon.py
```

**不要**与 `vla_auto_grasp_daemon.py` 同时运行。

## 手动测试

```bash
rostopic pub /vla/master_command std_msgs/String \
  '{"data": "{\"action\": \"grab\", \"target\": \"可乐\"}"}' -1
```

## ROS 参数（私有 `~`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `~tick_hz` | 10 | BT tick 频率 |
| `~yolo_topic` | `/vla/yolo_target` | YOLO 目标 |
| `~yolo_sample_count` | 10 | 采集中值帧数 |
| `~tts_url` | `http://127.0.0.1:5000/tts` | TTS HTTP |

详见 22.2 文档与 `config.py`。
