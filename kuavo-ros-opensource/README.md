# kuavo-ros-opensource — 下位机魔改代码（影子目录）

本目录**不是**乐聚官方完整工作空间，仅归档 NUC（`~/kuavo-ros-opensource`）内**修改或自建**的包与文件。

官方完整仓库：[LejuRobotics/kuavo-ros-opensource](https://github.com/LejuRobotics/kuavo-ros-opensource)

---

## 整包

| 路径 | 文档 |
|------|------|
| `src/demo/vla_grasp/` | [22.1](../kuavo_notes/22.1VLA_grasping.md) · [22.2](../kuavo_notes/22.2.tree_VLA_grasp.md) · [28](../kuavo_notes/28.moveit_grasping.md) |
| `src/kuavo_arm_moveit_config/` | [28.moveit_grasping.md](../kuavo_notes/28.moveit_grasping.md) |
| `src/kuavo_arm_control/` | [18.teaching_gravity_compensation.md](../kuavo_notes/18.teaching_gravity_compensation.md) |
| `src/demo/teleop/` | [27.camera_mtion_capture.md](../kuavo_notes/27.camera_mtion_capture.md) |

## 单文件

| 路径 | 文档 |
|------|------|
| `src/kuavo_assets/config/kuavo_v49/kuavo.json` | [25.update.md](../kuavo_notes/25.update.md) |
| `src/kuavo_assets/models/biped_s49/urdf/biped_s49.urdf` | [28](../kuavo_notes/28.moveit_grasping.md) |
| `src/humanoid-control/h12pro_controller_node/config/*.json` | [17.h12_remote_control.md](../kuavo_notes/17.h12_remote_control.md) |

## vla_grasp 要点

- `vla_auto_grasp_daemon.py` / `vla_bt_daemon.py` — VLA 入口（二选一）
- `moveit_auto_grasp.py` / `moveit_octomap_grasp.py` — MoveIt 抓取
- `nuc_speaker_service.py` — 下位机 UDP 音频
- [`bt/README.md`](./src/demo/vla_grasp/bt/README.md) — 行为树

## 未收录（见文档）

FAST_LIO 补丁、`april_tag_recognition.py`、`load_kuavo_real.launch` 魔改版。
