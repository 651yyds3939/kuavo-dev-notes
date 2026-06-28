# kuavo_ros_application — 上位机魔改代码（影子目录）

本目录**不是**官方完整工作空间，仅归档 Orin NX（`~/kuavo_ros_application`）内**修改或自建**的内容。

---

## 整包

| 路径 | 文档 |
|------|------|
| `src/kuavo_local_face_interaction/` | [32.1](../kuavo_notes/32.1.face_recognition.md) · [32.2](../kuavo_notes/32.2.face_recognition_traking.md) |
| `src/kuavo_visual_follow/` | [24.1.visual_tracking.md](../kuavo_notes/24.1.visual_tracking.md) |

## 大模型 `src/kuavo_large_model/`

| 文件 | 文档 |
|------|------|
| `rtasr_python3_demo.py` | [21.2](../kuavo_notes/21.2.local_AI_large_model.md) |
| `rtasr_python3_demo_grasping.py` | [22.1](../kuavo_notes/22.1VLA_grasping.md) |
| `orin_gemini_gateway.py` | [21.3](../kuavo_notes/21.3.gemini_model.md) |
| `vlm_person_observer.py` | [30.AI_image_identification.md](../kuavo_notes/30.AI_image_identification.md) |
| `nuc_speaker_service.py` | [21.3](../kuavo_notes/21.3.gemini_model.md) |

## YOLO `src/ros_vision/.../scripts/`

| 文件 | 文档 |
|------|------|
| `yolo_box_segment_ros.py` | [4.3](../kuavo_notes/4.3.real_robot_yolo_environment.md) · [4.4](../kuavo_notes/4.4real_visual_grasp.md) |
| `yolo_box_segment_ros_TF2.py` | [4.4](../kuavo_notes/4.4real_visual_grasp.md) · [22.1](../kuavo_notes/22.1VLA_grasping.md) |
| `yolo_vision_with_observer_trigger.py` | [30.AI_image_identification.md](../kuavo_notes/30.AI_image_identification.md) |

## Launch `src/dynamic_biped/launch/`

`orbbec_sensor_robot_enable.launch` · `orbbec_sensor_only_enable.launch` — 见 [28.moveit_grasping.md](../kuavo_notes/28.moveit_grasping.md)

## 未收录

`yolov8n-seg.pt`、`mini_omni_inference.py`、`yolo_person_tracker.py`（文档仅片段）。
