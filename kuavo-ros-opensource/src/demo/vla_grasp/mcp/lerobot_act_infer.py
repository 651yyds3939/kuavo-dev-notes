#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot ACT 真机推理节点 — 默认开环（只推理、不执行）。

订阅:
  - /camera/color/image_raw  → observation.images.head (resize 640×400 RGB)
  - /joint_states            → observation.state (14 轴双臂, rad)

默认只打印 / 写 CSV，不向 /kuavo_arm_target_poses 发指令。
加 --execute 才下发关节（闭环前请先通过开环验证）。

启动前（NUC）:
  终端 1  WBC  load_kuavo_real.launch
  终端 2  look_down.py
  终端 3  kuavo_state_publisher.py
  终端 4 (Orin) load_robot_head.launch

  conda activate lerobot
  source devel/setup.bash
  # 推理节点需要 ROS PYTHONPATH，不要 unset PYTHONPATH
  export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
  python src/demo/vla_grasp/mcp/lerobot_act_infer.py \\
      --checkpoint src/demo/vla_grasp/pretrained_model \\
      --fps 10 --device cpu

⚠️ 勿与 lerobot_data_harvester / moveit_auto_grasp / vla_bt_daemon 同时运行。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLA_DIR = os.path.dirname(_SCRIPT_DIR)
if _VLA_DIR not in sys.path:
    sys.path.insert(0, _VLA_DIR)

_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
_DEVEL_PYTHON = os.path.join(_WORKSPACE_ROOT, "devel", "lib", "python3", "dist-packages")
_ROS_PYTHON = "/opt/ros/noetic/lib/python3/dist-packages"


def _ensure_ros_workspace_pythonpath() -> None:
    """conda Python 默认找不到 catkin 编译的 kuavo_msgs，补工作空间 devel 路径。"""
    for path in (_DEVEL_PYTHON, _ROS_PYTHON):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _import_kuavo_msgs():
    _ensure_ros_workspace_pythonpath()
    try:
        from kuavo_msgs.msg import armTargetPoses  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    raise ModuleNotFoundError(
        "No module named 'kuavo_msgs' — 闭环推理需要 ROS 工作空间消息包。\n"
        "请先执行:\n"
        "  cd ~/kuavo-ros-opensource && source devel/setup.bash\n"
        "  export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH\n"
        "再运行本脚本（conda activate lerobot 之后）。"
    )


import rospy
from sensor_msgs.msg import Image, JointState

IMAGE_KEY = "observation.images.head"
STATE_KEY = "observation.state"

JOINT_NAMES_14 = [
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
]

# 与 pack --resize 400 640 一致 (H=400, W=640)
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640

# 与 moveit_auto_grasp / auto_grasp_TF2 一致
DUAL_ARM_INIT_DEG = [
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
]
SHOULDER_SWING_AVOID_DEG = 75.0


def _image_msg_to_rgb(msg: Image) -> np.ndarray:
    """从 sensor_msgs/Image 解 RGB，不依赖 cv_bridge（避免 conda libffi 冲突）。"""
    enc = (msg.encoding or "").lower()
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        raise ValueError("invalid image size %dx%d" % (w, h))

    if enc in ("bgr8", "rgb8"):
        channels = 3
        row_bytes = w * channels
        step = int(msg.step) if msg.step else row_bytes
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if step == row_bytes:
            img = buf.reshape(h, w, channels)
        else:
            img = buf.reshape(h, step)[:, :row_bytes].reshape(h, w, channels)
        if enc == "bgr8":
            return np.ascontiguousarray(img[:, :, ::-1])
        return np.ascontiguousarray(img)

    if enc in ("mono8", "8uc1"):
        step = int(msg.step) if msg.step else w
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if step == w:
            gray = buf.reshape(h, w)
        else:
            gray = buf.reshape(h, step)[:, :w]
        return np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))

    # 少数相机走 jpeg/mjpg，再尝试 cv2（仍可能受 libffi 影响）
    if enc in ("jpeg", "jpg", "mjpeg", "mjpg"):
        import cv2
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed for encoding=%s" % enc)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ValueError("unsupported image encoding: %s" % msg.encoding)


class RosObservationSource:
    """线程安全缓存最新相机帧与 14 轴关节反馈。"""

    def __init__(self, image_topic: str):
        self._png_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._image_stamp: float = 0.0
        self._image_count = 0

        self.current_joints_rad = np.zeros(14, dtype=np.float64)
        self.has_joint_states = False
        self._joint_stamp: float = 0.0

        rospy.Subscriber(image_topic, Image, self._image_cb, queue_size=1)
        rospy.Subscriber("/joint_states", JointState, self._joint_cb, queue_size=1)

    def _image_cb(self, msg: Image) -> None:
        try:
            rgb = _image_msg_to_rgb(msg)
            with self._png_lock:
                self._latest_rgb = rgb
                self._image_stamp = time.time()
                self._image_count += 1
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "图像解码失败: %s (encoding=%s)", exc, msg.encoding)

    def _joint_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in JOINT_NAMES_14:
                idx = JOINT_NAMES_14.index(name)
                self.current_joints_rad[idx] = msg.position[i]
        self.has_joint_states = True
        self._joint_stamp = time.time()

    def wait_ready(self, timeout_sec: float = 30.0) -> bool:
        deadline = time.time() + timeout_sec
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.has_joint_states and self._image_count > 0:
                return True
            rate.sleep()
        return False

    def snapshot(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        with self._png_lock:
            rgb = None if self._latest_rgb is None else self._latest_rgb.copy()
        state = self.current_joints_rad.astype(np.float32).copy()
        return rgb, state

    @property
    def image_age_sec(self) -> float:
        if self._image_count == 0:
            return float("inf")
        return time.time() - self._image_stamp

    @property
    def joint_age_sec(self) -> float:
        if not self.has_joint_states:
            return float("inf")
        return time.time() - self._joint_stamp


def _parse_args() -> argparse.Namespace:
    default_ckpt = os.path.join(_VLA_DIR, "pretrained_model")
    p = argparse.ArgumentParser(description="LeRobot ACT 真机推理（默认开环）")
    p.add_argument("--checkpoint", default=default_ckpt, help="pretrained_model 目录")
    p.add_argument("--image-topic", default="/camera/color/image_raw")
    p.add_argument("--fps", type=float, default=25.0, help="控制频率，与 pack stride 后有效 fps 一致")
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="闭环：将 action 发 /kuavo_arm_target_poses（默认关）",
    )
    p.add_argument(
        "--max-delta-deg",
        type=float,
        default=5.0,
        help="--execute 时每步单关节最大变化 (deg)，安全限幅",
    )
    p.add_argument(
        "--segment-dt",
        type=float,
        default=0.08,
        help="--execute 时下发航点时长 (s)；建议 ≥ 1/fps，或用 --sync-segment-dt",
    )
    p.add_argument(
        "--sync-segment-dt",
        action="store_true",
        help="自动设 segment_dt = 1.2/fps，减少指令重叠导致的一卡一卡",
    )
    p.add_argument(
        "--wait-segment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="闭环每步下发后等待 segment_dt 再发下一步（默认开）",
    )
    p.add_argument(
        "--warmup-inference",
        type=int,
        default=2,
        help="正式循环前先跑 N 步推理预热 torch（不执行、不写 CSV）",
    )
    p.add_argument(
        "--home-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="闭环结束/中断后自动收手归位（默认开）",
    )
    p.add_argument(
        "--home-mode",
        choices=("init", "vla"),
        default="vla",
        help="init=仅 DUAL_ARM_INIT 下垂；vla=大鹏展翅→护胸→下垂",
    )
    p.add_argument(
        "--active-arm",
        choices=("left", "right"),
        default="right",
        help="vla 收手时的活动臂（与本次主要运动的臂一致）",
    )
    p.add_argument(
        "--open-claw-at-start",
        action="store_true",
        help="闭环开始前张开双手夹爪",
    )
    p.add_argument(
        "--close-claw-at-end",
        action="store_true",
        help="步数跑满后闭合活动臂夹爪（需配合 --active-arm）",
    )
    p.add_argument(
        "--open-claw-on-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="收手归位后张开夹爪（默认开）",
    )
    p.add_argument(
        "--log-csv",
        default="",
        help="可选 CSV 日志路径，如 ~/lerobot_openloop_log.csv",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="跑 N 步后自动退出，0=不限",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="每 N 步打印一行摘要",
    )
    p.add_argument(
        "--reset-every",
        type=int,
        default=0,
        help="每 N 步 policy.reset()；0=仅启动时 reset 一次",
    )
    p.add_argument(
        "--stale-timeout",
        type=float,
        default=2.0,
        help="相机或 joint_states 超过此秒数未更新则告警",
    )
    return p.parse_args()


def _resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_policy(checkpoint: str, device):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def _resize_rgb(rgb: np.ndarray) -> np.ndarray:
    import cv2

    if rgb.shape[0] == IMAGE_HEIGHT and rgb.shape[1] == IMAGE_WIDTH:
        return rgb
    return cv2.resize(
        rgb,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def _build_obs_tensors(rgb_hwc: np.ndarray, state: np.ndarray, device):
    import torch

    obs_np = {
        IMAGE_KEY: np.ascontiguousarray(rgb_hwc, dtype=np.uint8),
        STATE_KEY: np.ascontiguousarray(state, dtype=np.float32),
    }
    try:
        from lerobot.policies.utils import prepare_observation_for_inference

        return prepare_observation_for_inference(obs_np, device)
    except ImportError:
        return {
            IMAGE_KEY: torch.from_numpy(obs_np[IMAGE_KEY])
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            / 255.0,
            STATE_KEY: torch.from_numpy(obs_np[STATE_KEY])
            .float()
            .unsqueeze(0)
            .to(device),
        }


def _clip_action_to_state(
    action: np.ndarray,
    state: np.ndarray,
    max_delta_rad: float,
) -> np.ndarray:
    delta = np.clip(action - state, -max_delta_rad, max_delta_rad)
    return state + delta


def _clamp_elbow_deg(target_deg: List[float]) -> List[float]:
    """肘关节死锁防护（与 moveit_auto_grasp 一致，左右臂第 4 轴）。"""
    out = list(target_deg)
    if out[3] > 0.0:
        out[3] = 0.0
    if out[10] > 0.0:
        out[10] = 0.0
    return out


def _ensure_arm_trajectory_mode() -> None:
    """闭环下发前切外部轨迹模式（control_mode=2）。"""
    _import_kuavo_msgs()
    try:
        from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

        rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
            changeArmCtrlModeRequest(control_mode=2)
        )
        rospy.loginfo("arm_traj_change_mode(2) OK")
    except Exception as exc:
        rospy.logwarn("arm_traj_change_mode(2) 失败: %s", exc)


def _make_arm_publisher():
    _import_kuavo_msgs()
    from kuavo_msgs.msg import armTargetPoses

    return rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)


def _publish_arm_target(
    pub,
    joint_rad: np.ndarray,
    dt: float,
) -> None:
    from kuavo_msgs.msg import armTargetPoses

    target_deg = _clamp_elbow_deg([math.degrees(float(v)) for v in joint_rad.tolist()])
    msg = armTargetPoses()
    msg.times = [float(dt)]
    msg.values = target_deg
    pub.publish(msg)


def _publish_arm_target_and_wait(pub, joint_rad: np.ndarray, dt: float) -> None:
    _publish_arm_target(pub, joint_rad, dt)
    rospy.sleep(dt + 0.15)


def _init_joints_rad() -> np.ndarray:
    return np.radians(DUAL_ARM_INIT_DEG)


def _auto_grasp_ready_deg(is_left_arm: bool) -> List[float]:
    if is_left_arm:
        return [40, 20, 0, -120, 0, 0, -20, 20, 0, 0, -30, 0, 0, 0]
    return [20, 0, 0, -30, 0, 0, 0, 40, -20, 0, -120, 0, 0, -20]


def _freeze_inactive_arm(joints_14_rad: np.ndarray, is_left_arm: bool) -> np.ndarray:
    frozen = np.copy(joints_14_rad)
    init = _init_joints_rad()
    inactive = slice(7, 14) if is_left_arm else slice(0, 7)
    frozen[inactive] = init[inactive]
    return frozen


def _build_high_safe_joints(q_lift: np.ndarray, is_left_arm: bool) -> np.ndarray:
    q = np.copy(q_lift)
    if is_left_arm:
        q[1] += math.radians(SHOULDER_SWING_AVOID_DEG)
    else:
        q[8] -= math.radians(SHOULDER_SWING_AVOID_DEG)
    return _freeze_inactive_arm(q, is_left_arm)


def _home_init(pub) -> None:
    rospy.loginfo("收手: DUAL_ARM_INIT 双臂下垂...")
    _publish_arm_target_and_wait(pub, _init_joints_rad(), 2.5)


def _home_vla(pub, last_joints_rad: np.ndarray, is_left_arm: bool) -> None:
    side = "左" if is_left_arm else "右"
    rospy.loginfo("收手: %s手 vla 宏（肩膀外摆 %d° → 护胸 → 下垂）...", side, int(SHOULDER_SWING_AVOID_DEG))
    q_high = _build_high_safe_joints(last_joints_rad, is_left_arm)
    ready_rad = np.radians(_auto_grasp_ready_deg(is_left_arm))
    _publish_arm_target_and_wait(pub, q_high, 2.0)
    _publish_arm_target_and_wait(pub, ready_rad, 3.0)
    _publish_arm_target_and_wait(pub, _init_joints_rad(), 2.5)


def _run_home_sequence(pub, mode: str, active_arm: str, last_joints_rad: np.ndarray) -> None:
    if pub is None:
        return
    try:
        is_left = active_arm == "left"
        if mode == "init":
            _home_init(pub)
        else:
            _home_vla(pub, last_joints_rad, is_left)
    except Exception as exc:
        rospy.logwarn("收手失败: %s — 可手动运行 DUAL_ARM_INIT 脚本", exc)


def _claw_open() -> None:
    from claw_safe import build_open_cmd, get_controller

    pos, vel, effort = build_open_cmd()
    get_controller().call(pos, vel, effort, tag="lerobot-open")
    rospy.loginfo("夹爪: 张开")


def _claw_close(active_arm: str) -> None:
    from claw_safe import build_close_cmd, get_controller

    is_left = active_arm == "left"
    pos, vel, effort = build_close_cmd(is_left_arm=is_left)
    ok = get_controller().call(pos, vel, effort, tag="lerobot-close")
    side = "左" if is_left else "右"
    rospy.loginfo("夹爪: %s手闭合 %s", side, "OK" if ok else "中止(堵转?)")


def _warmup_policy(policy, preprocessor, postprocessor, source, device, steps: int) -> None:
    import torch

    if steps <= 0:
        return
    rospy.loginfo("预热推理 %d 步（加载 torch 缓存，不执行）...", steps)
    for _ in range(steps):
        rgb, state = source.snapshot()
        if rgb is None:
            rospy.sleep(0.1)
            continue
        obs_t = _build_obs_tensors(_resize_rgb(rgb), state, device)
        with torch.inference_mode():
            proc = preprocessor(obs_t)
            postprocessor(policy.select_action(proc))
        rospy.sleep(0.05)
    policy.reset()
    rospy.loginfo("预热完成，policy.reset()")


def _open_csv_writer(path: str, joint_names: List[str]):
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="", encoding="utf-8")
    header = (
        ["step", "wall_time", "infer_ms", "delta_mae_rad", "mode"]
        + ["state_%s" % n for n in joint_names]
        + ["action_%s" % n for n in joint_names]
    )
    writer = csv.writer(f)
    writer.writerow(header)
    return f, writer, path


def _wait_for_ros_topics(image_topic: str, timeout_sec: float = 20.0) -> None:
    topics = [image_topic, "/joint_states"]
    deadline = time.time() + timeout_sec
    for topic in topics:
        while not rospy.is_shutdown() and time.time() < deadline:
            found = dict(rospy.get_published_topics())
            if topic in found:
                rospy.loginfo("话题就绪: %s (%s)", topic, found[topic])
                break
            rospy.logwarn_throttle(3.0, "等待话题: %s ...", topic)
            rospy.sleep(0.5)
        else:
            raise RuntimeError("超时未等到话题: %s" % topic)


def main() -> int:
    args = _parse_args()

    try:
        import torch
    except ImportError:
        print(
            "错误: 需要 torch + lerobot。\n"
            "  conda activate lerobot && pip install lerobot torch opencv-python-headless",
            file=sys.stderr,
        )
        return 1

    ckpt = os.path.expanduser(args.checkpoint)
    if not os.path.isdir(ckpt):
        print("错误: checkpoint 目录不存在: %s" % ckpt, file=sys.stderr)
        return 1
    if not os.path.isfile(os.path.join(ckpt, "model.safetensors")):
        print("错误: 缺少 model.safetensors: %s" % ckpt, file=sys.stderr)
        return 1

    rospy.init_node("lerobot_act_infer", anonymous=False)
    mode = "EXECUTE" if args.execute else "OPEN_LOOP"
    rospy.loginfo("LeRobot ACT 推理 | mode=%s | ckpt=%s", mode, ckpt)

    _wait_for_ros_topics(args.image_topic)
    source = RosObservationSource(args.image_topic)

    if not source.wait_ready(timeout_sec=30.0):
        rospy.logerr(
            "观测未就绪: image_count=%d has_joint_states=%s\n"
            "请确认 Orin 相机 + kuavo_state_publisher 已启动。",
            source._image_count,
            source.has_joint_states,
        )
        return 1

    device = _resolve_device(args.device)
    rospy.loginfo("加载策略 device=%s ...", device)
    policy, preprocessor, postprocessor = _load_policy(ckpt, device)
    policy.reset()
    rospy.loginfo(
        "策略就绪 chunk=%s n_action_steps=%s image=%dx%d",
        getattr(policy.config, "chunk_size", "?"),
        getattr(policy.config, "n_action_steps", "?"),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    arm_pub = _make_arm_publisher() if args.execute else None
    if args.execute:
        if args.sync_segment_dt:
            args.segment_dt = max(0.08, 1.2 / max(args.fps, 1.0))
            rospy.loginfo(
                "sync-segment-dt: segment_dt=%.3fs (fps=%.1f)",
                args.segment_dt,
                args.fps,
            )
        period = 1.0 / max(args.fps, 1.0)
        if args.segment_dt < period * 0.95 and not args.wait_segment:
            rospy.logwarn(
                "segment_dt(%.2fs) < 控制周期(%.2fs) 且未 wait-segment，易卡顿；"
                "建议 --sync-segment-dt 或 --wait-segment",
                args.segment_dt,
                period,
            )
        max_delta_rad = np.deg2rad(args.max_delta_deg)
        _ensure_arm_trajectory_mode()
        if args.open_claw_at_start:
            _claw_open()
        rospy.logwarn(
            "⚠️ EXECUTE 模式：将向 /kuavo_arm_target_poses 下发 "
            "(max_delta=%.1f°/step, dt=%.2fs, wait_segment=%s)",
            args.max_delta_deg,
            args.segment_dt,
            args.wait_segment,
        )
        rospy.sleep(0.5)

    _warmup_policy(policy, preprocessor, postprocessor, source, device, args.warmup_inference)

    csv_file = None
    csv_writer = None
    csv_path = ""
    if args.log_csv:
        csv_file, csv_writer, csv_path = _open_csv_writer(args.log_csv, JOINT_NAMES_14)
        rospy.loginfo("CSV 日志: %s", csv_path)

    period = 1.0 / max(args.fps, 1.0)
    step = 0
    infer_ms_ema = 0.0
    last_sent = source.current_joints_rad.astype(np.float64).copy()
    finished_normally = False

    try:
        while not rospy.is_shutdown():
            if args.max_steps > 0 and step >= args.max_steps:
                rospy.loginfo("已达 max_steps=%d，退出。", args.max_steps)
                finished_normally = True
                break

            if args.reset_every > 0 and step > 0 and step % args.reset_every == 0:
                policy.reset()
                rospy.loginfo("step=%d → policy.reset()", step)

            t_loop = time.time()
            rgb, state = source.snapshot()
            if rgb is None:
                rospy.logwarn_throttle(2.0, "尚无相机帧，跳过")
                rospy.sleep(period)
                continue

            if source.image_age_sec > args.stale_timeout:
                rospy.logwarn_throttle(
                    3.0,
                    "相机数据 stale %.1fs",
                    source.image_age_sec,
                )
            if source.joint_age_sec > args.stale_timeout:
                rospy.logwarn_throttle(
                    3.0,
                    "joint_states stale %.1fs",
                    source.joint_age_sec,
                )

            rgb_small = _resize_rgb(rgb)
            obs_t = _build_obs_tensors(rgb_small, state, device)

            t_infer = time.time()
            with torch.inference_mode():
                proc = preprocessor(obs_t)
                action_t = postprocessor(policy.select_action(proc))
                action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
            infer_ms = (time.time() - t_infer) * 1000.0
            infer_ms_ema = infer_ms if step == 0 else 0.9 * infer_ms_ema + 0.1 * infer_ms

            delta = action - state
            delta_mae = float(np.mean(np.abs(delta)))

            if args.execute and arm_pub is not None:
                safe_action = _clip_action_to_state(
                    action, state, max_delta_rad=np.deg2rad(args.max_delta_deg)
                )
                _publish_arm_target(arm_pub, safe_action, args.segment_dt)
                sent = safe_action
                last_sent = safe_action.copy()
                if args.wait_segment:
                    rospy.sleep(args.segment_dt)
            else:
                sent = action

            if step % max(1, args.print_every) == 0:
                rospy.loginfo(
                    "step=%4d infer=%5.1fms(ema=%.1f) delta_mae=%.4frad(%.2f°) "
                    "L1=%.3f R1=%.3f | mode=%s",
                    step,
                    infer_ms,
                    infer_ms_ema,
                    delta_mae,
                    np.degrees(delta_mae),
                    sent[0],
                    sent[7],
                    mode,
                )

            if csv_writer is not None:
                csv_writer.writerow(
                    [step, time.time(), infer_ms, delta_mae, mode]
                    + state.tolist()
                    + sent.tolist()
                )

            step += 1
            elapsed = time.time() - t_loop
            sleep_t = period - elapsed
            if sleep_t > 0:
                rospy.sleep(sleep_t)
            elif step % max(1, args.print_every) == 0:
                rospy.logwarn(
                    "推理慢于目标 fps=%.1f (loop=%.0fms > period=%.0fms)",
                    args.fps,
                    elapsed * 1000.0,
                    period * 1000.0,
                )

    except KeyboardInterrupt:
        rospy.loginfo("用户中断")
    finally:
        if csv_file is not None:
            csv_file.close()
            rospy.loginfo("CSV 已保存: %s (%d 行)", csv_path, step)

        if args.execute and arm_pub is not None:
            if args.close_claw_at_end and finished_normally:
                try:
                    _claw_close(args.active_arm)
                    rospy.sleep(1.0)
                except Exception as exc:
                    rospy.logwarn("夹爪闭合失败: %s", exc)
            if args.home_on_exit:
                _run_home_sequence(
                    arm_pub, args.home_mode, args.active_arm, last_sent
                )
                if args.open_claw_on_home:
                    try:
                        _claw_open()
                    except Exception as exc:
                        rospy.logwarn("收手后开爪失败: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
