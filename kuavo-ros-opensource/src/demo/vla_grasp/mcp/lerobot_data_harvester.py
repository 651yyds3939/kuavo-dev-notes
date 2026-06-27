#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot v3.0 数据收割机 — 以 moveit_auto_grasp.py 全闭环管线为专家工人。

在桌面有效空域内循环执行抓取 Episode，并以 50Hz 录制：
  - observation.images.head  (头部 RGB)
  - observation.state        (14 轴反馈，弧度)
  - action                   (14 轴目标下发，弧度)

依赖：
  pip install lerobot opencv-python-headless

启动（NUC root + WBC + IK + move_group + YOLO 已就绪）：
  source devel/setup.bash
  python3 src/demo/vla_grasp/mcp/lerobot_data_harvester.py \\
      --episodes 100 --output-dir ~/lerobot_kuavo_grasp

⚠️ 不修改 vla_bt_daemon.py / bt/ — 独立 LeRobot 升级通道。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLA_DIR = os.path.dirname(_SCRIPT_DIR)
if _VLA_DIR not in sys.path:
    sys.path.insert(0, _VLA_DIR)

import moveit_auto_grasp as mag
from claw_safe import build_open_cmd

import moveit_commander
import rospy
from cv_bridge import CvBridge
from kuavo_msgs.msg import armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from sensor_msgs.msg import Image, JointState

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None  # type: ignore

# ---------------------------------------------------------------------------
# 50Hz 高频采样器
# ---------------------------------------------------------------------------

class HighFreqRecorder:
    """Episode 生命周期内以固定频率采样图像 + 14 轴 state/action。"""

    def __init__(
        self,
        fps: float = 50.0,
        image_topic: str = "/camera/color/image_raw",
    ):
        self.fps = fps
        self.period = 1.0 / fps
        self.image_topic = image_topic
        self._bridge = CvBridge()
        self._latest_image: Optional[np.ndarray] = None
        self._image_lock = threading.Lock()
        self._frames: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        rospy.Subscriber(image_topic, Image, self._image_cb, queue_size=1)

    def _image_cb(self, msg: Image):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv_img[:, :, ::-1].copy()
            with self._image_lock:
                self._latest_image = rgb
        except Exception:
            pass

    def start(self):
        self._frames = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[Dict[str, Any]]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return list(self._frames)

    def _loop(self):
        while self._running and not rospy.is_shutdown():
            t0 = time.time()
            with self._image_lock:
                img = None if self._latest_image is None else self._latest_image.copy()
            state = np.copy(mag.current_joints_rad)
            action = np.copy(mag.last_commanded_joints_rad)
            if img is not None:
                self._frames.append(
                    {
                        "observation.images.head": img,
                        "observation.state": state.astype(np.float32),
                        "action": action.astype(np.float32),
                    }
                )
            elapsed = time.time() - t0
            sleep_t = self.period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


# ---------------------------------------------------------------------------
# 单 Episode：moveit_auto_grasp 管线（像素级复用）
# ---------------------------------------------------------------------------

def _init_ros_for_harvest():
    if not rospy.core.is_initialized():
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("lerobot_data_harvester", anonymous=True)
    rospy.Subscriber("/joint_states", JointState, mag.joint_states_callback, queue_size=1)
    while not mag.has_joint_states and not rospy.is_shutdown():
        rospy.sleep(0.05)
    mag.last_commanded_joints_rad = np.copy(mag.current_joints_rad)
    try:
        rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
            changeArmCtrlModeRequest(control_mode=2)
        )
    except Exception as exc:
        rospy.logwarn("arm_traj_change_mode(2): %s", exc)


def _collect_or_random_targets(
    use_vision: bool,
    rng: random.Random,
    x_range: tuple,
    y_left_range: tuple,
    y_right_range: tuple,
) -> tuple:
    if use_vision:
        x_hist, y_hist = mag._collect_vision_targets_tf2_style()
        if len(x_hist) >= 10:
            return x_hist, y_hist
        rospy.logwarn("视觉采点失败，本 Episode 改用随机桌面坐标")
    # 随机桌面空域（base_link 米）
    raw_x = rng.uniform(x_range[0], x_range[1])
    use_left = rng.random() > 0.5
    if use_left:
        raw_y = rng.uniform(y_left_range[0], y_left_range[1])
    else:
        raw_y = rng.uniform(y_right_range[0], y_right_range[1])
    # 伪造 10 帧中值输入（单点重复，与 mag 中值等价）
    return [raw_x] * 10, [raw_y] * 10


def run_grasp_episode(
    arm_pub: rospy.Publisher,
    left_arm,
    right_arm,
    x_hist: List[float],
    y_hist: List[float],
    recorder: HighFreqRecorder,
    min_frames: int = 50,
) -> tuple:
    """
    直接调用 moveit_auto_grasp._run_grasp_sequence 全闭环管线，并包裹 50Hz 录制。
    返回 (success, frames)。
    """
    recorder.start()
    try:
        mag._run_grasp_sequence(left_arm, right_arm, arm_pub, x_hist, y_hist)
    except Exception as exc:
        rospy.logerr("Episode 异常: %s", exc)
        frames = recorder.stop()
        return False, frames
    frames = recorder.stop()
    ok = len(frames) >= min_frames
    if not ok:
        rospy.logwarn("Episode 帧数不足 (%d < %d)，视为失败", len(frames), min_frames)
    return ok, frames


# ---------------------------------------------------------------------------
# LeRobot v3.0 数据集写入
# ---------------------------------------------------------------------------

def create_lerobot_dataset(output_dir: str, fps: int):
    if LeRobotDataset is None:
        raise RuntimeError("lerobot 未安装。请执行: pip install lerobot")

    joint_names = list(mag.joint_names_14)
    features = {
        "observation.images.head": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": joint_names,
        },
    }
    repo_id = "local/kuavo_vla_grasp"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type="kuavo",
        root=output_dir,
        use_videos=True,
    )
    return dataset


def append_episode_to_dataset(dataset, frames: List[Dict[str, Any]], task: str):
    for frame in frames:
        dataset.add_frame(
            {
                "observation.images.head": frame["observation.images.head"],
                "observation.state": frame["observation.state"],
                "action": frame["action"],
                "task": task,
            }
        )
    dataset.save_episode()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kuavo LeRobot v3.0 data harvester")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=os.path.expanduser("~/lerobot_kuavo_grasp"))
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--image-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--use-vision", action="store_true", default=True,
                        help="每 Episode 用 YOLO 10 帧采点（默认开）")
    parser.add_argument("--no-vision", action="store_false", dest="use_vision")
    parser.add_argument("--reposition-pause", type=float, default=5.0,
                        help="Episode 间隔秒数（便于人工挪动水瓶）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    _init_ros_for_harvest()

    arm_pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    rospy.sleep(0.3)

    pos, vel, effort = build_open_cmd()
    mag.call_leju_claw(pos, vel, effort, tag="harvest-init-open")
    mag.execute_dual_arm_init_home(arm_pub)

    rospy.loginfo("加载 MoveIt 规划器...")
    left_arm = moveit_commander.MoveGroupCommander("left_arm")
    right_arm = moveit_commander.MoveGroupCommander("right_arm")
    for grp in (left_arm, right_arm):
        grp.set_pose_reference_frame("base_link")

    dataset = create_lerobot_dataset(args.output_dir, args.fps)
    recorder = HighFreqRecorder(fps=args.fps, image_topic=args.image_topic)

    x_range = (0.38, 0.52)
    y_left_range = (0.02, 0.12)
    y_right_range = (-0.12, -0.02)

    success_count = 0
    for ep in range(args.episodes):
        rospy.loginfo("========== Episode %d / %d ==========", ep + 1, args.episodes)
        x_hist, y_hist = _collect_or_random_targets(
            args.use_vision, rng, x_range, y_left_range, y_right_range,
        )
        ok, frames = run_grasp_episode(
            arm_pub, left_arm, right_arm, x_hist, y_hist, recorder,
        )

        if ok and frames:
            task = "grasp bottle ep%d" % ep
            append_episode_to_dataset(dataset, frames, task)
            success_count += 1
            rospy.loginfo("Episode %d 成功，录制 %d 帧 @ %dHz", ep, len(frames), args.fps)
        else:
            rospy.logwarn("Episode %d 失败或无可写帧", ep)

        if ep + 1 < args.episodes:
            rospy.loginfo("等待 %.1fs — 请将水瓶移动到新位置...", args.reposition_pause)
            time.sleep(args.reposition_pause)

    dataset.finalize()
    rospy.loginfo(
        "收割完成: %d / %d episodes 成功，数据目录: %s",
        success_count, args.episodes, args.output_dir,
    )


if __name__ == "__main__":
    main()
