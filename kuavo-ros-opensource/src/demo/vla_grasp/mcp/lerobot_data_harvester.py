#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot v3.0 数据收割机 — 以 moveit_auto_grasp.py 全闭环管线为专家工人。

在桌面有效空域内循环执行抓取 Episode，并以 50Hz 录制：
  - observation.images.head  (头部 RGB)
  - observation.state        (14 轴反馈，弧度)
  - action                   (14 轴目标下发，弧度)

【NUC Python 3.8 推荐】默认 --raw-only，只写 npz，不依赖 lerobot 库。
【Orin / PC Python 3.12+】采完 raw 后运行 lerobot_pack_dataset.py 打包；
  或在已装 lerobot 的机器上加 --lerobot 直接写 LeRobot v3 格式。

启动（NUC root + WBC + IK + move_group + YOLO 已就绪）：
  source devel/setup.bash
  python3 src/demo/vla_grasp/mcp/lerobot_data_harvester.py \\
      --episodes 10 --output-dir ~/lerobot_kuavo_grasp_raw

⚠️ 不修改 vla_bt_daemon.py / bt/ — 独立 LeRobot 升级通道。
⚠️ 勿与 vla_bt_daemon / mcp_http_bridge 同时跑（抢手臂）。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

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

RAW_FORMAT_VERSION = 1
IMAGE_KEY = "observation.images.head"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

_active_save_proc: Optional[mp.Process] = None


def _write_npz_file(ep_path: str, images, states, actions, compress: bool) -> None:
    """子进程内写 npz，便于 Ctrl+C 时 terminate 子进程（主进程卡在 C 扩展里无法响应 SIGINT）。"""
    if compress:
        np.savez_compressed(ep_path, images=images, state=states, action=actions)
    else:
        np.savez(ep_path, images=images, state=states, action=actions)


def _terminate_save_subprocess() -> None:
    global _active_save_proc
    proc = _active_save_proc
    if proc is None or not proc.is_alive():
        _active_save_proc = None
        return
    rospy.logwarn("中断写盘子进程（npz 可能不完整，请删除该文件）...")
    proc.terminate()
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2.0)
    _active_save_proc = None


def _sigint_handler(signum, frame) -> None:
    _terminate_save_subprocess()
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass
    raise KeyboardInterrupt


def _run_save_subprocess(ep_path: str, images, states, actions, compress: bool) -> None:
    global _active_save_proc
    ctx = mp.get_context("fork")
    proc = ctx.Process(
        target=_write_npz_file,
        args=(ep_path, images, states, actions, compress),
        daemon=True,
    )
    _active_save_proc = proc
    proc.start()
    try:
        while proc.is_alive():
            proc.join(timeout=0.3)
    finally:
        _active_save_proc = None
    if proc.exitcode != 0:
        raise RuntimeError("写盘子进程异常退出 code=%s" % proc.exitcode)

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
                        IMAGE_KEY: img,
                        STATE_KEY: state.astype(np.float32),
                        ACTION_KEY: action.astype(np.float32),
                    }
                )
            elapsed = time.time() - t0
            sleep_t = self.period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


# ---------------------------------------------------------------------------
# 单 Episode：moveit_auto_grasp 管线（像素级复用）
# ---------------------------------------------------------------------------


class _MoveGroupStub:
    """抓取 IK 走 /compute_ik 服务时，不必连 move_group action server。"""

    def set_end_effector_link(self, _link: str) -> None:
        pass

    def set_pose_reference_frame(self, _frame: str) -> None:
        pass


def _wait_for_compute_ik(timeout_sec: float = 30.0) -> str:
    """Harvester 必需：/compute_ik 服务（与 move_group action server 不同）。"""
    for svc in ("/compute_ik", "/move_group/compute_ik"):
        try:
            rospy.wait_for_service(svc, timeout=timeout_sec)
            rospy.loginfo("IK 服务已就绪: %s", svc)
            return svc
        except rospy.ROSException:
            continue
    rospy.logerr(
        "无法连接 /compute_ik（%.0fs 超时）。\n"
        "请确认 NUC 终端 2 已运行:\n"
        "  roslaunch kuavo_arm_moveit_config move_group.launch\n"
        "验证: rosservice list | grep compute_ik",
    )
    sys.exit(1)


def _load_moveit_arm_groups() -> Tuple[Any, Any]:
    """优先 MoveGroupCommander；action server 异常时回退 stub + /compute_ik（实机常用）。"""
    _wait_for_compute_ik(timeout_sec=30.0)
    rospy.loginfo("尝试连接 MoveGroupCommander（可选，失败不影响抓取）...")
    try:
        left_arm = moveit_commander.MoveGroupCommander("left_arm", wait_for_servers=5.0)
        right_arm = moveit_commander.MoveGroupCommander("right_arm", wait_for_servers=5.0)
        for grp in (left_arm, right_arm):
            grp.set_pose_reference_frame("base_link")
        rospy.loginfo("MoveGroupCommander 已连接")
        return left_arm, right_arm
    except RuntimeError as exc:
        rospy.logwarn(
            "MoveGroupCommander 不可用 (%s)。\n"
            "  → 改用 /compute_ik 服务路径（你之前成功日志里的 service / zarm_* 即此路径）。\n"
            "  → 若反复出现，请重启终端 2 move_group.launch，并清理僵尸节点:\n"
            "       rosnode list | grep move_group_commander_wrappers",
            exc,
        )
        stub = _MoveGroupStub()
        return stub, stub


def _ensure_arm_trajectory_mode():
    """每条 Episode 前切回外部轨迹模式；_run_grasp_sequence 结束会设 control_mode=0。"""
    try:
        rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
            changeArmCtrlModeRequest(control_mode=2)
        )
    except Exception as exc:
        rospy.logwarn("arm_traj_change_mode(2): %s", exc)


def _init_ros_for_harvest():
    if not rospy.core.is_initialized():
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("lerobot_data_harvester", anonymous=True)
    rospy.Subscriber("/joint_states", JointState, mag.joint_states_callback, queue_size=1)
    deadline = time.time() + 15.0
    while not mag.has_joint_states and not rospy.is_shutdown() and time.time() < deadline:
        rospy.sleep(0.05)
    if not mag.has_joint_states:
        rospy.logwarn(
            "harvester: /joint_states 未对齐。请确认 kuavo_state_publisher.py 已启动。"
        )
    mag.last_commanded_joints_rad = np.copy(mag.current_joints_rad)
    _ensure_arm_trajectory_mode()


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
    raw_x = rng.uniform(x_range[0], x_range[1])
    use_left = rng.random() > 0.5
    if use_left:
        raw_y = rng.uniform(y_left_range[0], y_left_range[1])
    else:
        raw_y = rng.uniform(y_right_range[0], y_right_range[1])
    return [raw_x] * 10, [raw_y] * 10


def run_grasp_episode(
    arm_pub: rospy.Publisher,
    left_arm,
    right_arm,
    x_hist: List[float],
    y_hist: List[float],
    recorder: HighFreqRecorder,
    min_frames: int = 50,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """调用 _run_grasp_sequence 并 50Hz 录制。返回 (success, frames)。"""
    _ensure_arm_trajectory_mode()
    recorder.start()
    try:
        mag._run_grasp_sequence(left_arm, right_arm, arm_pub, x_hist, y_hist)
    except KeyboardInterrupt:
        recorder.stop()
        rospy.logwarn("抓取过程中断")
        raise
    except Exception as exc:
        rospy.logerr("Episode 异常: %s", exc)
        return False, recorder.stop()
    frames = recorder.stop()
    ok = len(frames) >= min_frames
    if not ok:
        rospy.logwarn("Episode 帧数不足 (%d < %d)，视为失败", len(frames), min_frames)
    return ok, frames


# ---------------------------------------------------------------------------
# Raw npz 写入（NUC Python 3.8 路径）
# ---------------------------------------------------------------------------


def _ensure_raw_layout(output_dir: str) -> Tuple[str, str, str]:
    episodes_dir = os.path.join(output_dir, "episodes")
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(episodes_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    return output_dir, episodes_dir, meta_dir


def _check_output_writable(output_dir: str) -> None:
    """启动前检查输出目录可写，避免抓取成功后才 PermissionError。"""
    output_dir = os.path.expanduser(output_dir)
    _, episodes_dir, meta_dir = _ensure_raw_layout(output_dir)
    probe = os.path.join(episodes_dir, ".write_probe_%s" % os.getpid())
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except PermissionError:
        owner = "unknown"
        try:
            st = os.stat(output_dir)
            import pwd

            owner = pwd.getpwuid(st.st_uid).pw_name
        except Exception:
            pass
        raise PermissionError(
            "无法写入 %s（属主 %s，当前用户 %s）。"
            "请 sudo chown -R $(whoami):$(whoami) %s，"
            "或 sudo su 后以 root 运行 harvester。"
            % (output_dir, owner, os.getenv("USER", "?"), output_dir)
        )
    jsonl_path = os.path.join(meta_dir, "episodes.jsonl")
    if os.path.isfile(jsonl_path) and not os.access(jsonl_path, os.W_OK):
        raise PermissionError(
            "无法追加 %s。请 chown 输出目录或换 --output-dir。" % jsonl_path
        )


def _write_dataset_info(meta_dir: str, fps: int, image_shape: List[int]):
    info_path = os.path.join(meta_dir, "dataset_info.json")
    if os.path.isfile(info_path):
        return
    info = {
        "format_version": RAW_FORMAT_VERSION,
        "fps": fps,
        "robot_type": "kuavo",
        "joint_names": list(mag.joint_names_14),
        "image_key": IMAGE_KEY,
        "state_key": STATE_KEY,
        "action_key": ACTION_KEY,
        "image_shape": image_shape,
        "repo_id": "local/kuavo_vla_grasp",
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def save_raw_episode(
    output_dir: str,
    episode_index: int,
    frames: List[Dict[str, Any]],
    task: str,
    fps: int,
    extra_meta: Optional[Dict[str, Any]] = None,
    compress: bool = True,
) -> str:
    """保存单个 Episode 为 npz + 追加 episodes.jsonl。返回 npz 相对路径。"""
    _, episodes_dir, meta_dir = _ensure_raw_layout(output_dir)

    images = np.stack([f[IMAGE_KEY] for f in frames], axis=0).astype(np.uint8)
    states = np.stack([f[STATE_KEY] for f in frames], axis=0).astype(np.float32)
    actions = np.stack([f[ACTION_KEY] for f in frames], axis=0).astype(np.float32)
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    _write_dataset_info(meta_dir, fps, [h, w, c])

    ep_name = "ep_%06d.npz" % episode_index
    ep_path = os.path.join(episodes_dir, ep_name)
    rospy.loginfo(
        "正在写入 %s (%d 帧，%dx%d，%s)...",
        ep_name,
        len(frames),
        w,
        h,
        "zlib 压缩" if compress else "无压缩",
    )
    t0 = time.time()
    if compress:
        _run_save_subprocess(ep_path, images, states, actions, compress=True)
    else:
        _run_save_subprocess(ep_path, images, states, actions, compress=False)
    rospy.loginfo("写入完成，耗时 %.1fs", time.time() - t0)

    rel_path = os.path.join("episodes", ep_name)
    record = {
        "episode_index": episode_index,
        "file": rel_path,
        "task": task,
        "num_frames": len(frames),
        "fps": fps,
    }
    if extra_meta:
        record.update(extra_meta)

    jsonl_path = os.path.join(meta_dir, "episodes.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return rel_path


# ---------------------------------------------------------------------------
# LeRobot v3.0 直接写入（Python 3.12+ 可选）
# ---------------------------------------------------------------------------


def _lerobot_features(joint_names: List[str], image_shape: Tuple[int, int, int]):
    h, w, c = image_shape
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
    }


def create_lerobot_dataset(output_dir: str, fps: int, image_shape: Tuple[int, int, int]):
    if LeRobotDataset is None:
        raise RuntimeError(
            "lerobot 未安装。NUC 请用 --raw-only；Orin 上 pip install lerobot 后再 --lerobot，"
            "或采完 raw 后运行 lerobot_pack_dataset.py"
        )
    joint_names = list(mag.joint_names_14)
    dataset = LeRobotDataset.create(
        repo_id="local/kuavo_vla_grasp",
        fps=fps,
        features=_lerobot_features(joint_names, image_shape),
        robot_type="kuavo",
        root=output_dir,
        use_videos=True,
    )
    return dataset


def append_episode_to_dataset(dataset, frames: List[Dict[str, Any]], task: str):
    for frame in frames:
        dataset.add_frame(
            {
                IMAGE_KEY: frame[IMAGE_KEY],
                STATE_KEY: frame[STATE_KEY],
                ACTION_KEY: frame[ACTION_KEY],
                "task": task,
            }
        )
    dataset.save_episode()


def _episode_target_meta(x_hist: List[float], y_hist: List[float]) -> Dict[str, Any]:
    raw_x = float(np.median(x_hist))
    raw_y = float(np.median(y_hist))
    return {
        "raw_x": raw_x,
        "raw_y": raw_y,
        "is_left_arm": raw_y > 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Kuavo LeRobot data harvester")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="首个 npz 序号（续采时用，如已有 ep_000009 则 --start-index 10）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/lerobot_kuavo_grasp_raw"),
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--image-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument(
        "--use-vision",
        action="store_true",
        default=True,
        help="每 Episode 用 YOLO 10 帧采点（默认开）",
    )
    parser.add_argument("--no-vision", action="store_false", dest="use_vision")
    parser.add_argument(
        "--reposition-pause",
        type=float,
        default=10.0,
        help="Episode 间隔秒数（便于人工挪动水瓶，默认 10s）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-frames",
        type=int,
        default=50,
        help="Episode 最少有效帧数（默认 50 ≈ 1s @50Hz）",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="npz 不 zlib 压缩，写盘快 5~10 倍（文件约 3~4GB/ep，需预留磁盘）",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--raw-only",
        action="store_true",
        default=None,
        help="写 raw npz（NUC 默认，不依赖 lerobot）",
    )
    mode.add_argument(
        "--lerobot",
        action="store_true",
        help="直接写 LeRobot v3（需 Python>=3.12 + pip install lerobot）",
    )
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    signal.signal(signal.SIGINT, _sigint_handler)

    if args.lerobot:
        use_raw = False
    elif args.raw_only is True:
        use_raw = True
    else:
        # 默认：无 lerobot 库则 raw，否则仍默认 raw（NUC 工作流）
        use_raw = True

    if not use_raw and LeRobotDataset is None:
        print(
            "错误: 指定了 --lerobot 但未安装 lerobot。"
            "请 pip install lerobot 或改用 --raw-only。",
            file=sys.stderr,
        )
        sys.exit(1)

    if use_raw:
        try:
            _check_output_writable(args.output_dir)
        except PermissionError as exc:
            print("错误: %s" % exc, file=sys.stderr)
            sys.exit(1)

    rng = random.Random(args.seed)
    _init_ros_for_harvest()

    arm_pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    rospy.sleep(0.3)

    pos, vel, effort = build_open_cmd()
    mag.call_leju_claw(pos, vel, effort, tag="harvest-init-open")
    mag.execute_dual_arm_init_home(arm_pub)

    rospy.loginfo("加载 MoveIt 规划器...")
    left_arm, right_arm = _load_moveit_arm_groups()

    recorder = HighFreqRecorder(fps=args.fps, image_topic=args.image_topic)
    rospy.loginfo(
        "收割模式: %s | 输出: %s | %d ep @ %dHz",
        "raw npz" if use_raw else "lerobot v3",
        args.output_dir,
        args.episodes,
        args.fps,
    )

    dataset = None
    default_image_shape = (480, 640, 3)

    x_range = (0.38, 0.52)
    y_left_range = (0.02, 0.12)
    y_right_range = (-0.12, -0.02)

    success_count = 0
    saved_episode_index = args.start_index

    for ep in range(args.episodes):
        rospy.loginfo("========== Episode %d / %d ==========", ep + 1, args.episodes)
        x_hist, y_hist = _collect_or_random_targets(
            args.use_vision, rng, x_range, y_left_range, y_right_range,
        )
        ok, frames = run_grasp_episode(
            arm_pub,
            left_arm,
            right_arm,
            x_hist,
            y_hist,
            recorder,
            min_frames=args.min_frames,
        )

        if ok and frames:
            task = "grasp bottle ep%d" % ep
            target_meta = _episode_target_meta(x_hist, y_hist)

            if use_raw:
                rel = save_raw_episode(
                    args.output_dir,
                    saved_episode_index,
                    frames,
                    task,
                    args.fps,
                    extra_meta=target_meta,
                    compress=not args.no_compress,
                )
                rospy.loginfo(
                    "Episode %d 成功 → raw %s (%d 帧 @ %dHz)",
                    ep,
                    rel,
                    len(frames),
                    args.fps,
                )
            else:
                if dataset is None:
                    h, w, c = frames[0][IMAGE_KEY].shape
                    dataset = create_lerobot_dataset(
                        args.output_dir, args.fps, (h, w, c)
                    )
                append_episode_to_dataset(dataset, frames, task)
                rospy.loginfo(
                    "Episode %d 成功 → lerobot (%d 帧 @ %dHz)",
                    ep,
                    len(frames),
                    args.fps,
                )

            success_count += 1
            saved_episode_index += 1
        else:
            rospy.logwarn("Episode %d 失败或无可写帧", ep)

        if ep + 1 < args.episodes:
            rospy.loginfo("等待 %.1fs — 请将水瓶移动到新位置...", args.reposition_pause)
            time.sleep(args.reposition_pause)

    if dataset is not None:
        dataset.finalize()

    if use_raw:
        rospy.loginfo(
            "raw 收割完成: %d / %d episodes 成功 → %s\n"
            "下一步（Orin conda 3.12）:\n"
            "  python3 lerobot_pack_dataset.py --raw-dir %s --output-dir ~/lerobot_kuavo_grasp",
            success_count,
            args.episodes,
            args.output_dir,
            args.output_dir,
        )
    else:
        rospy.loginfo(
            "LeRobot 收割完成: %d / %d episodes 成功，数据目录: %s",
            success_count,
            args.episodes,
            args.output_dir,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\n已中断（Ctrl+C）。若曾 Ctrl+Z 挂起，请 kill 该进程而非仅 suspend。\n")
        sys.exit(130)
