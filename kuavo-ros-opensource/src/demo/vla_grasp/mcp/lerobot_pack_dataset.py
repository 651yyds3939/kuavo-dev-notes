#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 NUC raw npz 数据集打包为 LeRobot v3.0 标准格式（Parquet + MP4）。

在 Orin 或开发 PC 上运行（Python 3.12+）：
  conda activate lerobot
  pip install lerobot opencv-python-headless

Orin 内存较紧（8GB 统一内存 + torch），1230 帧全分辨率易 OOM。
推荐 Orin 首次打包：
  python3 lerobot_pack_dataset.py \\
      --raw-dir ~/lerobot_kuavo_grasp_raw \\
      --output-dir ~/lerobot_kuavo_grasp \\
      --overwrite --no-video --frame-stride 2

PC / 大内存机器可全量：
  python3 lerobot_pack_dataset.py ... --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:
    print(
        "错误: 未安装 lerobot。请执行:\n"
        "  conda create -n lerobot python=3.12 -y && conda activate lerobot\n"
        "  pip install lerobot opencv-python-headless",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

IMAGE_KEY = "observation.images.head"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def load_dataset_info(raw_dir: Path) -> Dict[str, Any]:
    info_path = raw_dir / "meta" / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError("缺少 meta/dataset_info.json: %s" % info_path)
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def load_episode_index(raw_dir: Path) -> List[Dict[str, Any]]:
    jsonl_path = raw_dir / "meta" / "episodes.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError("缺少 meta/episodes.jsonl: %s" % jsonl_path)
    episodes = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    if not episodes:
        raise ValueError("episodes.jsonl 为空，无 Episode 可打包")
    return episodes


def peek_image_shape(raw_dir: Path, rel_path: str) -> Tuple[int, int, int]:
    npz_path = raw_dir / rel_path
    data = np.load(npz_path, mmap_mode="r")
    try:
        images = data["images"]
        if images.ndim != 4:
            raise ValueError("images 维度应为 (T,H,W,C)，实际: %s" % (images.shape,))
        return int(images.shape[1]), int(images.shape[2]), int(images.shape[3])
    finally:
        data.close()


def iter_episode_frames(
    raw_dir: Path,
    rel_path: str,
    frame_stride: int = 1,
    max_frames: int = 0,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """mmap 逐帧读取，避免一次性把整集 images 载入 RAM。"""
    npz_path = raw_dir / rel_path
    if not npz_path.is_file():
        raise FileNotFoundError("Episode 文件不存在: %s" % npz_path)

    data = np.load(npz_path, mmap_mode="r")
    try:
        images = data["images"]
        states = data["state"]
        actions = data["action"]
        n = int(images.shape[0])
        if states.shape[0] != n or actions.shape[0] != n:
            raise ValueError(
                "帧数不一致 images=%d state=%d action=%d" % (
                    n, states.shape[0], actions.shape[0],
                )
            )
        stride = max(1, int(frame_stride))
        indices = range(0, n, stride)
        if max_frames > 0:
            indices = list(indices)[:max_frames]
        for i in indices:
            img = np.asarray(images[i], dtype=np.uint8)
            if resize_hw is not None:
                if cv2 is None:
                    raise RuntimeError("需要 opencv: pip install opencv-python-headless")
                h, w = resize_hw
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            yield (
                img,
                np.asarray(states[i], dtype=np.float32),
                np.asarray(actions[i], dtype=np.float32),
            )
    finally:
        data.close()


def build_lerobot_features(
    joint_names: List[str],
    image_shape: Tuple[int, int, int],
    use_videos: bool,
) -> Dict[str, Any]:
    h, w, c = image_shape
    image_dtype = "video" if use_videos else "image"
    return {
        IMAGE_KEY: {
            "dtype": image_dtype,
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


def pack_raw_to_lerobot(
    raw_dir: Path,
    output_dir: Path,
    repo_id: str,
    max_episodes: int = 0,
    overwrite: bool = False,
    use_videos: bool = True,
    frame_stride: int = 1,
    max_frames: int = 0,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> int:
    info = load_dataset_info(raw_dir)
    episodes = load_episode_index(raw_dir)

    fps = int(info.get("fps", 50))
    joint_names = list(info["joint_names"])
    robot_type = info.get("robot_type", "kuavo")

    if max_episodes > 0:
        episodes = episodes[:max_episodes]

    stride = max(1, int(frame_stride))
    effective_fps = max(1, fps // stride) if stride > 1 else fps

    src_h, src_w, src_c = peek_image_shape(raw_dir, episodes[0]["file"])
    if resize_hw is not None:
        image_shape = (resize_hw[0], resize_hw[1], src_c)
    else:
        image_shape = (src_h, src_w, src_c)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                "输出目录已存在: %s\n"
                "请加 --overwrite 覆盖，或换 --output-dir。"
                % output_dir
            )
        shutil.rmtree(output_dir)

    print(
        "打包参数: use_videos=%s fps=%d stride=%d effective_fps=%d max_frames=%s resize=%s"
        % (
            use_videos, fps, stride, effective_fps, max_frames or "all",
            "%dx%d" % (resize_hw[0], resize_hw[1]) if resize_hw else "off",
        ),
        flush=True,
    )
    if resize_hw:
        print(
            "图像尺寸: %dx%d → %dx%d"
            % (src_h, src_w, image_shape[0], image_shape[1]),
            flush=True,
        )
    else:
        print("图像尺寸: %dx%dx%d" % image_shape, flush=True)

    print("正在创建 LeRobot 数据集（首次可能较慢）...", flush=True)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=effective_fps,
        features=build_lerobot_features(joint_names, image_shape, use_videos),
        robot_type=robot_type,
        root=str(output_dir),
        use_videos=use_videos,
    )
    print("数据集已创建，开始逐帧写入...", flush=True)

    packed = 0
    progress_interval = 50
    for rec in episodes:
        rel = rec["file"]
        task = rec.get("task", "grasp bottle")
        ep_idx = rec.get("episode_index", packed)
        print("Episode %d: 读取 %s" % (ep_idx, rel), flush=True)
        frame_count = 0
        for img, state, action in iter_episode_frames(
            raw_dir, rel,
            frame_stride=stride,
            max_frames=max_frames,
            resize_hw=resize_hw,
        ):
            dataset.add_frame(
                {
                    IMAGE_KEY: img,
                    STATE_KEY: state,
                    ACTION_KEY: action,
                    "task": task,
                }
            )
            frame_count += 1
            if frame_count % progress_interval == 0:
                print("  add_frame %d ..." % frame_count, flush=True)

        if frame_count == 0:
            print("跳过 %s: 无有效帧" % rel, flush=True)
            continue

        print(
            "  save_episode (%d 帧，Parquet 落盘，上位机全分辨率可能 10~30min/ep)..."
            % frame_count,
            flush=True,
        )
        t0 = time.time()
        dataset.save_episode()
        elapsed = time.time() - t0
        packed += 1
        print(
            "已打包 episode %d: %s (%d 帧，save 耗时 %.0fs)"
            % (ep_idx, rel, frame_count, elapsed),
            flush=True,
        )
        gc.collect()

    dataset.finalize()
    return packed


def main():
    parser = argparse.ArgumentParser(description="Kuavo raw npz → LeRobot v3 packer")
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/lerobot_kuavo_grasp"),
    )
    parser.add_argument("--repo-id", type=str, default="local/kuavo_vuavo_grasp")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不写 MP4，图像进 Parquet（Orin 省内存，推荐）",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="每隔 N 帧采 1 帧（Orin 可设 2 即 25Hz）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="每个 episode 最多打包帧数（0=全部，调试可设 200）",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="打包时缩小图像，如 --resize 400 640（上位机强烈推荐，可快 5~10 倍）",
    )
    args = parser.parse_args()

    if args.repo_id == "local/kuavo_vuavo_grasp":
        args.repo_id = "local/kuavo_vla_grasp"

    raw_dir = Path(os.path.expanduser(args.raw_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    if not raw_dir.is_dir():
        print("错误: raw-dir 不存在: %s" % raw_dir, file=sys.stderr)
        sys.exit(1)

    print("raw  输入: %s" % raw_dir, flush=True)
    print("lerobot 输出: %s" % output_dir, flush=True)

    resize_hw = tuple(args.resize) if args.resize else None

    n = pack_raw_to_lerobot(
        raw_dir,
        output_dir,
        repo_id=args.repo_id,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        use_videos=not args.no_video,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        resize_hw=resize_hw,
    )
    print("\n打包完成: %d episodes → %s" % (n, output_dir), flush=True)
    print("验证:")
    print(
        "  python -c \"from pathlib import Path; "
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
        "root=str(Path('%s').expanduser()); "
        "ds=LeRobotDataset(repo_id='%s', root=root); "
        "print('frames:', len(ds)); "
        "print('image:', ds[0]['observation.images.head'].shape)\""
        % (output_dir, args.repo_id)
    )


if __name__ == "__main__":
    main()
