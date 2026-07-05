#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot ACT checkpoint 预检（无需 ROS）。

在真机开环前于 NUC / PC 上运行，确认权重、preprocessor、前向推理正常。

用法:
  conda activate lerobot
  unset PYTHONPATH
  python3 src/demo/vla_grasp/mcp/lerobot_act_verify_checkpoint.py \\
      --checkpoint src/demo/vla_grasp/pretrained_model
"""

from __future__ import annotations

import argparse
import sys
import time


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeRobot ACT checkpoint 预检（无 ROS）")
    p.add_argument(
        "--checkpoint",
        default="src/demo/vla_grasp/pretrained_model",
        help="pretrained_model 目录（含 model.safetensors）",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="推理设备（NUC 无 GPU 时用 cpu）",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="dummy 前向预热次数",
    )
    return p.parse_args()


def _resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> int:
    args = _parse_args()

    try:
        import numpy as np
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError as exc:
        print(
            "错误: 未安装 lerobot / torch。\n"
            "  conda activate lerobot\n"
            "  pip install lerobot torch opencv-python-headless\n"
            "详情: %s" % exc,
            file=sys.stderr,
        )
        return 1

    ckpt = args.checkpoint
    device = _resolve_device(args.device)
    print("checkpoint : %s" % ckpt)
    print("device     : %s" % device)

    t0 = time.time()
    policy = ACTPolicy.from_pretrained(ckpt).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("load OK (%.2fs)" % (time.time() - t0))

    cfg = policy.config
    img_shape = tuple(cfg.input_features["observation.images.head"].shape)
    state_dim = int(cfg.input_features["observation.state"].shape[0])
    print("image shape (CHW): %s" % (img_shape,))
    print("state dim        : %d" % state_dim)
    print("chunk_size       : %s" % getattr(cfg, "chunk_size", "?"))
    print("n_action_steps   : %s" % getattr(cfg, "n_action_steps", "?"))

    c, h, w = img_shape
    dummy_rgb = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
    dummy_state = np.zeros(state_dim, dtype=np.float32)

    obs_np = {
        "observation.images.head": dummy_rgb,
        "observation.state": dummy_state,
    }

    try:
        from lerobot.policies.utils import prepare_observation_for_inference

        obs_t = prepare_observation_for_inference(obs_np, device)
    except ImportError:
        obs_t = {
            "observation.images.head": torch.from_numpy(dummy_rgb)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            / 255.0,
            "observation.state": torch.from_numpy(dummy_state)
            .float()
            .unsqueeze(0)
            .to(device),
        }

    latencies = []
    with torch.inference_mode():
        for i in range(max(1, args.warmup)):
            policy.reset()
            t1 = time.time()
            proc = preprocessor(obs_t)
            action = postprocessor(policy.select_action(proc)).squeeze(0).cpu().numpy()
            latencies.append((time.time() - t1) * 1000.0)

    print("dummy action shape: %s" % (action.shape,))
    print("dummy action[:4] : %s" % np.array2string(action[:4], precision=4))
    print(
        "infer latency ms : mean=%.1f  max=%.1f"
        % (np.mean(latencies), np.max(latencies))
    )
    print("PASS — checkpoint 可加载且前向正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
