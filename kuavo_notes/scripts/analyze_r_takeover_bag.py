#!/usr/bin/env python3
"""Parse r_takeover.bag around RL takeover (R key).

Requires ROS1: pip install rosbag (or use system python with rospy/rosbag).

Usage:
  cd ~/kuavo_ws
  python3 ~/kuavo_all/kuavo_notes/scripts/analyze_r_takeover_bag.py r_takeover.bag
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import rosbag
except ImportError as e:
    print("Need ROS1 rosbag module. Source your workspace first, e.g.:")
    print("  source /opt/ros/noetic/setup.bash && source ~/kuavo_ws/devel/setup.bash")
    print(f"Import error: {e}")
    sys.exit(1)

OBS_LABELS = [
    ("bodyAngVel", 0, 3),
    ("gravity_body", 3, 6),
    ("command", 6, 9),
    ("jointPos", 9, 35),
    ("jointVel", 35, 61),
    ("referenceJointPos", 61, 87),
    ("action", 87, 113),
    ("commandPhase", 113, 115),
]


def fmt(v):
    return [round(float(x), 4) for x in v]


def ref_is_zero(d, start=61, end=87, eps=1e-6):
    return all(abs(d[i]) < eps for i in range(start, end))


def print_obs_summary(d, tag):
    print(f"\n=== {tag} (len={len(d)}) ===")
    for name, a, b in OBS_LABELS:
        seg = d[a:b]
        if len(seg) <= 6:
            print(f"  {name:20s}: {fmt(seg)}")
        else:
            print(f"  {name:20s}: head6={fmt(seg[:6])}  max|.|={max(abs(x) for x in seg):.4f}")


def main():
    bag_path = Path(sys.argv[1] if len(sys.argv) > 1 else "r_takeover.bag")
    if not bag_path.is_file():
        print(f"Bag not found: {bag_path}")
        sys.exit(1)

    obs_msgs = []
    act_msgs = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, t in bag.read_messages(
            topics=["/rl_controller/singleInputData", "/rl_controller/actions"]
        ):
            if topic.endswith("singleInputData"):
                obs_msgs.append((t.to_sec(), list(msg.data)))
            else:
                act_msgs.append((t.to_sec(), list(msg.data)))

    print(f"Bag: {bag_path}")
    print(f"obs messages: {len(obs_msgs)}, action messages: {len(act_msgs)}")
    if not obs_msgs:
        print("No singleInputData in bag.")
        sys.exit(1)

    t0 = obs_msgs[0][0]
    # R takeover: referenceJointPos block switches from all-zero (WBC) to CSV-driven (RL)
    r_idx = None
    for i, (_, d) in enumerate(obs_msgs):
        if len(d) < 115:
            continue
        if not ref_is_zero(d) and (i == 0 or ref_is_zero(obs_msgs[i - 1][1])):
            r_idx = i
            break

    if r_idx is None:
        print("\nCould not detect R takeover (referenceJointPos never left all-zero).")
        print("Showing first / middle / last obs instead.")
        for idx in {0, len(obs_msgs) // 2, len(obs_msgs) - 1}:
            print_obs_summary(obs_msgs[idx][1], f"obs@{idx} t={obs_msgs[idx][0]-t0:.3f}s")
        return

    t_r = obs_msgs[r_idx][0] - t0
    print(f"\nDetected R takeover at obs index {r_idx}, t={t_r:.3f}s from bag start")

    for offset in range(-2, 5):
        j = r_idx + offset
        if 0 <= j < len(obs_msgs):
            print_obs_summary(obs_msgs[j][1], f"obs[{j}] t={obs_msgs[j][0]-t0:.3f}s (offset {offset:+d})")

    # Match actions by nearest timestamp to each obs window
    print("\n--- Nearest actions around takeover ---")
    for offset in range(-2, 5):
        j = r_idx + offset
        if not (0 <= j < len(obs_msgs)):
            continue
        t_obs = obs_msgs[j][0]
        best = min(act_msgs, key=lambda x: abs(x[0] - t_obs))
        print(f"  offset {offset:+d} t={t_obs-t0:.3f}s  actions head6={fmt(best[1][:6])}  max|.|={max(abs(x) for x in best[1]):.4f}")

    # Pre-R standing check: last obs before takeover with ref still zero
    pre = r_idx - 1
    if pre >= 0:
        d = obs_msgs[pre][1]
        g = d[3:6]
        print(f"\n--- Pre-R (obs[{pre}], ref still zero) gravity_body = {fmt(g)} ---")
        print("  Expected standing: ~ [0, 0, -1]. If far off, gravity obs is wrong before R.")


if __name__ == "__main__":
    main()
