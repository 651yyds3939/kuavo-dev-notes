#!/usr/bin/env python3
import sys
from pathlib import Path

import rosbag

OBS = [
    ("bodyAngVel", 0, 3), ("gravity_body", 3, 6), ("command", 6, 9),
    ("jointPos", 9, 35), ("jointVel", 35, 61), ("referenceJointPos", 61, 87),
    ("action", 87, 113), ("commandPhase", 113, 115),
]

def fmt(v):
    return [round(float(x), 4) for x in v]

def ref_zero(d, eps=1e-6):
    return all(abs(d[i]) < eps for i in range(61, 87))

def show(d, tag):
    print(f"\n=== {tag} (len={len(d)}) ===")
    for name, a, b in OBS:
        seg = d[a:b]
        if len(seg) <= 6:
            print(f"  {name:20s}: {fmt(seg)}")
        else:
            print(f"  {name:20s}: head6={fmt(seg[:6])}  max|.|={max(abs(x) for x in seg):.4f}")

bag_path = Path(sys.argv[1] if len(sys.argv) > 1 else "r_takeover.bag")
obs_msgs, act_msgs = [], []
with rosbag.Bag(str(bag_path), "r") as bag:
    for topic, msg, t in bag.read_messages(
        topics=["/rl_controller/singleInputData", "/rl_controller/actions"]
    ):
        (obs_msgs if topic.endswith("singleInputData") else act_msgs).append(
            (t.to_sec(), list(msg.data))
        )

print(f"Bag: {bag_path}, obs={len(obs_msgs)}, actions={len(act_msgs)}")
t0 = obs_msgs[0][0]
r_idx = None
for i, (_, d) in enumerate(obs_msgs):
    if len(d) < 115:
        continue
    if not ref_zero(d) and (i == 0 or ref_zero(obs_msgs[i - 1][1])):
        r_idx = i
        break

if r_idx is None:
    print("No R takeover detected (referenceJointPos never left zero).")
    for idx in (0, len(obs_msgs)//2, len(obs_msgs)-1):
        show(obs_msgs[idx][1], f"obs@{idx}")
    sys.exit(0)

print(f"R takeover: obs index {r_idx}, t={obs_msgs[r_idx][0]-t0:.3f}s")
for off in range(-2, 5):
    j = r_idx + off
    if 0 <= j < len(obs_msgs):
        show(obs_msgs[j][1], f"obs[{j}] offset {off:+d}")

print("\n--- Nearest actions ---")
for off in range(-2, 5):
    j = r_idx + off
    if 0 <= j < len(obs_msgs):
        t = obs_msgs[j][0]
        a = min(act_msgs, key=lambda x: abs(x[0]-t))[1]
        print(f"  off {off:+d}: head6={fmt(a[:6])}  max|.|={max(abs(x) for x in a):.4f}")

pre = r_idx - 1
if pre >= 0:
    g = obs_msgs[pre][1][3:6]
    print(f"\nPre-R gravity_body: {fmt(g)}  (expect ~ [0,0,-1])")
