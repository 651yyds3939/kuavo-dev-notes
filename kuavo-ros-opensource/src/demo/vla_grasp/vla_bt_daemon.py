#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22 案例 VLA 抓取 — py_trees 版守护进程。

可替代 vla_auto_grasp_daemon.py；流程用行为树编排，支持 grab + chat。
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import rospy
import py_trees
from std_msgs.msg import String

from bt import config
from bt.blackboard import BBKeys, get, init_blackboard, set_key
from bt.grasp_skills import GraspSkills
from bt.nodes import parse_master_command_json
from bt.trees import build_vla_root_tree


class VLABtDaemon:
    def __init__(self):
        rospy.init_node("vla_bt_daemon", anonymous=True)
        config.load_from_ros_params()

        rospy.loginfo("========== VLA BT daemon (py_trees) ==========")
        self.skills = GraspSkills()
        init_blackboard(self.skills)
        self.tree = build_vla_root_tree()
        self._last_root_status = None
        self._last_grab_cmd_time = 0.0

        rospy.Subscriber("/vla/master_command", String, self._on_master_command, queue_size=1)
        rospy.loginfo("tick_hz=%.1f yolo_topic=%s tts_url=%s", config.BT_TICK_HZ, config.YOLO_TOPIC, config.TTS_URL)
        rospy.loginfo("\n%s", py_trees.display.unicode_tree(self.tree))

    def _on_master_command(self, msg):
        if get(BBKeys.WORKING):
            rospy.logwarn_throttle(1.0, "VLA BT: ignore command while working")
            return
        cmd = parse_master_command_json(msg.data)
        if cmd is None:
            return
        if cmd.get("action") == "grab":
            now = rospy.Time.now().to_sec()
            if now - self._last_grab_cmd_time < config.GRAB_CMD_MIN_INTERVAL_SEC:
                rospy.logwarn(
                    "VLA BT: ignore grab (debounce %.0fs, Orin ASR 误触发防护)",
                    config.GRAB_CMD_MIN_INTERVAL_SEC,
                )
                return
            self._last_grab_cmd_time = now
        set_key(BBKeys.COMMAND, cmd)
        rospy.loginfo("VLA BT: queued command %s", cmd)

    def _log_status_if_changed(self):
        if not config.LOG_STATUS_CHANGE:
            return
        status = self.tree.status
        if status != self._last_root_status:
            rospy.loginfo("VLA BT root -> %s", status)
            self._last_root_status = status

    def spin(self):
        rate = rospy.Rate(config.BT_TICK_HZ)
        while not rospy.is_shutdown():
            self.tree.tick_once()
            self._log_status_if_changed()
            rate.sleep()


if __name__ == "__main__":
    daemon = VLABtDaemon()
    daemon.spin()
