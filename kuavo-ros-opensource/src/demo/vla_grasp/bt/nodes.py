#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""22 案例 VLA 抓取的 py_trees 节点。"""

import json
from typing import Callable, Optional

import py_trees
import rospy
from geometry_msgs.msg import PointStamped

from . import config
from .blackboard import BBKeys, get, set_key
from .grasp_skills import GraspPlan, GraspSkills


# ---------------------------------------------------------------------------
# 装饰器 / 工具
# ---------------------------------------------------------------------------

class BusyScope(py_trees.decorators.Decorator):
    """子树结束时（成功或失败）自动释放 working 锁；失败时清指令防 YOLO 无限重试。"""

    def __init__(self, name: str, child: py_trees.behaviour.Behaviour):
        super().__init__(name=name, child=child)

    def update(self):
        return self.decorated.status

    def terminate(self, new_status):
        set_key(BBKeys.WORKING, False)
        if new_status == py_trees.common.Status.FAILURE:
            set_key(BBKeys.COMMAND, None)
            set_key(BBKeys.PLAN, None)
            rospy.logwarn("VLA BT: grasp pipeline failed, command cleared (re-issue to retry)")
        super().terminate(new_status)


class GraspMotion(py_trees.behaviour.Behaviour):
    """调用 GraspSkills 的单个动作步骤。"""

    def __init__(self, name: str, step_fn: Callable, need_plan: bool = True):
        super().__init__(name=name)
        self.step_fn = step_fn
        self.need_plan = need_plan

    def update(self):
        skills: GraspSkills = get(BBKeys.SKILLS)
        if skills is None:
            return py_trees.common.Status.FAILURE
        try:
            if self.need_plan:
                plan: Optional[GraspPlan] = get(BBKeys.PLAN)
                if plan is None:
                    return py_trees.common.Status.FAILURE
                self.step_fn(skills, plan)
            else:
                self.step_fn(skills)
        except Exception as exc:
            rospy.logerr("VLA BT motion [%s] failed: %s", self.name, exc)
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# 指令与状态
# ---------------------------------------------------------------------------

class HasActionCommand(py_trees.behaviour.Behaviour):
    """黑板指令 action 匹配时 SUCCESS。"""

    def __init__(self, action: str, name: Optional[str] = None):
        super().__init__(name=name or f"HasAction[{action}]")
        self.action = action

    def update(self):
        cmd = get(BBKeys.COMMAND)
        if not cmd or cmd.get("action") != self.action:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class EnsureNotBusy(py_trees.behaviour.Behaviour):
    def __init__(self, name="EnsureNotBusy"):
        super().__init__(name=name)

    def update(self):
        if get(BBKeys.WORKING):
            rospy.logwarn_throttle(2.0, "VLA BT: busy, skip command")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class MarkBusy(py_trees.behaviour.Behaviour):
    def __init__(self, name="MarkBusy"):
        super().__init__(name=name)

    def update(self):
        set_key(BBKeys.WORKING, True)
        return py_trees.common.Status.SUCCESS


class ParseGrabFields(py_trees.behaviour.Behaviour):
    def __init__(self, name="ParseGrabFields"):
        super().__init__(name=name)

    def update(self):
        cmd = get(BBKeys.COMMAND) or {}
        target = cmd.get("target", "目标")
        set_key(BBKeys.TARGET, target)
        rospy.loginfo("VLA BT: grab target = %s", target)
        return py_trees.common.Status.SUCCESS


class ParseChatFields(py_trees.behaviour.Behaviour):
    def __init__(self, name="ParseChatFields"):
        super().__init__(name=name)

    def update(self):
        cmd = get(BBKeys.COMMAND) or {}
        speech = cmd.get("text") or cmd.get("response") or cmd.get("reply") or ""
        if not speech:
            rospy.logwarn("VLA BT: chat command missing text")
            return py_trees.common.Status.FAILURE
        set_key(BBKeys.SPEECH, speech)
        rospy.loginfo("VLA BT: chat text = %s", speech[:80])
        return py_trees.common.Status.SUCCESS


class ClearCommand(py_trees.behaviour.Behaviour):
    def __init__(self, name="ClearCommand"):
        super().__init__(name=name)

    def update(self):
        set_key(BBKeys.COMMAND, None)
        set_key(BBKeys.PLAN, None)
        return py_trees.common.Status.SUCCESS


class Idle(py_trees.behaviour.Behaviour):
    def __init__(self, name="Idle"):
        super().__init__(name=name)

    def update(self):
        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# YOLO 订阅（非阻塞，避免 wait_for_message 空等拖慢采集）
# ---------------------------------------------------------------------------

_yolo_latest_msg = None


def _yolo_target_cb(msg: PointStamped):
    global _yolo_latest_msg
    if config.YOLO_X_MIN <= msg.point.x <= config.YOLO_X_MAX:
        _yolo_latest_msg = msg


def ensure_yolo_subscriber():
    if not hasattr(ensure_yolo_subscriber, "_sub"):
        ensure_yolo_subscriber._sub = rospy.Subscriber(
            config.YOLO_TOPIC, PointStamped, _yolo_target_cb, queue_size=1
        )


# ---------------------------------------------------------------------------
# 感知 / 规划
# ---------------------------------------------------------------------------

class CheckYoloAlive(py_trees.behaviour.Behaviour):
    """确认 YOLO 目标话题有发布者且能收到一帧。"""

    def __init__(self, name="CheckYoloAlive"):
        super().__init__(name=name)

    def update(self):
        ensure_yolo_subscriber()
        topic = config.YOLO_TOPIC
        pubs = [t for t, _ in rospy.get_published_topics() if t == topic]
        if not pubs:
            rospy.logerr("VLA BT: no publisher on %s", topic)
            return py_trees.common.Status.FAILURE
        global _yolo_latest_msg
        if _yolo_latest_msg is not None:
            return py_trees.common.Status.SUCCESS
        try:
            rospy.wait_for_message(topic, PointStamped, timeout=2.0)
            return py_trees.common.Status.SUCCESS
        except rospy.ROSException:
            rospy.logerr("VLA BT: timeout waiting first message on %s", topic)
            return py_trees.common.Status.FAILURE


class CollectYoloTarget(py_trees.behaviour.Behaviour):
    """增量采集中值：订阅缓存最新帧，每 tick 取一帧，够样本后 SUCCESS。"""

    def __init__(self, name="CollectYoloTarget"):
        super().__init__(name=name)
        self._x_hist = []
        self._y_hist = []
        self._deadline = None

    def initialise(self):
        ensure_yolo_subscriber()
        self._x_hist = []
        self._y_hist = []
        self._deadline = rospy.Time.now() + rospy.Duration(config.YOLO_COLLECT_TIMEOUT_SEC)

    def update(self):
        global _yolo_latest_msg
        if rospy.Time.now() > self._deadline:
            rospy.logerr(
                "VLA BT: only got %d/%d YOLO samples",
                len(self._x_hist), config.YOLO_SAMPLE_COUNT,
            )
            return py_trees.common.Status.FAILURE

        if _yolo_latest_msg is not None:
            msg = _yolo_latest_msg
            _yolo_latest_msg = None
            self._x_hist.append(msg.point.x)
            self._y_hist.append(msg.point.y)
            rospy.loginfo(
                "VLA BT: sample %d/%d x=%.3f y=%.3f",
                len(self._x_hist), config.YOLO_SAMPLE_COUNT,
                msg.point.x, msg.point.y,
            )

        if len(self._x_hist) < config.YOLO_SAMPLE_COUNT:
            return py_trees.common.Status.RUNNING

        import numpy as np
        raw_x = float(np.median(self._x_hist))
        raw_y = float(np.median(self._y_hist))
        set_key(BBKeys.RAW_X, raw_x)
        set_key(BBKeys.RAW_Y, raw_y)
        rospy.loginfo("VLA BT: median target x=%.3f y=%.3f", raw_x, raw_y)
        return py_trees.common.Status.SUCCESS


class PlanGraspTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self, name="PlanGraspTrajectory"):
        super().__init__(name=name)

    def update(self):
        skills: GraspSkills = get(BBKeys.SKILLS)
        raw_x = get(BBKeys.RAW_X)
        raw_y = get(BBKeys.RAW_Y)
        if raw_x is None or raw_y is None:
            return py_trees.common.Status.FAILURE
        plan = skills.plan_grasp(raw_x, raw_y)
        if plan is None:
            return py_trees.common.Status.FAILURE
        set_key(BBKeys.PLAN, plan)
        arm = "left" if plan.is_left_arm else "right"
        rospy.loginfo("VLA BT: plan ready, arm=%s", arm)
        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

class SpeakTTS(py_trees.behaviour.Behaviour):
    """POST 到 tts_server（默认 :5000/tts）。"""

    def __init__(self, name="SpeakTTS"):
        super().__init__(name=name)

    def update(self):
        speech = get(BBKeys.SPEECH, "")
        if not speech:
            return py_trees.common.Status.FAILURE
        try:
            import urllib.request
            payload = json.dumps({"text": speech}).encode("utf-8")
            req = urllib.request.Request(
                config.TTS_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=config.TTS_TIMEOUT_SEC) as resp:
                if resp.status != 200:
                    return py_trees.common.Status.FAILURE
            rospy.loginfo("VLA BT: TTS dispatched")
            return py_trees.common.Status.SUCCESS
        except Exception as exc:
            rospy.logerr("VLA BT: TTS failed: %s", exc)
            return py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def motion(name: str, fn: Callable, need_plan: bool = True) -> GraspMotion:
    return GraspMotion(name=name, step_fn=fn, need_plan=need_plan)


def parse_master_command_json(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        rospy.logerr("VLA BT: bad JSON: %s", exc)
        return None
