#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""22 案例行为树结构定义。"""

import py_trees

from .grasp_skills import GraspSkills
from .nodes import (
    BusyScope,
    CheckYoloAlive,
    ClearCommand,
    CollectYoloTarget,
    EnsureNotBusy,
    HasActionCommand,
    Idle,
    MarkBusy,
    ParseChatFields,
    ParseGrabFields,
    PlanGraspTrajectory,
    SpeakTTS,
    motion,
)


def _wrap_work(child, name):
    """MarkBusy 后执行子树，失败/成功都自动解锁。"""
    return BusyScope(name=name, child=child)


def build_motion_sequence(name="RunMotions"):
    """抓取动作链：每步一个 BT 节点，便于日志定位失败点。"""
    seq = py_trees.composites.Sequence(name=name, memory=True)
    seq.add_children([
        motion("OpenClaw", GraspSkills.step_open_claw, need_plan=False),
        motion("MoveInit", GraspSkills.step_move_init, need_plan=False),
        motion("MoveReady", GraspSkills.step_move_ready),
        motion("MovePreAim", GraspSkills.step_move_pre),
        motion("MoveGrasp", GraspSkills.step_move_grasp),
        motion("CloseClaw", GraspSkills.step_close_claw),
        motion("Lift", GraspSkills.step_lift),
        motion("Retreat", GraspSkills.step_retreat),
        motion("FoldBack", GraspSkills.step_fold_back),
        motion("MoveHome", GraspSkills.step_move_home, need_plan=False),
        motion("ReleaseClaw", GraspSkills.step_release_claw, need_plan=False),
    ])
    return seq


def build_grasp_pipeline(name="GraspPipeline"):
    pipeline = py_trees.composites.Sequence(name=name, memory=True)
    pipeline.add_children([
        ParseGrabFields(name="ParseTarget"),
        CheckYoloAlive(),
        CollectYoloTarget(name="CollectTF2_median"),
        PlanGraspTrajectory(name="PlanIK_ABDE"),
        build_motion_sequence(),
    ])
    return pipeline


def build_handle_grab():
    work = py_trees.composites.Sequence(name="GrabWork", memory=True)
    work.add_children([
        build_grasp_pipeline(),
        ClearCommand(),
    ])

    handle = py_trees.composites.Sequence(name="HandleGrab", memory=True)
    handle.add_children([
        HasActionCommand("grab"),
        EnsureNotBusy(),
        MarkBusy(),
        _wrap_work(work, "GrabBusyScope"),
    ])
    return handle


def build_handle_chat():
    work = py_trees.composites.Sequence(name="ChatWork", memory=True)
    work.add_children([
        ParseChatFields(),
        SpeakTTS(),
        ClearCommand(),
    ])

    handle = py_trees.composites.Sequence(name="HandleChat", memory=True)
    handle.add_children([
        HasActionCommand("chat"),
        EnsureNotBusy(),
        MarkBusy(),
        _wrap_work(work, "ChatBusyScope"),
    ])
    return handle


def build_vla_root_tree():
    """
    Selector [VLA_ROOT]
    ├── HandleGrab
    ├── HandleChat
    └── Idle
    """
    root = py_trees.composites.Selector(name="VLA_ROOT", memory=False)
    root.add_children([
        build_handle_grab(),
        build_handle_chat(),
        Idle(),
    ])
    return root
