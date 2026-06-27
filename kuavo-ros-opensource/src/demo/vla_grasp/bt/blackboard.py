#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLA 行为树黑板键与访问封装。"""

import py_trees


class BBKeys:
    COMMAND = "vla/command"
    WORKING = "vla/working"
    TARGET = "vla/target"
    SPEECH = "vla/speech"
    RAW_X = "vla/raw_x"
    RAW_Y = "vla/raw_y"
    PLAN = "grasp/plan"
    SKILLS = "grasp/skills"


def bb():
    return py_trees.blackboard.Blackboard()


def get(key, default=None):
    return bb().get(key) or default


def set_key(key, value):
    bb().set(key, value)


def init_blackboard(skills):
    set_key(BBKeys.COMMAND, None)
    set_key(BBKeys.WORKING, False)
    set_key(BBKeys.TARGET, "")
    set_key(BBKeys.SPEECH, "")
    set_key(BBKeys.RAW_X, None)
    set_key(BBKeys.RAW_Y, None)
    set_key(BBKeys.PLAN, None)
    set_key(BBKeys.SKILLS, skills)
