#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BT 可调参数（由 daemon 在 rospy.init 后加载）。"""

import rospy

YOLO_TOPIC = "/vla/yolo_target"
YOLO_X_MIN = 0.30
YOLO_X_MAX = 0.65
YOLO_SAMPLE_COUNT = 10
YOLO_COLLECT_TIMEOUT_SEC = 12.0
TTS_URL = "http://127.0.0.1:5000/tts"
TTS_TIMEOUT_SEC = 2.0
BT_TICK_HZ = 10.0
LOG_STATUS_CHANGE = True
GRAB_CMD_MIN_INTERVAL_SEC = 8.0

# 抓取 TCP 补偿（TF2：y>0 左臂，y<0 右臂；减小左 y 往中心线修）
TCP_OFFSET_X = 0.005
TCP_OFFSET_Y_LEFT = -0.015
TCP_OFFSET_Y_RIGHT = 0.055


def load_from_ros_params():
    global YOLO_TOPIC, YOLO_X_MIN, YOLO_X_MAX, YOLO_SAMPLE_COUNT
    global YOLO_COLLECT_TIMEOUT_SEC, TTS_URL, TTS_TIMEOUT_SEC, BT_TICK_HZ, LOG_STATUS_CHANGE
    global GRAB_CMD_MIN_INTERVAL_SEC, TCP_OFFSET_X, TCP_OFFSET_Y_LEFT, TCP_OFFSET_Y_RIGHT

    YOLO_TOPIC = rospy.get_param("~yolo_topic", YOLO_TOPIC)
    YOLO_X_MIN = rospy.get_param("~yolo_x_min", YOLO_X_MIN)
    YOLO_X_MAX = rospy.get_param("~yolo_x_max", YOLO_X_MAX)
    YOLO_SAMPLE_COUNT = rospy.get_param("~yolo_sample_count", YOLO_SAMPLE_COUNT)
    YOLO_COLLECT_TIMEOUT_SEC = rospy.get_param("~yolo_collect_timeout", YOLO_COLLECT_TIMEOUT_SEC)
    TTS_URL = rospy.get_param("~tts_url", TTS_URL)
    TTS_TIMEOUT_SEC = rospy.get_param("~tts_timeout", TTS_TIMEOUT_SEC)
    BT_TICK_HZ = rospy.get_param("~tick_hz", BT_TICK_HZ)
    LOG_STATUS_CHANGE = rospy.get_param("~log_status_change", LOG_STATUS_CHANGE)
    GRAB_CMD_MIN_INTERVAL_SEC = rospy.get_param("~grab_cmd_min_interval", GRAB_CMD_MIN_INTERVAL_SEC)
    TCP_OFFSET_X = rospy.get_param("~tcp_offset_x", TCP_OFFSET_X)
    TCP_OFFSET_Y_LEFT = rospy.get_param("~tcp_offset_y_left", TCP_OFFSET_Y_LEFT)
    TCP_OFFSET_Y_RIGHT = rospy.get_param("~tcp_offset_y_right", TCP_OFFSET_Y_RIGHT)
