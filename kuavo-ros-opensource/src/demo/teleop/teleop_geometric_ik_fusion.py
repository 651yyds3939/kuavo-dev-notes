#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 遥操 + 几何关节 + 可选 IK 腕部微调。

几何映射与 teleop_math_solver.py 完全一致（主控制路径）。
IK 不再使用 MediaPipe 绝对坐标（与机器人 base_link 不对齐，会把手拉到胸前）；
仅在 ~use_ik:=true 时，用 FK(几何关节) 生成一致目标，并只微调腕部关节。
"""

import rospy
import cv2
import numpy as np
import math
import mediapipe as mp
import threading
import time
from sensor_msgs.msg import JointState
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from kuavo_msgs.srv import twoArmHandPoseCmdSrv, fkSrv
from kuavo_msgs.msg import twoArmHandPoseCmd

LIMITS = {
    "l_pitch": (-np.pi, np.pi / 2),
    "l_roll": (-0.349, 2.094),
    "l_yaw": (-np.pi / 2, np.pi / 2),
    "l_elbow": (-2.618, 0.0),
    "l_wrist_yaw": (-np.pi / 2, np.pi / 2),
    "l_wrist_roll": (-1.309, 0.698),
    "r_pitch": (-np.pi, np.pi / 2),
    "r_roll": (-2.094, 0.349),
    "r_yaw": (-np.pi / 2, np.pi / 2),
    "r_elbow": (-2.618, 0.0),
    "r_wrist_yaw": (-np.pi / 2, np.pi / 2),
    "r_wrist_roll": (-0.698, 1.309),
}

WRIST_JOINT_IDX = [5, 6, 12, 13]
CORE_JOINT_IDX = [0, 1, 2, 3, 7, 8, 9, 10]


def calculate_vector_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = dot_product / (norm_v1 * norm_v2)
    return math.acos(np.clip(cos_angle, -1.0, 1.0))


def compute_elbow_bend(v_upper, v_fore, elbow_gain=1.2):
    ang = calculate_vector_angle(v_upper, v_fore)
    raw = -ang
    if ang > math.pi / 3:
        raw = -(math.pi / 3 + (ang - math.pi / 3) * elbow_gain)
    return raw


def detect_chest_cross(v_upper, v_fore, already_active=False):
    upper_len = np.linalg.norm(v_upper)
    fore_len = np.linalg.norm(v_fore)
    if upper_len < 1e-4 or fore_len < 1e-4:
        return False

    fx, fy, fz = v_fore[0], v_fore[1], v_fore[2]
    fore_horiz_ratio = math.hypot(fx, fz) / fore_len
    fore_y_ratio = abs(fy) / fore_len
    elbow_ang = calculate_vector_angle(v_upper, v_fore)

    x, y, z = v_upper[0], v_upper[1], v_upper[2]
    if elbow_ang < 0.45 and abs(x) / upper_len > 0.55:
        return False

    if already_active:
        return fore_horiz_ratio > 0.55 and fore_y_ratio < 0.58 and elbow_ang > math.pi / 5.5
    return fore_horiz_ratio > 0.68 and fore_y_ratio < 0.48 and elbow_ang > math.pi / 4.0


def is_arm_hanging(v_upper, v_fore):
    upper_len = np.linalg.norm(v_upper)
    if upper_len < 1e-4:
        return False
    down_ratio = v_upper[1] / upper_len
    elbow_ang = calculate_vector_angle(v_upper, v_fore)
    return down_ratio > 0.55 and elbow_ang < 0.55


def compute_wrist_normal(v_upper, v_fore, is_left):
    if not is_arm_hanging(v_upper, v_fore):
        return 0.0, 0.0
    nf = np.linalg.norm(v_fore)
    if nf < 1e-4:
        return 0.0, 0.0
    f = v_fore / nf
    wrist_roll = 0.35 * math.atan2(f[0], abs(f[1]) + 1e-6)
    if not is_left:
        wrist_roll = -wrist_roll
    return 0.0, wrist_roll


def update_chest_state(detected, active, enter_count, exit_count, enter_need=6, exit_need=12):
    if detected:
        enter_count += 1
        exit_count = 0
        if enter_count >= enter_need:
            active = True
    else:
        enter_count = 0
        exit_count += 1
        if exit_count >= exit_need:
            active = False
    return active, enter_count, exit_count


def solve_arm(v_upper, v_fore, last_pitch, limits_roll, limits_yaw, is_left, elbow_gain, chest_active):
    x, y, z = v_upper[0], v_upper[1], v_upper[2]
    yz_norm = math.hypot(y, z)
    fx, fy, fz = v_fore[0], v_fore[1], v_fore[2]
    wrist_yaw, wrist_roll = 0.0, 0.0

    if chest_active:
        pitch = math.atan2(z, math.hypot(x, y) + 1e-6)
        pitch = float(np.clip(pitch, -0.45, 0.55))
        roll = math.atan2(x, yz_norm + 1e-6)
        upper_az = math.atan2(x, z + 1e-6)
        fore_az = math.atan2(fx, fz + 1e-6)
        yaw = float(np.clip(fore_az - upper_az, -1.0, 1.0))
        wrist_yaw = math.atan2(fx, math.hypot(fy, fz) + 1e-6)
        if is_left:
            wrist_yaw = -wrist_yaw
        elbow = compute_elbow_bend(v_upper, v_fore, elbow_gain=1.0)
    else:
        if yz_norm < 0.08:
            raw_pitch = last_pitch
        else:
            raw_pitch = math.atan2(z, y)
            last_pitch = raw_pitch
        pitch = raw_pitch
        roll = math.atan2(x, yz_norm)
        yaw = 0.0
        elbow = compute_elbow_bend(v_upper, v_fore, elbow_gain)
        wrist_yaw, wrist_roll = compute_wrist_normal(v_upper, v_fore, is_left)

    roll = float(np.clip(roll, *limits_roll))
    yaw = float(np.clip(yaw, *limits_yaw))
    wrist_yaw = float(np.clip(wrist_yaw, -np.pi / 2, np.pi / 2))
    return pitch, roll, yaw, elbow, wrist_yaw, wrist_roll, last_pitch


def build_arm_seed(pitch, roll, yaw, elbow, wrist_yaw, wrist_roll, limits_prefix):
    return [
        float(np.clip(pitch, *LIMITS[f"{limits_prefix}_pitch"])),
        float(np.clip(roll, *LIMITS[f"{limits_prefix}_roll"])),
        float(np.clip(yaw, *LIMITS[f"{limits_prefix}_yaw"])),
        float(np.clip(elbow, *LIMITS[f"{limits_prefix}_elbow"])),
        float(np.clip(wrist_yaw, *LIMITS[f"{limits_prefix}_wrist_yaw"])),
        float(np.clip(wrist_roll, *LIMITS[f"{limits_prefix}_wrist_roll"])),
        0.0,
    ]


def seeds_to_q_rad(seed_l, seed_r):
    return list(seed_l) + list(seed_r)


def geometric_q_arm_deg(l_seed, r_seed):
    q = np.zeros(14)
    q[0:6] = np.array(l_seed[:6]) * (180.0 / np.pi)
    q[7:13] = np.array(r_seed[:6]) * (180.0 / np.pi)
    return q


class ThreadedVideoCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            ret, frame = self.cap.read()
            if ret:
                with self.read_lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.read_lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.started = False
        self.cap.release()


class KuavoTeleopIkFusionSolver:
    def __init__(self):
        rospy.init_node('kuavo_teleop_ik_fusion_node')

        self.traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=1)
        self.joint_names = ["arm_joint_" + str(i) for i in range(1, 15)]

        self.filter_alpha = rospy.get_param("~filter_alpha", 0.35)
        self.elbow_gain = rospy.get_param("~elbow_gain", 1.2)
        self.swap_arms = rospy.get_param("~swap_arms", False)
        # 默认关闭 IK：几何解已足够准；开启时仅做腕部微调
        self.use_ik = rospy.get_param("~use_ik", False)
        self.ik_wrist_only = rospy.get_param("~ik_wrist_only", True)
        self.ik_interval = max(1, int(rospy.get_param("~ik_interval", 3)))
        self.fk_dof = int(rospy.get_param("~fk_dof", 14))

        ik_service = rospy.get_param(
            "~ik_service", "/ik/two_arm_hand_pose_cmd_srv_muli_refer"
        )
        if self.use_ik:
            rospy.loginfo("⏳ 连接 FK/IK 服务 (腕部微调模式)...")
            rospy.wait_for_service("/ik/fk_srv", timeout=5.0)
            rospy.wait_for_service(ik_service, timeout=5.0)
            self.fk_client = rospy.ServiceProxy("/ik/fk_srv", fkSrv)
            self.ik_client = rospy.ServiceProxy(ik_service, twoArmHandPoseCmdSrv)
            rospy.loginfo("✅ FK/IK 已连接 (~use_ik:=true, ~ik_wrist_only:=%s)", self.ik_wrist_only)
        else:
            rospy.loginfo("✅ 纯几何模式 (与 teleop_math_solver 一致, ~use_ik:=false)")

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.last_raw_l_pitch = 0.0
        self.last_raw_r_pitch = 0.0
        self.filtered_positions = np.zeros(14)
        self.frame_idx = 0

        self.l_chest_active = False
        self.r_chest_active = False
        self.l_chest_enter_cnt = 0
        self.r_chest_enter_cnt = 0
        self.l_chest_exit_cnt = 0
        self.r_chest_exit_cnt = 0

        self.activate_robot_control()

    def activate_robot_control(self):
        try:
            rospy.wait_for_service('/arm_traj_change_mode', timeout=5.0)
            mode_srv = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            mode_srv(changeArmCtrlModeRequest(control_mode=2))
            rospy.loginfo("✅ 夸父手臂接管模式启动")
        except Exception as e:
            rospy.logerr(f"无法接管手臂: {e}")

    def _call_fk(self, q_rad):
        q = list(q_rad)
        if len(q) < self.fk_dof:
            q = q + [0.0] * (self.fk_dof - len(q))
        elif len(q) > self.fk_dof:
            q = q[:self.fk_dof]
        res = self.fk_client(q)
        if not res.success:
            return None
        return res.hand_poses

    def _build_ik_request(self, fk_poses, seed_l, seed_r):
        req = twoArmHandPoseCmd()
        req.use_custom_ik_param = True
        req.joint_angles_as_q0 = True

        # pos_cost_weight=0：以 seed 为主，FK 目标仅作一致性约束
        req.ik_param.constraint_mode = 1
        req.ik_param.pos_cost_weight = 0.0
        req.ik_param.pos_constraint_tol = 0.01
        req.ik_param.oritation_constraint_tol = 0.3
        req.ik_param.major_optimality_tol = 1e-4
        req.ik_param.major_feasibility_tol = 1e-4
        req.ik_param.minor_feasibility_tol = 1e-4
        req.ik_param.major_iterations_limit = 200

        lp = fk_poses.left_pose
        rp = fk_poses.right_pose
        req.hand_poses.left_pose.pos_xyz = list(lp.pos_xyz)
        req.hand_poses.left_pose.quat_xyzw = list(lp.quat_xyzw)
        req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.left_pose.joint_angles = seed_l

        req.hand_poses.right_pose.pos_xyz = list(rp.pos_xyz)
        req.hand_poses.right_pose.quat_xyzw = list(rp.quat_xyzw)
        req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.right_pose.joint_angles = seed_r
        return req

    def _merge_ik_into_geo(self, geo_deg, ik_deg):
        """几何解保护肩/肘；IK 只许在合理范围内改腕部。"""
        out = geo_deg.copy()
        if self.ik_wrist_only:
            for idx in WRIST_JOINT_IDX:
                if abs(ik_deg[idx] - geo_deg[idx]) < 40.0:
                    out[idx] = ik_deg[idx]
            return out

        core_dev = max(abs(ik_deg[i] - geo_deg[i]) for i in CORE_JOINT_IDX)
        if core_dev < 20.0:
            return ik_deg.copy()
        return out

    def _filter_positions(self, raw_positions):
        controlled_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        for idx in controlled_idx:
            chest_fast = (
                (self.l_chest_active and idx in (0, 1, 2, 4))
                or (self.r_chest_active and idx in (7, 8, 9, 11))
            )
            alpha = 0.55 if chest_fast else self.filter_alpha
            self.filtered_positions[idx] = (
                alpha * raw_positions[idx] + (1.0 - alpha) * self.filtered_positions[idx]
            )
        return self.filtered_positions.copy()

    def run_pipeline(self, url):
        print("🚀 正在拉起后台双线程极速视频流捕获管线...")
        threaded_cap = ThreadedVideoCapture(url).start()
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            ret, frame = threaded_cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)

            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark

                if self.swap_arms:
                    l_idx, r_idx = (12, 14, 16), (11, 13, 15)
                else:
                    l_idx, r_idx = (11, 13, 15), (12, 14, 16)

                p_ls = np.array([landmarks[l_idx[0]].x, landmarks[l_idx[0]].y, landmarks[l_idx[0]].z])
                p_le = np.array([landmarks[l_idx[1]].x, landmarks[l_idx[1]].y, landmarks[l_idx[1]].z])
                p_lw = np.array([landmarks[l_idx[2]].x, landmarks[l_idx[2]].y, landmarks[l_idx[2]].z])
                p_rs = np.array([landmarks[r_idx[0]].x, landmarks[r_idx[0]].y, landmarks[r_idx[0]].z])
                p_re = np.array([landmarks[r_idx[1]].x, landmarks[r_idx[1]].y, landmarks[r_idx[1]].z])
                p_rw = np.array([landmarks[r_idx[2]].x, landmarks[r_idx[2]].y, landmarks[r_idx[2]].z])

                v_lu = p_le - p_ls
                v_ru = p_re - p_rs
                v_lf = p_lw - p_le
                v_rf = p_rw - p_re

                l_detected = detect_chest_cross(v_lu, v_lf, self.l_chest_active)
                r_detected = detect_chest_cross(v_ru, v_rf, self.r_chest_active)
                (self.l_chest_active, self.l_chest_enter_cnt,
                 self.l_chest_exit_cnt) = update_chest_state(
                    l_detected, self.l_chest_active, self.l_chest_enter_cnt, self.l_chest_exit_cnt)
                (self.r_chest_active, self.r_chest_enter_cnt,
                 self.r_chest_exit_cnt) = update_chest_state(
                    r_detected, self.r_chest_active, self.r_chest_enter_cnt, self.r_chest_exit_cnt)

                (raw_l_pitch, raw_l_roll, raw_l_yaw, raw_l_elbow, raw_l_wy, raw_l_wr,
                 self.last_raw_l_pitch) = solve_arm(
                    v_lu, v_lf, self.last_raw_l_pitch, LIMITS["l_roll"], LIMITS["l_yaw"],
                    is_left=True, elbow_gain=self.elbow_gain, chest_active=self.l_chest_active)
                (raw_r_pitch, raw_r_roll, raw_r_yaw, raw_r_elbow, raw_r_wy, raw_r_wr,
                 self.last_raw_r_pitch) = solve_arm(
                    v_ru, v_rf, self.last_raw_r_pitch, LIMITS["r_roll"], LIMITS["r_yaw"],
                    is_left=False, elbow_gain=self.elbow_gain, chest_active=self.r_chest_active)

                seed_l = build_arm_seed(
                    raw_l_pitch, raw_l_roll, raw_l_yaw, raw_l_elbow, raw_l_wy, raw_l_wr, "l")
                seed_r = build_arm_seed(
                    raw_r_pitch, raw_r_roll, raw_r_yaw, raw_r_elbow, raw_r_wy, raw_r_wr, "r")
                geo_q_deg = geometric_q_arm_deg(seed_l, seed_r)
                raw_positions = geo_q_deg.copy()

                mode_tag = "GEO"
                self.frame_idx += 1

                if self.use_ik and (self.frame_idx % self.ik_interval == 0):
                    try:
                        fk_poses = self._call_fk(seeds_to_q_rad(seed_l, seed_r))
                        if fk_poses is not None:
                            ik_req = self._build_ik_request(fk_poses, seed_l, seed_r)
                            ik_res = self.ik_client(ik_req)
                            if ik_res.success and len(ik_res.q_arm) >= 12:
                                ik_deg = np.array(ik_res.q_arm[:14]) * (180.0 / np.pi)
                                raw_positions = self._merge_ik_into_geo(geo_q_deg, ik_deg)
                                mode_tag = "GEO+WRIST" if self.ik_wrist_only else "GEO+IK"
                    except Exception as e:
                        rospy.logwarn_throttle(2.0, "IK 腕部微调失败，保持几何解: %s", e)

                filtered = self._filter_positions(raw_positions)

                msg = JointState()
                msg.header.stamp = rospy.Time.now()
                msg.name = self.joint_names
                msg.position = filtered.tolist()
                self.traj_pub.publish(msg)

                chest_tag = (" [左胸]" if self.l_chest_active else "") + (" [右胸]" if self.r_chest_active else "")
                print(
                    f"📊 [{mode_tag}] 左 p:{filtered[0]:.0f} r:{filtered[1]:.0f} "
                    f"y:{filtered[2]:.0f} e:{filtered[3]:.0f} | "
                    f"右 p:{filtered[7]:.0f} r:{filtered[8]:.0f} "
                    f"y:{filtered[9]:.0f} e:{filtered[10]:.0f}"
                    f"{chest_tag}",
                    end="\r",
                )

                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Teleop IK Fusion Solver', cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rate.sleep()

        threaded_cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        solver = KuavoTeleopIkFusionSolver()
        solver.run_pipeline("http://10.220.197.154:8080/video")
    except rospy.ROSInterruptException:
        pass
