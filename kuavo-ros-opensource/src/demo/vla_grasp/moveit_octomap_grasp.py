#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoveIt + OctoMap 水瓶抓取（与 moveit_auto_grasp.py 并存，不覆盖旧版）

架构分工：
  - 抓取段（曲肘→预瞄→12cm 插入→夹爪）：auto_grasp_TF2 验证过的硬编码关节轨迹（不碰倒瓶）
  - 收手段：MoveIt OMPL + OctoMap 点云 + 桌面碰撞盒 避障规划；失败回退 vla 肩膀外摆

发车：终端 3 须用 move_group_octomap.launch（不是普通 move_group.launch）
  roslaunch kuavo_arm_moveit_config move_group_octomap.launch
"""

import os
import sys
import math
import time
import rospy
import moveit_commander
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from claw_safe import build_close_cmd, build_open_cmd

# 复用 moveit_auto_grasp 已验证的 IK / 执行 / 视觉逻辑
import moveit_auto_grasp as mag

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import MoveItErrorCodes

try:
    from octomap_msgs.msg import Octomap
    _HAS_OCTOMAP_MSG = True
except ImportError:
    _HAS_OCTOMAP_MSG = False

# =================================================================
# OctoMap 版专用参数
# =================================================================
# 常见点云话题（RealSense / 奥比中光 / 旧版驱动命名不一）
POINT_CLOUD_CANDIDATES = [
    "/camera/depth/color/points",   # RealSense align_depth + pointcloud
    "/camera/depth/points",
    "/camera/depth_registered/points",
    "/camera/depth/points2",
]
POINTCLOUD_PROBE_TIMEOUT = 3.0    # 每个候选话题探测秒数
OCTOMAP_PROBE_TIMEOUT = 2.5       # 启动时快速探测（勿阻塞 20s）
OCTOMAP_SETTLE_AFTER_LIFT = 2.0   # 抬升后等待点云刷新再规划收手
AVOID_OUTWARD_Y = 0.10            # 与 auto_grasp_TF2 高空外移量一致

# 桌面碰撞盒（点云补盲：桌面平面 / 书本，base_link 坐标）
DESK_BOX_ID = "vla_grasp_desk"
DESK_BOX_SIZE = (1.20, 0.90, 0.06)   # x, y, z (m)
DESK_BOX_CENTER = (0.42, 0.0, 0.33)  # 略低于 SAFE_LOCKED_Z


def _log_banner():
    print("========== 🗺️ MoveIt OctoMap 抓取（IK+OMPL 避障版）==========")
    print("  终端3: roslaunch kuavo_arm_moveit_config move_group_octomap.launch")
    print("  抓取段: auto_grasp 硬编码 | 收手: OctoMap+OMPL（失败→vla外摆）")
    print("==========================================================")


_resolved_point_cloud_topic = None


def _list_camera_pointcloud_topics():
    """列出当前 ROS 图里所有 PointCloud2 话题（便于排查）。"""
    found = []
    try:
        for name, msg_type in rospy.get_published_topics():
            if msg_type == "sensor_msgs/PointCloud2":
                found.append(name)
    except Exception:
        pass
    return sorted(found)


def resolve_pointcloud_topic():
    """
    自动探测点云话题。
    注意：/vla/yolo_target 是视觉检测坐标，与点云无关；YOLO 能刷屏不代表有点云。
    常见陷阱：rostopic list 能看到 /camera/depth/color/points，但无发布者（hz 为 no new messages）。
    """
    global _resolved_point_cloud_topic
    if _resolved_point_cloud_topic:
        return _resolved_point_cloud_topic

    rospy.loginfo("⏳ 探测点云话题（YOLO 坐标 ≠ 点云，两者独立）...")
    per_topic = max(POINTCLOUD_PROBE_TIMEOUT, 1.0)

    for topic in POINT_CLOUD_CANDIDATES:
        try:
            rospy.wait_for_message(topic, PointCloud2, timeout=per_topic)
            _resolved_point_cloud_topic = topic
            rospy.loginfo(f"✅ 点云已锁定: {topic}")
            return topic
        except rospy.ROSException:
            rospy.logwarn(f"⚠️ {per_topic:.0f}s 内无数据: {topic}")

    # 话题存在但无数据：深度图有、点云无 → 需 depth_to_pointcloud.launch
    depth_ok = False
    try:
        import sensor_msgs.msg
        rospy.wait_for_message("/camera/depth/image_raw", sensor_msgs.msg.Image, timeout=2.0)
        depth_ok = True
    except rospy.ROSException:
        pass

    if depth_ok:
        rospy.logerr(
            "❌ /camera/depth/image_raw 有数据，但点云话题无发布！\n"
            "   原因：相机 launch 未 enable_pointcloud，话题名存在但没人发。\n"
            "   在上位机另开终端执行：\n"
            "     roslaunch ~/kuavo-ros-opensource/src/demo/vla_grasp/launch/depth_to_pointcloud.launch\n"
            "   然后确认：rostopic hz /camera/depth/color/points"
        )

    # 扫描任意含 camera/depth 的 PointCloud2
    all_pc = _list_camera_pointcloud_topics()
    for topic in all_pc:
        if topic in POINT_CLOUD_CANDIDATES:
            continue
        if "camera" not in topic and "depth" not in topic:
            continue
        try:
            rospy.wait_for_message(topic, PointCloud2, timeout=per_topic)
            _resolved_point_cloud_topic = topic
            rospy.loginfo(f"✅ 点云已锁定（自动发现）: {topic}")
            rospy.logwarn(f"   请把 sensors_3d_octomap.yaml 的 point_cloud_topic 改为: {topic}")
            return topic
        except rospy.ROSException:
            continue

    rospy.logerr(
        "❌ 未收到任何相机点云！YOLO 坐标正常只说明 RGB+检测在跑，不代表深度/点云正常。\n"
        "   上位机相机日志若出现 Decode frame failed / MJPG conversion failed，\n"
        "   则深度流损坏，点云不会发布。\n"
        "   请在上位机执行排查：\n"
        "     rostopic list | grep -E 'points|PointCloud'\n"
        "     rostopic hz /camera/depth/color/points\n"
        "     rostopic hz /camera/depth/image_raw\n"
        "   RealSense 需在 launch 里 enable_pointcloud:=true align_depth:=true\n"
        "   奥比中光请确认 use_orbbec:=true 且驱动有点云输出。\n"
        f"   当前图中 PointCloud2 话题: {all_pc or '(无)'}"
    )
    return None


def wait_for_pointcloud():
    """兼容旧调用名。"""
    return resolve_pointcloud_topic() is not None


def probe_octomap_ready(timeout=OCTOMAP_PROBE_TIMEOUT):
    """
    启动时短时探测 OctoMap 管线（非阻塞）。
    MoveIt 常在 move_group 进程内建图，不一定对外发布 /octomap_full；
    收手 OMPL 成功即表示 OctoMap 已生效，此处勿长时间等待。
    """
    if not _HAS_OCTOMAP_MSG:
        rospy.loginfo("ℹ️ 未安装 octomap_msgs，跳过 OctoMap 启动探测")
        return False

    cloud_topics = [
        "/move_group/filtered_cloud",
        "/filtered_cloud",
    ]
    octomap_topics = ["/octomap_binary", "/octomap_full", "/move_group/octomap_full"]
    deadline = time.time() + timeout
    while time.time() < deadline and not rospy.is_shutdown():
        for topic in cloud_topics:
            try:
                rospy.wait_for_message(topic, PointCloud2, timeout=0.6)
                rospy.loginfo(f"✅ OctoMap 点云管线就绪: {topic}")
                return True
            except rospy.ROSException:
                continue
        for topic in octomap_topics:
            try:
                rospy.wait_for_message(topic, Octomap, timeout=0.4)
                rospy.loginfo(f"✅ OctoMap 已就绪: {topic}")
                return True
            except rospy.ROSException:
                continue
        rospy.sleep(0.15)

    rospy.loginfo(
        "ℹ️ 启动阶段未见 /octomap_full（常见且正常：MoveIt 在收手规划时内部建图）。"
        "  点云已有数据即可；收手将使用 OctoMap + 桌面碰撞盒。"
    )
    return False


def wait_for_octomap(timeout=OCTOMAP_PROBE_TIMEOUT):
    """兼容旧名；收手前可再探测一次。"""
    return probe_octomap_ready(timeout)


def setup_desk_collision_box(scene_iface):
    """向 PlanningScene 添加桌面长方体，弥补 URDF 无桌面的缺陷。"""
    box = CollisionObject()
    box.header.frame_id = "base_link"
    box.id = DESK_BOX_ID
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(DESK_BOX_SIZE)
    box.primitives.append(primitive)
    pose = Pose()
    pose.position.x = DESK_BOX_CENTER[0]
    pose.position.y = DESK_BOX_CENTER[1]
    pose.position.z = DESK_BOX_CENTER[2]
    pose.orientation.w = 1.0
    box.primitive_poses.append(pose)
    box.operation = CollisionObject.ADD
    scene_iface.add_object(box)
    rospy.sleep(0.5)
    rospy.loginfo(
        f"📦 已添加桌面碰撞盒 id={DESK_BOX_ID} "
        f"center=({DESK_BOX_CENTER[0]:.2f},{DESK_BOX_CENTER[1]:.2f},{DESK_BOX_CENTER[2]:.2f})"
    )


def remove_desk_collision_box(scene_iface):
    try:
        scene_iface.remove_world_object(DESK_BOX_ID)
    except Exception:
        pass


def _solve_ik_avoid_collisions(ik_client, arm_group, is_left_arm, pose_stamped, seed_14, step_name):
    """带碰撞检测的 IK（收手航点用）。"""
    group_name, ee_link = mag._ik_group_profile(is_left_arm)
    seed = mag._build_ik_seed_for_pose(is_left_arm, seed_14)
    ps = PoseStamped()
    ps.header.frame_id = pose_stamped.header.frame_id
    ps.header.stamp = rospy.Time(0)
    ps.pose = pose_stamped.pose

    req = GetPositionIKRequest()
    req.ik_request.group_name = group_name
    req.ik_request.ik_link_name = ee_link
    req.ik_request.pose_stamped = ps
    req.ik_request.robot_state = mag._build_robot_state_seed(seed)
    req.ik_request.avoid_collisions = True
    req.ik_request.timeout = rospy.Duration(0.8)
    try:
        resp = ik_client(req)
    except Exception as exc:
        rospy.logwarn(f"⚠️ {step_name} 碰撞IK异常: {exc}")
        return None
    if resp.error_code.val != MoveItErrorCodes.SUCCESS:
        return None
    return mag._apply_ik_solution_to_14d(seed, resp.solution.joint_state, is_left_arm)


def _build_retract_pose_stamped(locked_x, locked_y, locked_z, quat, is_left_arm, shoulder_x, shoulder_y, dist):
    """auto_grasp_TF2 步骤 E：抬升高度 + 向外 Y + 略后退。"""
    for fallback_dist in (0.15, 0.12, 0.09, 0.06, 0.0):
        ratio = (dist - fallback_dist) / dist if dist > 1e-6 else 1.0
        safe_x = shoulder_x + (locked_x - shoulder_x) * ratio
        safe_y = locked_y + AVOID_OUTWARD_Y if is_left_arm else locked_y - AVOID_OUTWARD_Y
        ps = mag._build_pose_stamped(safe_x, safe_y, locked_z + mag.LIFT_HEIGHT, quat)
        yield fallback_dist, ps


def plan_and_execute_collision_aware(arm_group, arm_pub, step_name, total_time,
                                     is_left_arm=None, use_joint_target=False):
    """
    OMPL 规划并下发；默认 **不清除** OctoMap，真正利用八叉树避障。
    """
    rospy.loginfo(f"\n🗺️ MoveIt 碰撞感知规划: [{step_name}]（保留 OctoMap）...")
    arm_group.set_start_state_to_current_state()
    arm_group.set_max_velocity_scaling_factor(mag.VELOCITY_SCALING)
    arm_group.set_max_acceleration_scaling_factor(mag.ACCELERATION_SCALING)
    arm_group.set_planning_time(10.0)

    if not use_joint_target:
        # pose target 应在调用前已 set_pose_target
        pass

    success, plan, _, error_code = arm_group.plan()
    if not success or len(plan.joint_trajectory.points) == 0:
        rospy.logwarn(f"⚠️ [{step_name}] OMPL 失败 (code={error_code.val})")
        return False

    rospy.loginfo(f"✅ [{step_name}] OMPL 成功，{len(plan.joint_trajectory.points)} 点")
    chain = []
    target_rad = np.copy(mag.last_commanded_joints_rad)
    for point in plan.joint_trajectory.points:
        for j, name in enumerate(plan.joint_trajectory.joint_names):
            if name in mag.joint_names_14:
                target_rad[mag.joint_names_14.index(name)] = point.positions[j]
        chain.append(np.copy(target_rad))

    ok = mag.publish_arm_trajectory_batch(arm_pub, chain, total_time, is_left_arm=is_left_arm)
    if ok:
        mag.wait_joints_settle()
    return ok


def octomap_retract_after_lift(arm_group, arm_pub, ik_client, is_left_arm,
                               locked_x, locked_y, locked_z, quat,
                               shoulder_x, shoulder_y, dist, side):
    """
    抬升后收手：优先 OMPL+OctoMap 到高空外移点，再 OMPL 到 ready；失败走 vla 硬编码。
    """
    rospy.loginfo(f"\n🛡️ [{side}手] OctoMap 收手：等待点云刷新 {OCTOMAP_SETTLE_AFTER_LIFT}s ...")
    rospy.sleep(OCTOMAP_SETTLE_AFTER_LIFT)

    seed = np.copy(mag.last_commanded_joints_rad)
    planned = False

    for fallback_dist, retract_ps in _build_retract_pose_stamped(
        locked_x, locked_y, locked_z, quat, is_left_arm, shoulder_x, shoulder_y, dist
    ):
        q_retract = _solve_ik_avoid_collisions(
            ik_client, arm_group, is_left_arm, retract_ps, seed,
            f"高空外移+后退{int(fallback_dist*100)}cm"
        )
        if q_retract is None:
            continue

        arm_group.set_start_state_to_current_state()
        arm_group.set_joint_value_target(
            [q_retract[mag.joint_names_14.index(n)] for n in arm_group.get_active_joints()]
        )
        if plan_and_execute_collision_aware(
            arm_group, arm_pub,
            f"OctoMap外移(退{int(fallback_dist*100)}cm)",
            3.0, is_left_arm=is_left_arm, use_joint_target=True
        ):
            planned = True
            rospy.loginfo(
                f"💡 OctoMap 收手：外移 {int(AVOID_OUTWARD_Y*100)}cm + "
                f"后退 {int(fallback_dist*100)}cm 规划成功"
            )
            break

    if planned:
        ready_name = "left_ready" if is_left_arm else "right_ready"
        arm_group.set_named_target(ready_name)
        if plan_and_execute_collision_aware(
            arm_group, arm_pub, "OctoMap曲肘ready", 3.5, is_left_arm=is_left_arm
        ):
            mag.execute_dual_arm_init_home(arm_pub)
            return True
        rospy.logwarn("⚠️ OctoMap ready 规划失败，继续 vla 回退")

    # vla 硬编码回退（与 vla_auto_grasp_daemon 一致）
    rospy.logwarn("⚠️ OctoMap 收手失败，回退 vla 肩膀外摆...")
    mag.execute_vla_style_return(arm_pub, mag.last_commanded_joints_rad, is_left_arm)
    return False


def _run_octomap_grasp_sequence(left_arm, right_arm, arm_pub, scene_iface, x_hist, y_hist):
    is_left_arm = np.median(y_hist) > 0.0
    arm = left_arm if is_left_arm else right_arm
    side = "左" if is_left_arm else "右"

    off_x, off_y = mag.tcp_offsets_for_arm(is_left_arm)
    locked_x = np.median(x_hist) + off_x
    locked_y = np.median(y_hist) + off_y
    locked_z = mag.SAFE_LOCKED_Z

    print(f"\n🎯 TF2 融合打击点: X={locked_x:.3f}, Y={locked_y:.3f}, Z={locked_z:.3f} ({side}手)")

    quat = mag.get_horizontal_claw_quat(locked_x, locked_y, is_left_arm)
    shoulder_x = -0.017
    shoulder_y = 0.292 if is_left_arm else -0.292
    dist = math.hypot(locked_x - shoulder_x, locked_y - shoulder_y)
    if dist <= mag.PRE_GRASP_DIST + 0.01:
        rospy.logerr(f"❌ 目标距肩点仅 {dist*100:.1f}cm，过近")
        mag.execute_dual_arm_init_home(arm_pub)
        return

    ratio = (dist - mag.PRE_GRASP_DIST) / dist
    pre_x = shoulder_x + (locked_x - shoulder_x) * ratio
    pre_y = shoulder_y + (locked_y - shoulder_y) * ratio
    grasp_pose = mag._build_pose_stamped(locked_x, locked_y, locked_z, quat)
    pre_pose = mag._build_pose_stamped(pre_x, pre_y, locked_z, quat)
    lift_pose = mag._build_pose_stamped(locked_x, locked_y, locked_z + mag.LIFT_HEIGHT, quat)

    _, ee_link = mag._ik_group_profile(is_left_arm)
    try:
        arm.set_end_effector_link(ee_link)
    except Exception:
        pass

    ik_client = mag._resolve_ik_service()
    if ik_client is None:
        mag.execute_dual_arm_init_home(arm_pub)
        return

    q_grasp = mag._solve_pose_ik(
        ik_client, arm, is_left_arm, grasp_pose, mag.last_commanded_joints_rad, f"[{side}手] 终点切入"
    )
    if q_grasp is None:
        mag.execute_dual_arm_init_home(arm_pub)
        return

    q_pre = mag._solve_pose_ik(ik_client, arm, is_left_arm, pre_pose, q_grasp, f"[{side}手] 预瞄12cm")
    if q_pre is None:
        q_pre = q_grasp

    q_lift = mag._solve_pose_ik(ik_client, arm, is_left_arm, lift_pose, q_grasp, f"[{side}手] 抬升")
    if q_lift is None:
        q_lift = q_grasp

    print("\n🚀 阶段A [硬编码抓取] 曲肘→预瞄→水平插入（auto_grasp_TF2 同款，防碰瓶）")
    print("🚀 阶段B [OctoMap收手]  抬升后 OMPL 碰撞规划（失败→vla外摆）")

    ready_rad = np.radians(mag._auto_grasp_ready_deg(is_left_arm))
    mag.execute_single_pose(arm_pub, ready_rad, 2.5, "曲肘护胸", is_left_arm)

    q_pre_exec = mag._solve_pose_ik(
        ik_client, arm, is_left_arm, pre_pose, mag.last_commanded_joints_rad, f"[{side}手] 曲肘后预瞄"
    )
    if q_pre_exec is None:
        q_pre_exec = q_pre
    mag.execute_single_pose(arm_pub, q_pre_exec, 2.5, "退至预瞄点 12cm", is_left_arm)

    # 水瓶在点云里：插入前短暂清 OctoMap，仅影响最后 12cm 段
    mag.clear_octomap_cache()
    mag.execute_single_pose(arm_pub, q_grasp, 1.5, "水平笔直插入", is_left_arm)

    print("✊ 夹爪闭合...")
    pos, vel, effort = build_close_cmd(is_left_arm)
    mag.call_leju_claw(pos, vel, effort, tag="close")
    time.sleep(2.0)

    mag.execute_single_pose(arm_pub, q_lift, 2.0, f"垂直抬升 {int(mag.LIFT_HEIGHT*100)}cm", is_left_arm)

    print("\n🗺️ 阶段B：OctoMap + 桌面盒 碰撞感知收手...")
    octomap_retract_after_lift(
        arm, arm_pub, ik_client, is_left_arm,
        locked_x, locked_y, locked_z, quat,
        shoulder_x, shoulder_y, dist, side
    )

    print("👐 松开夹爪")
    pos, vel, effort = build_open_cmd()
    mag.call_leju_claw(pos, vel, effort, tag="release")
    try:
        rospy.ServiceProxy('/arm_traj_change_mode', mag.changeArmCtrlMode)(
            mag.changeArmCtrlModeRequest(control_mode=0)
        )
    except Exception:
        pass


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_octomap_grasp")
    _log_banner()

    rospy.Subscriber("/joint_states", mag.JointState, mag.joint_states_callback)
    rospy.loginfo("⏳ 等待 /joint_states ...")
    while not mag.has_joint_states and not rospy.is_shutdown():
        rospy.sleep(0.1)
    mag.last_commanded_joints_rad = np.copy(mag.current_joints_rad)

    mag._log_ros_network()

    pc_topic = resolve_pointcloud_topic()
    if pc_topic is None:
        rospy.logwarn(
            "⚠️ 无点云：进入降级模式（仅桌面碰撞盒 + vla 收手），"
            "OctoMap 避障不可用。请修复上位机深度流后重试。"
        )
    else:
        rospy.set_param("/move_group/octomap_frame", "base_link")
        # 若 move_group 已启动，尝试同步传感器话题（需与 launch 里 yaml 一致）
        rospy.loginfo(
            f"💡 请确认 move_group_octomap.launch 中 sensors_3d_octomap.yaml "
            f"的 point_cloud_topic = {pc_topic}"
        )

    try:
        rospy.ServiceProxy("/arm_traj_change_mode", mag.changeArmCtrlMode)(
            mag.changeArmCtrlModeRequest(control_mode=2)
        )
    except Exception:
        rospy.logwarn("⚠️ arm_traj_change_mode 失败，继续...")

    arm_pub = rospy.Publisher("/kuavo_arm_target_poses", mag.armTargetPoses, queue_size=10)
    rospy.sleep(0.3)

    print("🖐️ 双爪张开...")
    pos, vel, effort = build_open_cmd()
    mag.call_leju_claw(pos, vel, effort, tag="open")
    time.sleep(1.0)
    mag.execute_dual_arm_init_home(arm_pub)

    x_hist, y_hist = mag._collect_vision_targets_tf2_style()
    if len(x_hist) < 10:
        rospy.logerr("❌ 视觉失败")
        mag.execute_dual_arm_init_home(arm_pub)
        return

    rospy.loginfo("✅ 视觉就绪，加载 MoveIt（约 2～3s，属正常）...")
    left_arm = moveit_commander.MoveGroupCommander("left_arm")
    right_arm = moveit_commander.MoveGroupCommander("right_arm")
    for grp in (left_arm, right_arm):
        grp.set_pose_reference_frame("base_link")

    if pc_topic is not None:
        probe_octomap_ready()
    else:
        rospy.logwarn("⚠️ 跳过 OctoMap 启动探测（无点云）")

    scene_iface = moveit_commander.PlanningSceneInterface()
    setup_desk_collision_box(scene_iface)

    try:
        _run_octomap_grasp_sequence(left_arm, right_arm, arm_pub, scene_iface, x_hist, y_hist)
    except Exception as exc:
        rospy.logerr(f"❌ 异常: {exc}")
        mag.execute_dual_arm_init_home(arm_pub)
        raise
    finally:
        remove_desk_collision_box(scene_iface)


if __name__ == "__main__":
    main()
