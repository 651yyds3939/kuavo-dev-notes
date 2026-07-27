#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge, CvBridgeError

# 🔥 引入 TF2 空间树
import tf2_ros
import tf2_geometry_msgs

import torch
import torchvision
from ultralytics import YOLO

def pure_torch_nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0: return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1: break
        xx1, yy1 = torch.max(x1[i], x1[order[1:]]), torch.max(y1[i], y1[order[1:]])
        xx2, yy2 = torch.min(x2[i], x2[order[1:]]), torch.min(y2[i], y2[order[1:]])
        w, h = torch.clamp(xx2 - xx1, min=0.0), torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = torch.where(ovr <= iou_threshold)[0]
        order = order[ids + 1]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

torchvision.ops.nms = pure_torch_nms

# ============================================================
# 🔵 蓝色瓶盖检测开关 (默认关闭，通过 ROS 参数 ~enable_blue_cap_detection 打开)
# ============================================================
ENABLE_BLUE_CAP_DETECTION = False

# 蓝色 HSV 阈值 (百事可乐瓶盖深蓝色)
BLUE_LOWER = np.array([100, 50, 50])   # H:100-130, S:50-255, V:50-255
BLUE_UPPER = np.array([130, 255, 255])
BLUE_MASK_MIN_AREA = 100               # mask 最小面积 (px)，低于此值不发
BLUE_MAX_DEPTH_M = 1.5                # 最大深度 (米)，过滤远处地面等干扰
CAMERA_UPSIDE_DOWN = True              # 相机倒装：设为 True 自动翻转图像

color_image, depth_image, camera_info = None, None, None
frame_lock = threading.Lock()
bridge = CvBridge()
vla_pub, cap_pub, tf_buffer = None, None, None

def image_callback(msg):
    global color_image
    try: color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except: pass

def depth_callback(msg):
    global depth_image
    try: depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")
    except: pass

def camera_info_callback(msg):
    global camera_info
    camera_info = msg

def convert_to_3d(u, v, depth_image, camera_info, box, region_factor=0.5):
    fx, fy, cx, cy = camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5]
    bw, bh = box[2] - box[0], box[3] - box[1]
    rw, rh = int(bw * region_factor), int(bh * region_factor)
    u_min, u_max = max(0, u - rw // 2), min(depth_image.shape[1], u + rw // 2)
    v_min, v_max = max(0, v - rh // 2), min(depth_image.shape[0], v + rh // 2)
    
    depth_region = depth_image[v_min:v_max, u_min:u_max]
    depth_values = depth_region[depth_region > 0]  
    if len(depth_values) == 0: return None  

    # 提取相机坐标系下的原始 3D 点
    z = np.median(depth_values) / 1000.0  
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def detect_blue_cap(input_image, depth_image, camera_info, roi=None):
    """
    🔵 HSV 颜色过滤检测蓝色瓶盖
    roi: (x1, y1, x2, y2) 可选，YOLO 瓶身框的上半部分，限定搜索范围
    返回 (PointStamped, cx, cy, contour) 或 None
    """
    global tf_buffer

    h_img, w_img = input_image.shape[:2]

    # 1. 限定搜索区域
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        rx1 = max(0, int(rx1)); ry1 = max(0, int(ry1))
        rx2 = min(w_img, int(rx2)); ry2 = min(h_img, int(ry2))
        if rx2 <= rx1 or ry2 <= ry1:
            roi = None  # 退化：全图搜索
        else:
            search_img = input_image[ry1:ry2, rx1:rx2]
            search_depth = depth_image[ry1:ry2, rx1:rx2] if depth_image is not None else None

    if roi is None:
        rx1, ry1 = 0, 0
        search_img = input_image
        search_depth = depth_image

    # 2. BGR → HSV → 蓝色 mask
    hsv = cv2.cvtColor(search_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    # 形态学开闭运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 面积过滤 (ROI 模式放宽阈值)
    min_area = BLUE_MASK_MIN_AREA if roi is None else max(10, BLUE_MASK_MIN_AREA // 5)
    if cv2.countNonZero(mask) < min_area:
        return None

    # 3. 找最大轮廓的质心（相对于搜索区域，需加回偏移）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] <= 0:
        return None

    cx = int(M["m10"] / M["m00"]) + rx1
    cy = int(M["m01"] / M["m00"]) + ry1

    # 轮廓偏移回原图坐标
    contour_offset = largest_contour + [rx1, ry1]

    # 4. 深度采样：以质心为中心取小窗口做中值滤波
    if depth_image is None or camera_info is None:
        return None

    fx = camera_info.K[0]
    fy = camera_info.K[4]
    ppx = camera_info.K[2]
    ppy = camera_info.K[5]

    half_win = 5
    u_min = max(0, cx - half_win)
    u_max = min(w_img, cx + half_win)
    v_min = max(0, cy - half_win)
    v_max = min(h_img, cy + half_win)

    depth_region = depth_image[v_min:v_max, u_min:u_max]
    depth_values = depth_region[depth_region > 0]
    if len(depth_values) == 0:
        return None

    z = np.median(depth_values) / 1000.0  # mm → m

    # 深度过滤：跳过远处地面等干扰
    if z > BLUE_MAX_DEPTH_M:
        return None

    x = (cx - ppx) * z / fx
    y = (cy - ppy) * z / fy

    # 5. TF 变换 camera → base_link
    try:
        point_in_camera = PointStamped()
        point_in_camera.header.frame_id = camera_info.header.frame_id
        point_in_camera.header.stamp = rospy.Time(0)
        point_in_camera.point.x = x
        point_in_camera.point.y = y
        point_in_camera.point.z = z

        transform = tf_buffer.lookup_transform(
            "base_link",
            camera_info.header.frame_id,
            rospy.Time(0),
            rospy.Duration(0.1)
        )
        point_in_base = tf2_geometry_msgs.do_transform_point(point_in_camera, transform)

        cap_msg = PointStamped()
        cap_msg.header.stamp = rospy.Time.now()
        cap_msg.header.frame_id = "base_link"
        cap_msg.point = point_in_base.point
        return (cap_msg, cx, cy, contour_offset)

    except Exception as e:
        rospy.logwarn_throttle(5.0, f"🔵 瓶盖 TF 变换失败: {e}")
        return None

def process_frame(model, input_image, depth_image, camera_info):
    global vla_pub, tf_buffer
    start_time = time.time()
    results = model(input_image, imgsz=640, verbose=False)

    boxes, scores, class_ids = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy(), results[0].boxes.cls.cpu().numpy().astype(int)
    combined_img = input_image.copy()
    detection_msg = Detection2DArray()
    best_vla_msg, best_score = None, 0.0
    best_bottle_box, best_bottle_box_score = None, 0.0

    for box, score, class_id in zip(boxes, scores, class_ids):
        if model.names[int(class_id)] != 'bottle' or score < 0.15: continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 追踪最高分的瓶子框（供 HSV ROI 使用）
        if best_bottle_box is None or score > best_bottle_box_score:
            best_bottle_box_score = score
            best_bottle_box = (x1, y1, x2, y2)

        if depth_image is not None and score > best_score:
            u, v = int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0)
            res_3d = convert_to_3d(u, v, depth_image, camera_info, box)
            
            if res_3d is not None:
                cam_x, cam_y, cam_z = res_3d
                
                # 🔥 TF2 动态抗晃动解算：无论头怎么扭，算出相对于骨盆的绝对坐标
                try:
                    point_in_camera = PointStamped()
                    point_in_camera.header.frame_id = camera_info.header.frame_id
                    point_in_camera.header.stamp = rospy.Time(0)
                    point_in_camera.point.x, point_in_camera.point.y, point_in_camera.point.z = cam_x, cam_y, cam_z
                    
                    # 向小脑查询此时此刻头歪了多少，并抵消
                    transform = tf_buffer.lookup_transform("base_link", camera_info.header.frame_id, rospy.Time(0), rospy.Duration(0.1))
                    point_in_base = tf2_geometry_msgs.do_transform_point(point_in_camera, transform)
                    
                    best_score = score
                    best_vla_msg = PointStamped()
                    best_vla_msg.header.stamp = rospy.Time.now()
                    best_vla_msg.header.frame_id = "base_link"
                    best_vla_msg.point = point_in_base.point
                        
                except Exception as e:
                    pass

    if best_vla_msg is not None and vla_pub is not None:
        rospy.loginfo_throttle(0.5, f"🎯 绝对坐标 (抗晃动): X={best_vla_msg.point.x:.3f}, Y={best_vla_msg.point.y:.3f}")
        vla_pub.publish(best_vla_msg)

    # ============================================================
    # 🔵 蓝色瓶盖 HSV 颜色检测 (独立通道，不影响 YOLO)
    # ============================================================
    if ENABLE_BLUE_CAP_DETECTION:
        # 用 YOLO 瓶子框的上 35% 区域作为 HSV 搜索 ROI，同时左右各扩展 10%
        cap_roi = None
        if best_bottle_box is not None:
            bx1, by1, bx2, by2 = best_bottle_box
            bh = by2 - by1
            bw = bx2 - bx1
            # 水平略微扩展，垂直取 35% 瓶盖区域
            margin_x = int(bw * 0.1)
            if CAMERA_UPSIDE_DOWN:
                # 相机倒装：图像中瓶盖在瓶子框的"下方"（像素坐标大的一端）
                cap_roi = (
                    bx1 - margin_x,
                    by2 - int(bh * 0.35),
                    bx2 + margin_x,
                    by2
                )
            else:
                # 正常：瓶盖在瓶子框上方
                cap_roi = (
                    bx1 - margin_x,
                    by1,
                    bx2 + margin_x,
                    by1 + int(bh * 0.35)
                )

            # 画黄色虚线框标记 ROI 搜索区域
            if cap_roi is not None:
                cv2.rectangle(combined_img, (cap_roi[0], cap_roi[1]), (cap_roi[2], cap_roi[3]),
                              (0, 255, 255), 1)

        result = detect_blue_cap(input_image, depth_image, camera_info, roi=cap_roi)
        if result is not None and cap_pub is not None:
            cap_msg, cx, cy, contour = result
            cap_pub.publish(cap_msg)
            rospy.loginfo_throttle(0.5,
                f"🔵 瓶盖坐标: X={cap_msg.point.x:.3f}, Y={cap_msg.point.y:.3f}, Z={cap_msg.point.z:.3f} | 像素:({cx},{cy})")
            # 画蓝色轮廓 + 半透明填充
            cv2.drawContours(combined_img, [contour], -1, (255, 0, 0), 2)
            overlay = combined_img.copy()
            cv2.drawContours(overlay, [contour], -1, (255, 0, 0), -1)
            combined_img = cv2.addWeighted(overlay, 0.3, combined_img, 0.7, 0)
            # 画质心十字
            cv2.drawMarker(combined_img, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.circle(combined_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(combined_img, f"CAP ({cx},{cy})", (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            rospy.logdebug("🔵 未检测到蓝色瓶盖")

    cv2.putText(combined_img, f"FPS: {1 / (time.time() - start_time):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return combined_img, detection_msg

def process_frames(model, executor, pub, image_pub):
    global color_image, depth_image, camera_info
    while not rospy.is_shutdown():
        if color_image is None or depth_image is None or camera_info is None:
            time.sleep(0.1)
            continue
        with frame_lock:
            in_img, in_depth = color_image.copy(), depth_image.copy()

        future = executor.submit(process_frame, model, in_img, in_depth, camera_info)
        combined_img, detection_msg = future.result()
        pub.publish(detection_msg)
        try: image_pub.publish(bridge.cv2_to_imgmsg(combined_img, "bgr8"))
        except: pass
        time.sleep(0.01)

def main():
    global vla_pub, cap_pub, tf_buffer, ENABLE_BLUE_CAP_DETECTION
    rospy.init_node('yolo_bottle_detection_node')

    # 🔵 从 ROS 参数读取蓝色瓶盖检测开关 (默认关闭)
    ENABLE_BLUE_CAP_DETECTION = rospy.get_param('~enable_blue_cap_detection', False)
    if ENABLE_BLUE_CAP_DETECTION:
        rospy.loginfo("🔵 蓝色瓶盖 HSV 检测已启用")
    else:
        rospy.loginfo("🔵 蓝色瓶盖 HSV 检测未启用 (可通过 _enable_blue_cap_detection:=true 打开)")

    # 🔥 启动 TF2 监听器
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)

    pub = rospy.Publisher('/object_yolo_box_segment_result', Detection2DArray, queue_size=1)
    image_pub = rospy.Publisher('/object_yolo_box_segment_image', Image, queue_size=1)
    vla_pub = rospy.Publisher('/vla/yolo_target', PointStamped, queue_size=1)
    cap_pub = rospy.Publisher('/vla/cap_target', PointStamped, queue_size=1)

    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    if rospy.get_param('use_orbbec', True): rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback)
    else: rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_callback)
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_callback)

    model = YOLO('yolov8n-seg.pt').to('cuda')
    executor = ThreadPoolExecutor(max_workers=2)
    threading.Thread(target=process_frames, args=(model, executor, pub, image_pub), daemon=True).start()
    rospy.spin()

if __name__ == '__main__':
    main()
