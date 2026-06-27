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

# 🔥 引入 TF2 空间树核心库
import tf2_ros
import tf2_geometry_msgs

import torch
import torchvision
from ultralytics import YOLO

# ---------------------------------------------------------
# [核心机制 1]：底层 GPU NMS (非极大值抑制) 极速后处理
# ---------------------------------------------------------
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

color_image, depth_image, camera_info = None, None, None
frame_lock = threading.Lock()
bridge = CvBridge()
vla_pub, tf_buffer = None, None

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

# ---------------------------------------------------------
# [核心机制 2]：抗噪中值滤波深度提取 (提取相机坐标系下的原生 3D 点)
# ---------------------------------------------------------
def convert_to_3d(u, v, depth_image, camera_info, box, region_factor=0.5):
    # 获取相机内参矩阵 (焦距与光学中心)
    fx, fy, cx, cy = camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5]
    
    # 截取 Bounding Box 中心 50% 的矩形区域
    bw, bh = box[2] - box[0], box[3] - box[1]
    rw, rh = int(bw * region_factor), int(bh * region_factor)
    u_min, u_max = max(0, u - rw // 2), min(depth_image.shape[1], u + rw // 2)
    v_min, v_max = max(0, v - rh // 2), min(depth_image.shape[0], v + rh // 2)
    
    # 提取深度矩阵并剔除 0 值黑洞
    depth_region = depth_image[v_min:v_max, u_min:u_max]
    depth_values = depth_region[depth_region > 0]  
    if len(depth_values) == 0: return None  

    # 提取该区域的深度中位数，无视反光与穿透噪点
    z = np.median(depth_values) / 1000.0  
    
    # 根据相似三角形原理，逆投影计算相机坐标系下的 X(左右) 和 Y(上下)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 返回纯粹的、相对于相机镜头的 [x, y, z] 物理坐标
    return x, y, z

# ---------------------------------------------------------
# [核心机制 3]：视觉与 TF2 矩阵的巅峰融合
# ---------------------------------------------------------
def process_frame(model, input_image, depth_image, camera_info):
    global vla_pub, tf_buffer
    start_time = time.time()
    # YOLO 推理
    results = model(input_image, imgsz=640, verbose=False)
    
    boxes, scores, class_ids = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy(), results[0].boxes.cls.cpu().numpy().astype(int)
    combined_img = input_image.copy()
    detection_msg = Detection2DArray()
    best_vla_msg, best_score = None, 0.0

    for box, score, class_id in zip(boxes, scores, class_ids):
        # 极低阈值 (0.15) 锁敌，防丢失
        if model.names[int(class_id)] != 'bottle' or score < 0.15: continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if depth_image is not None and score > best_score:
            u, v = int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0)
            res_3d = convert_to_3d(u, v, depth_image, camera_info, box)
            
            if res_3d is not None:
                cam_x, cam_y, cam_z = res_3d
                
                # 🔥🔥🔥 TF2 动态抗晃动解算核心区块 🔥🔥🔥
                try:
                    # 1. 封装相机局部坐标
                    point_in_camera = PointStamped()
                    point_in_camera.header.frame_id = camera_info.header.frame_id
                    point_in_camera.header.stamp = rospy.Time(0) # 取最新时间
                    point_in_camera.point.x, point_in_camera.point.y, point_in_camera.point.z = cam_x, cam_y, cam_z
                    
                    # 2. 查询变换矩阵：向机器人小脑查询从 camera_link 到 base_link 的旋转平移关系
                    transform = tf_buffer.lookup_transform("base_link", camera_info.header.frame_id, rospy.Time(0), rospy.Duration(0.1))
                    
                    # 3. 空间矩阵相乘：瞬间抹平头部晃动误差
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

    cv2.putText(combined_img, f"FPS: {1 / (time.time() - start_time):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return combined_img, detection_msg

# 多线程解耦，防 ROS 通讯撕裂
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
    global vla_pub, tf_buffer
    rospy.init_node('yolo_bottle_detection_node')

    # 🔥 启动 TF2 监听器池
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)

    pub = rospy.Publisher('/object_yolo_box_segment_result', Detection2DArray, queue_size=1)
    image_pub = rospy.Publisher('/object_yolo_box_segment_image', Image, queue_size=1)
    vla_pub = rospy.Publisher('/vla/yolo_target', PointStamped, queue_size=1)

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
