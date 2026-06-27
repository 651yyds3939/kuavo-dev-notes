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
import torch
import torchvision
from ultralytics import YOLO

# NMS (非极大值抑制) 用于过滤重叠的检测框
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
vla_pub = None

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

# 🔥 [核心机制 1]：中心区域深度提取与中值滤波
def convert_to_3d(u, v, depth_image, camera_info, box, region_factor=0.5):
    fx, fy, cx, cy = camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5]
    
    # 不取单一像素点，而是取目标中心 50% 的矩形区域
    bw, bh = box[2] - box[0], box[3] - box[1]
    rw, rh = int(bw * region_factor), int(bh * region_factor)
    u_min, u_max = max(0, u - rw // 2), min(depth_image.shape[1], u + rw // 2)
    v_min, v_max = max(0, v - rh // 2), min(depth_image.shape[0], v + rh // 2)
    
    depth_region = depth_image[v_min:v_max, u_min:u_max]
    depth_values = depth_region[depth_region > 0]  # 剔除无效深度黑洞
    if len(depth_values) == 0: return None  

    # 🔥 取该区域所有深度像素的“中位数”，有效免疫水瓶边缘的反光和折射噪点！
    z = np.median(depth_values) / 1000.0  # z: 距离相机的真实物理直线深度 (m)
    x = (u - cx) * z / fx                 # x: 像素在水平方向的物理偏移 (右正左负)
    
    # 丢弃相机的上下 y 轴，因为我们高度靠下位机定死
    return z, x

def process_frame(model, input_image, depth_image, camera_info):
    global vla_pub
    start_time = time.time()
    results = model(input_image, imgsz=640, verbose=False)
    
    boxes, scores, class_ids = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy(), results[0].boxes.cls.cpu().numpy().astype(int)
    combined_img = input_image.copy()
    detection_msg = Detection2DArray()

    best_vla_msg, best_score = None, 0.0

    for box, score, class_id in zip(boxes, scores, class_ids):
        # 仅过滤水瓶，且置信度大于 0.15
        if model.names[int(class_id)] != 'bottle' or score < 0.15: continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if depth_image is not None and score > best_score:
            u, v = int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0)
            res_3d = convert_to_3d(u, v, depth_image, camera_info, box)
            
            if res_3d is not None:
                depth_z, horiz_x = res_3d
                best_score = score
                best_vla_msg = PointStamped()
                best_vla_msg.header.stamp = rospy.Time.now()
                
                # 🔥 [核心机制 2]：坐标系暴力替换法！抛弃不稳定的 TF2 矩阵。
                # 由于假设头低着不晃动，我们直接把相机的测量值粗暴塞给基座 X 和 Y：
                # 1. 机器人的前方 X 轴，直接等于相机的直线测距 Z
                best_vla_msg.point.x = depth_z
                # 2. 机器人的左方 Y 轴，直接等于相机的水平偏移 X (由于相机右为正，所以这里取负号抵消)
                best_vla_msg.point.y = -horiz_x 
                # 3. 机器人的高度 Z 轴，置 0 丢弃！留给下位机安全锁死。
                best_vla_msg.point.z = 0.0 

    if best_vla_msg is not None and vla_pub is not None:
        rospy.loginfo_throttle(0.5, f"🎯 原生物理坐标锁定: 前向X={best_vla_msg.point.x:.3f}m, 侧向Y={best_vla_msg.point.y:.3f}m")
        vla_pub.publish(best_vla_msg)

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
    global vla_pub
    rospy.init_node('yolo_bottle_detection_node')
    pub = rospy.Publisher('/object_yolo_box_segment_result', Detection2DArray, queue_size=1)
    image_pub = rospy.Publisher('/object_yolo_box_segment_image', Image, queue_size=1)
    vla_pub = rospy.Publisher('/vla/yolo_target', PointStamped, queue_size=1)

    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback)
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_callback)

    model = YOLO('yolov8n-seg.pt').to('cuda')
    executor = ThreadPoolExecutor(max_workers=2)
    threading.Thread(target=process_frames, args=(model, executor, pub, image_pub), daemon=True).start()
    rospy.spin()

if __name__ == '__main__':
    main()
