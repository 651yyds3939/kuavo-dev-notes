# -*- encoding:utf-8 -*-
# 运行于上位机，绝对不要加 proxychains4！
import cv2
import rospy
import socket
import json
import base64
import time
import os
import glob
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from insightface.app import FaceAnalysis

bridge = CvBridge()
latest_frame = None
last_seen_person = None
person_leave_time = 0

# 🌟 本地 UDP 发射器，专门向 Gemini 网关空投照片
udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
GATEWAY_ADDR = ("127.0.0.1", 7000)

print("==================================================")
print("正在加载 InsightFace 视觉识别引擎...")
face_app = FaceAnalysis(name="buffalo_l", root="~/.insightface", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

known_faces = {}
faces_dir = os.path.join(os.path.dirname(__file__), "faces")
if os.path.exists(faces_dir):
    for img_path in glob.glob(os.path.join(faces_dir, "*.*")):
        name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is not None:
            faces = face_app.get(img)
            if faces:
                known_faces[name] = faces[0].embedding
print(f"✅ 已成功录入熟人特征库: {list(known_faces.keys())}")

def image_callback(msg):
    global latest_frame
    try: latest_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except: pass

def main():
    global last_seen_person, person_leave_time, latest_frame
    rospy.init_node('vision_observer_standalone', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    
    print("-> 👁️ 视觉观察者已启动，全自动巡逻中...")
    
    while not rospy.is_shutdown():
        time.sleep(1.0) # 1Hz 低频巡视
        if latest_frame is None: continue
            
        frame_copy = latest_frame.copy()
        faces = face_app.get(frame_copy)
        
        if not faces:
            if last_seen_person is not None and time.time() - person_leave_time > 10.0:
                print(f"-> 📭 {last_seen_person} 已离开，记忆重置。")
                last_seen_person = None
            if last_seen_person is not None and person_leave_time == 0:
                person_leave_time = time.time()
            continue
            
        person_leave_time = 0
        main_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        target_emb = main_face.embedding
        
        best_match = "陌生人"
        max_sim = 0.35 
        for name, emb in known_faces.items():
            sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb))
            if sim > max_sim:
                max_sim = sim
                best_match = name
                
        # 🌟 发现新面孔，立刻向 127.0.0.1:7000 空投！
        if best_match != last_seen_person:
            print(f"\n🚨 [视觉扳机] 锁定目标: {best_match}！正在将照片传送给 Gemini 大脑...")
            last_seen_person = best_match
            
            # 暴力压缩照片到几KB以内，防 UDP 丢包
            resized_frame = cv2.resize(frame_copy, (320, 240))
            _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            if best_match == "陌生人":
                prompt_text = "系统提示：面前出现了一位陌生人。这是他的照片。请主动用一句热情的中文打招呼，描述一下他的衣着特征，并问他有什么需要。"
            else:
                prompt_text = f"系统提示：你的老熟人【{best_match}】出现在你面前。这是他的照片。请主动用一句像老朋友一样的中文打招呼，顺便夸一夸他的打扮。"

            payload = json.dumps({"text": prompt_text, "image": base64_image})
            udp_client.sendto(payload.encode('utf-8'), GATEWAY_ADDR)

if __name__ == "__main__":
    main()
