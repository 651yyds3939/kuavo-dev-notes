#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import base64
import requests
import threading
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

class VLMPersonObserver:
    def __init__(self):
        # 1. 注册为独立的 ROS 节点，anonymous=True 防止名称冲突
        rospy.init_node('vlm_person_observer', anonymous=True)
        self.bridge = CvBridge()
        self.latest_frame = None  # 图像无锁内存池
        
        # ================= 100% 本地局域网配置 =================
        self.nuc_ip = "192.168.26.1"  # 严格指向下位机大脑所在的局域网 IP
        self.vlm_model = "minicpm-v"  # 挂载面壁智能纯血国产多模态大模型
        
        # 2. 系统防御机制初始化
        self.cooldown = 15.0       # 硬件 CD 冷却时间，防止复读机式播报
        self.last_trigger_time = 0.0
        self.is_analyzing = False  # 线程并发锁，严防并发网络风暴
        
        # 3. 核心：双通道订阅
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/vla/person_detected', Bool, self.person_trigger_callback)
        
        print("\n==================================================")
        print("👁️ 视觉大模型独立观察者节点已启动，当前完全由 [图像识别] 触发！")
        print("==================================================\n")

    # ---------------- 图像内存池高频刷新回调 ----------------
    def image_callback(self, msg):
        try:
            # 将 ROS Image 消息转化为 OpenCV BGR8 格式矩阵
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass # 战略性忽略脏数据帧

    # ---------------- 扳机信号拦截回调 ----------------
    def person_trigger_callback(self, msg):
        # 拦截过滤网 1：非 True 信号不予理睬
        if not msg.data: return
        # 拦截过滤网 2：并发锁，大脑在思考时拒绝新请求
        if self.is_analyzing: return
        # 拦截过滤网 3：空指针防御，防止相机离线导致的段错误
        if self.latest_frame is None: return
        
        # 拦截过滤网 4：冷酷的 15 秒 CD 时间限制
        if time.time() - self.last_trigger_time < self.cooldown:
            return

        print("\n🚨 [图像触发器] YOLO 确认视野内出现人类！截取当前帧画面...")
        self.last_trigger_time = time.time()
        
        # 深拷贝当前瞬间的快照，防止被后台的高频图像流覆盖污染
        frame_to_analyze = self.latest_frame.copy()
        
        # 启动独立线程向 NUC 发起网络请求，防止阻塞 ROS 的回调循环 (Spin)
        threading.Thread(target=self.analyze_person_task, args=(frame_to_analyze,)).start()

    # ---------------- 核心：多模态编码与跨端 API 交互 ----------------
    def analyze_person_task(self, frame):
        # 加锁
        self.is_analyzing = True
        try:
            # 1. 暴力算力降维：截断图像尺寸，极大削减 Vision Token
            resized_frame = cv2.resize(frame, (320, 240))
            
            # 2. 内存级 Base64 编码，无需经过磁盘 IO
            _, buffer = cv2.imencode('.jpg', resized_frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            # 3. 🌟 Bug 修复：强行显式声明局域网推断 API 地址
            ollama_url = f"http://{self.nuc_ip}:11434/api/chat"
            
            # 4. 防幻觉提示词工程
            system_prompt = """
            你现在是一个机器人的视觉中枢。请简练地描述画面里的人。
            要求：
            1. 致命规则：如果画面中没有清晰的人类，请直接回复“未检测到人”，绝不能凭空捏造！
            2. 如果有人，只描述其明显的衣着颜色或姿态动作。
            3. 字数严控在25字以内，直接输出结果，不要废话。
            """
            
            # 5. 组装符合 Ollama /api/chat 协议规范的 RESTful Payload
            payload = {
                "model": self.vlm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": "请分析画面里的人。", 
                        "images": [base64_image]  # 注入编码后的图像阵列
                    }
                ],
                "stream": False # 强制非流式输出，等待整句思考完毕后一并返回
            }

            print("🧠 [VLM分析中] 纯视觉快照已上传下位机，等待 MiniCPM-V 大模型推理...")
            
            # 6. 发起 HTTP 阻塞请求，配置 120 秒防御性超时
            response = requests.post(ollama_url, json=payload, timeout=120)
            # 如果下位机返回 404 或 500 错误，直接抛出异常被 except 捕获
            response.raise_for_status()
            
            # 7. 提取大模型纯文本精华输出
            analysis_result = response.json()["message"]["content"]
            print(f"✅ [大模型输出]: {analysis_result}")
            
            # 8. 无缝转接：跨端直接打给 NUC 上的 5000 端口驱动 TTS 发声
            tts_url = f"http://{self.nuc_ip}:5000/tts"
            requests.post(tts_url, json={"text": analysis_result}, timeout=5)
            
        except Exception as e:
            print(f"❌ [VLM分析失败]: {e}")
        finally:
            # 绝对不能忘记在 finally 块释放并发锁
            self.is_analyzing = False

if __name__ == '__main__':
    observer = VLMPersonObserver()
    rospy.spin() # ROS 主循环，接管主线程
