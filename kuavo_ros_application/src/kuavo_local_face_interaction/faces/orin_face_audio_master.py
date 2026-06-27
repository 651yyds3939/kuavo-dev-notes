# -*- encoding:utf-8 -*-
# 文件名: orin_face_audio_master.py
# 部署于: 上位机 Jetson Orin NX (192.168.26.12)

import os
import sys
import time
import wave
import audioop
import pyaudio
import requests
import json
import cv2
import numpy as np
import glob
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from faster_whisper import WhisperModel
from insightface.app import FaceAnalysis

# ================= 1. 基础配置 =================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
THRESHOLD = 5000        # 风扇降噪门槛
SILENCE_DURATION = 0.8  
INPUT_FILE = "input.wav"

LOWER_COMPUTER_IP = "192.168.26.1"
global_audio = pyaudio.PyAudio()
bridge = CvBridge()
latest_frame = None

print("==================================================")
print("正在加载系统资源...")

# ================= 2. 视觉引擎初始化 =================
# 强制使用 CPU 进行人脸提取，避免和 Whisper 抢夺 Orin NX 珍贵的 GPU 显存
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
                print(f"✅ 已成功录入人脸特征: {name}")

# ================= 3. 听觉引擎初始化 =================
# 满血释放 Ampere GPU 算力
whisper_model = WhisperModel(
    "/home/leju_kuavo/faster-whisper-small", 
    device="cuda", 
    compute_type="float16", 
    local_files_only=True
)
print("多模态融合系统初始化成功！")
print("==================================================")

# ================= 4. ROS 图像回调 =================
def image_callback(msg):
    """无锁内存池，高频刷新相机快照"""
    global latest_frame
    try:
        latest_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception:
        pass

# ================= 5. 核心逻辑函数 =================
def identify_person_in_view():
    """执行人脸 1:1 特征向量比对"""
    global latest_frame
    if latest_frame is None:
        return None

    faces = face_app.get(latest_frame)
    if not faces:
        return None

    # 提取画面中最大的人脸，防止背景路人干扰
    main_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    target_emb = main_face.embedding

    best_match = None
    max_sim = 0.5  # 相似度红线

    for name, emb in known_faces.items():
        sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb))
        if sim > max_sim:
            max_sim = sim
            best_match = name

    return best_match

def chat():
    """麦克风阵列 VAD 监听"""
    stream = global_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("\n>>> 【夸父】正在倾听，请提问 (可包含关键词'是谁')...")

    frames = []
    recording = False
    silence_start = None
    radar_counter = 0

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except KeyboardInterrupt:
            print("\n[!] 收到退出信号，正在安全释放声卡并关闭系统...")
            # 1. 立即停止并有关闭当前的音频流
            stream.stop_stream()
            stream.close()
            # 2. 销毁 PyAudio 全局句柄
            global_audio.terminate()
            # 3. 干净利落地直接终止整个 Python 进程
            import sys
            sys.exit(0)
        rms = audioop.rms(data, 2)
        
        radar_counter += 1
        if radar_counter % 5 == 0 and not recording:
            print(f"📊 [音量雷达] 当前环境音量: {rms:4d}  (门槛: {THRESHOLD})", end='\r')

        if rms > THRESHOLD:
            if not recording:
                recording = True
                print("\n-> 🎙️ 检测到有效人声，开始录音...")
            frames.append(data)
            silence_start = None  
        else:
            if recording:
                frames.append(data)  
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("-> 🛑 说话结束，准备转写...")
                    break

    stream.stop_stream()
    stream.close()

    with wave.open(INPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(global_audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def local_stt(file_path):
    """GPU 音频转写"""
    try:
        segments, info = whisper_model.transcribe(
            file_path, beam_size=5, language="zh",
            initial_prompt="这是一段日常对话。", condition_on_previous_text=False
        )
        text = "".join([segment.text for segment in segments if segment.no_speech_prob <= 0.8])
        return text.strip()
    except Exception as e:
        print(f"【耳朵报错】: {e}")
        return ""

def local_offline_closed_loop_chat(text, detected_name=None):
    """跨端呼叫下位机 NUC"""
    ollama_url = f"http://{LOWER_COMPUTER_IP}:11434/api/chat"
    
    # 🌟 动态 Prompt 注入：赋予大模型情境认知
    if detected_name:
        sys_prompt = f"你叫夸父，是由乐聚机器人研发的功能丰富的人形机器人。你的视觉雷达刚刚识别到了面前的人是【{detected_name}】。你的回答必须保持专业简洁，字数严格控制在60字以内。绝密指令：你的语音发声器官只能朗读纯汉字，因此你的回答中绝对不能包含阿拉伯数字和英文字母。请直接用一句热情的话欢迎他，并回答他的问题。"
    else:
        sys_prompt = "你叫夸父，是由乐聚机器人研发的功能丰富的人形机器人。面前出现了一个你不认识的陌生人。你的回答必须保持专业简洁，字数严格控制在60字以内。绝密指令：你的语音发声器官只能朗读纯汉字，因此你的回答中绝对不能包含阿拉伯数字和英文字母。请礼貌地说明你不认识他，并询问他有什么需要帮助的。"

    ollama_payload = {
        "model": "qwen2:7b",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text}
        ],
        "stream": False
    }

    try:
        print(f"  [你]: {text}")
        response = requests.post(ollama_url, json=ollama_payload, timeout=60)
        response.raise_for_status() 
        reply = response.json()["message"]["content"]
        
        print(f"  [夸父]: {reply}")
        
        tts_url = f"http://{LOWER_COMPUTER_IP}:5000/tts"
        requests.post(tts_url, json={"text": reply}, timeout=60)
        print("-> ✅ 交互闭环完成！")
        
    except Exception as e:
        print(f"【局域网通信失败】: {e}")

# ================= 6. 主控制流 =================
if __name__ == '__main__':
    # 注册独立 ROS 节点以订阅相机画面
    rospy.init_node('standalone_face_audio_master', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)

    try:
        while not rospy.is_shutdown():
            chat()
            sentence = local_stt(INPUT_FILE)
            clean_sentence = sentence.replace("。", "").replace("，", "").replace("！", "").replace("？", "").strip()
            
            if clean_sentence:
                # 🌟 核心拦截机制：听到“是谁”才扣动视觉扳机
                if "是谁" in clean_sentence:
                    print("-> 👁️ 触发视觉指令，执行特征抽取与比对...")
                    person_name = identify_person_in_view()
                    if person_name:
                        print(f"-> 🎯 锁定目标: {person_name}")
                    else:
                        print("-> ⚠️ 未匹配到已知人脸。")
                    local_offline_closed_loop_chat(sentence, detected_name=person_name)
                else:
                    local_offline_closed_loop_chat(sentence)
            else:
                print("-> 📭 录音空白，重新倾听。")
                
    except KeyboardInterrupt:
        print("\n[!] 安全关闭。")
        global_audio.terminate()
        sys.exit(0)