# -*- encoding:utf-8 -*-
import os
import sys
import time
import wave
import audioop
import pyaudio
import requests
import json
from faster_whisper import WhisperModel

# ================= 音频配置 =================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# 🌟 这里的门槛请根据你雷达测出的实际数值调整（4000左右）
THRESHOLD = 4000         
SILENCE_DURATION = 0.8   
RECORD_SECONDS = 60      
INPUT_FILE = "input.wav" 
OUTPUT_FILE = "output.pcm"

LOWER_COMPUTER_IP = "192.168.26.1" 

global_audio = pyaudio.PyAudio()

print("==================================================")
print("正在加载本地 Faster-Whisper 语音识别模型...")
# 🌟 确保这里的路径是你物理拷贝进去的 small 模型的绝对路径
whisper_model = WhisperModel(
    "/home/leju_kuavo/faster-whisper-small", 
    device="cuda", 
    compute_type="float16", 
    local_files_only=True
)
print("本地语音识别系统初始化成功！")
print("==================================================")


def chat():
    stream = global_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("\n>>> 【夸父】正在倾听，请开始提问...")

    frames = []
    recording = False
    silence_start = None
    start_time = time.time()
    
    radar_counter = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms = audioop.rms(data, 2)  
        
        radar_counter += 1
        if radar_counter % 5 == 0 and not recording:
            print(f"📊 [音量雷达] 当前环境音量: {rms:4d}  (触发门槛: {THRESHOLD})", end='\r')
        
        if rms > THRESHOLD:
            if not recording:
                recording = True
                print("\n-> 🎙️ 突破门槛！检测到有效人声，开始录音...")
            frames.append(data)
            silence_start = None  
        else:
            if recording:
                frames.append(data)  
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("-> 🛑 说话结束，正在分析文本...")
                    break

        if time.time() - start_time > RECORD_SECONDS:
            break

    stream.stop_stream()
    stream.close()

    with wave.open(INPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(global_audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


def local_stt(file_path):
    try:
        segments, info = whisper_model.transcribe(
            file_path, 
            beam_size=5, 
            language="zh",
            initial_prompt="这是一段日常对话。",
            condition_on_previous_text=False
        )
        
        text = ""
        for segment in segments:
            if segment.no_speech_prob <= 0.8:
                text += segment.text
                
        return text.strip()
    except Exception as e:
        print(f"【耳朵报错】: {e}")
        return ""


def local_offline_closed_loop_chat(text):
    ollama_url = f"http://{LOWER_COMPUTER_IP}:11434/api/chat"
    
    # 🌟 核心修复：通过系统提示词，强迫大模型自己做“文本正则化”，将所有数字和英文转为纯汉字！
    ollama_payload = {
        "model": "qwen2:7b",
        "messages": [
            {
                "role": "system", 
                "content": "你叫夸父，是由乐聚机器人研发的功能丰富的人形机器人。你的回答必须保持专业简洁，字数严格控制在60字以内。绝密指令：你的语音发声器官只能朗读纯汉字，因此你的回答中【绝对不能】包含任何阿拉伯数字（如1、2、172）和英文字母（如AI、OK）。如果需要用到数字或英文，请务必将其翻译或音译为纯中文汉字输出（例如将'172'写为'一百七十二'，将'AI'写为'人工智能'）。"
            },
            {"role": "user", "content": text}
        ],
        "stream": False
    }

    try:
        print(f"  [你]: {text}")
        ollama_response = requests.post(ollama_url, json=ollama_payload, timeout=60)
        ollama_response.raise_for_status() 
        reply = ollama_response.json()["message"]["content"]
        
        print(f"  [夸父]: {reply}")
        print("-> 🧠 正在呼叫下位机进行发声，请稍候...")
        
        tts_url = f"http://{LOWER_COMPUTER_IP}:5000/tts"
        tts_response = requests.post(tts_url, json={"text": reply}, timeout=60)
        tts_response.raise_for_status()
        
        print("-> ✅ 夸父回答完毕！")
        
    except requests.exceptions.RequestException as err:
        print(f"【局域网通信失败】: {err}")
    except Exception as e:
        print(f"【发生未知错误】: {e}")


if __name__ == '__main__':
    print("\n==================================================")
    print("      Kuavo 100% 本地全离线具身交互系统已就绪       ")
    print("==================================================\n")
    
    try:
        while True:
            chat()
            sentence = local_stt(INPUT_FILE)
            
            clean_sentence = sentence.replace("。", "").replace("，", "").replace("！", "").replace("？", "").strip()
            
            if clean_sentence:
                local_offline_closed_loop_chat(sentence)
            else:
                print("-> 📭 录音已被算法判定为环境底噪或幻觉，已自动过滤，重新倾听。")
                
    except KeyboardInterrupt:
        print("\n[!] 收到退出指令，安全关闭。")
        global_audio.terminate()
        sys.exit(0)
