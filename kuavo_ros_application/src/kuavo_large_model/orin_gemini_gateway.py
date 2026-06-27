# -*- encoding:utf-8 -*-
# 运行在上位机 Jetson Orin NX (192.168.26.12)
import os
import sys
import asyncio
import websockets
import json
import base64
import pyaudio
import socket
import audioop
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  
CHUNK = 1024

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("[致命错误] 找不到 GEMINI_API_KEY 环境变量！请先执行 export")
    sys.exit(1)

WS_URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}"

LOWER_COMPUTER_IP = "192.168.26.1"
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

audio = pyaudio.PyAudio()

# ================= 全局状态变量 =================
last_audio_recv_time = 0.0
expected_finish_time = 0.0  

# ================= 1. 上位机真听觉雷达 (Ear) =================
async def mic_stream(ws):
    global expected_finish_time
    
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("-> 🎙️【上位机头部麦克风】已成功唤醒，开始长连接捕获人声...")
    
    # 💥 核心修复 2：将门槛拉高到 5000，彻底镇压 3900+ 的极限风扇噪音！
    THRESHOLD = 5000  
    radar_counter = 0

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = audioop.rms(data, 2)
            
            is_robot_speaking = time.time() < (expected_finish_time + 0.5)

            if is_robot_speaking:
                # 机器人说话及“网络卡顿喘气”期间，麦克风绝对静音
                data = b'\x00' * len(data)
            else:
                radar_counter += 1
                if radar_counter % 10 == 0:
                    print(f"\r\033[K📊 [物理雷达] 当前环境音量: {rms:4d}  (风扇门槛: {THRESHOLD})", end='', flush=True)

                if rms < THRESHOLD:
                    data = b'\x00' * len(data)

            msg = {
                "realtimeInput": {
                    "audio": {
                        "mimeType": "audio/pcm;rate=16000",
                        "data": base64.b64encode(data).decode("utf-8")
                    }
                }
            }
            await ws.send(json.dumps(msg))
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()

# ================= 2. 跨机器网络嘴巴流式投送 (Mouth) =================
async def speaker_stream(ws):
    global last_audio_recv_time, expected_finish_time
    
    print("-> 🚀【跨端音频网络总线】已打通！云端音频将流式全额投送至下位机物理扬声器...")
    try:
        while True:
            response_raw = await ws.recv()
            response = json.loads(response_raw)
            
            server_content = response.get("serverContent", {})
            model_turn = server_content.get("modelTurn", {})
            parts = model_turn.get("parts", [])

            for part in parts:
                inline_data = part.get("inlineData", {})
                if inline_data.get("mimeType", "").startswith("audio/"):
                    audio_b64 = inline_data.get("data")
                    if audio_b64:
                        pcm_data = base64.b64decode(audio_b64)
                        
                        chunk_duration = len(pcm_data) / 48000.0
                        now = time.time()
                        
                        if now > expected_finish_time:
                            expected_finish_time = now + chunk_duration
                        else:
                            expected_finish_time += chunk_duration
                            
                        # 💥 核心修复 1：终极看门狗机制！
                        # 无论这个切片有多短，只要收到了包，麦克风强制封锁期至少延长到 3.0 秒后！
                        # 完美免疫谷歌服务器“发一半停2秒再发”的网络抽风导致的开麦打断。
                        expected_finish_time = max(expected_finish_time, now + 3.0)
                            
                        if now - last_audio_recv_time > 2.0:
                            print("\n🧠 [夸父]: 🔊 (正在流式输出原生男声...)")
                            
                        last_audio_recv_time = now
                        udp_socket.sendto(pcm_data, (LOWER_COMPUTER_IP, 6000))
                        
    except asyncio.CancelledError:
        pass
    except websockets.exceptions.ConnectionClosed:
        print("\n-> 🛑 谷歌空间站连接已断开。")

# ================= 3. 神经调度中枢 =================
async def main():
    print("\n==================================================")
    print(" 🚀 夸父分布式全双工网关启动 (Orin耳朵 + NUC嘴巴) ...")
    print("==================================================\n")
    
    async with websockets.connect(WS_URL) as ws:
        setup_msg = {
            "setup": {
                "model": "models/gemini-3.1-flash-live-preview",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                # 💥 核心修复 3：换装 Charon，这是纯正、低沉的标准成熟男声！
                                "voiceName": "Charon" 
                            }
                        }
                    }
                },
                "systemInstruction": {
                    "parts": [{"text": "你叫夸父，是由乐聚机器人研发的功能丰富的人形机器人。你的回答必须保持专业简洁，字数严格控制在60字以内。"}]
                }
            }
        }
        await ws.send(json.dumps(setup_msg))
        await ws.recv()
        print("-> ✅ 系统指令注入完毕，全链路跨端级联彻底通车！")
        print(">>> 请直接对着机器人的头部大喊下达指令...\n")
        
        mic_task = asyncio.create_task(mic_stream(ws))
        speaker_task = asyncio.create_task(speaker_stream(ws))
        
        await asyncio.gather(mic_task, speaker_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] 收到退出信号，安全释放系统资源。")
        audio.terminate()
        sys.exit(0)
