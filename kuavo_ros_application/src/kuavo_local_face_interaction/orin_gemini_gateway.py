# -*- encoding:utf-8 -*-
# 运行于上位机，必须使用 proxychains4！
import os
import sys
import asyncio
import websockets
import json
import base64
import pyaudio
import socket
import audioop
import wave
import requests

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("[致命错误] 找不到 GEMINI_API_KEY")
    sys.exit(1)

WS_URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}"
LOWER_COMPUTER_IP = "192.168.26.1"

# 监听视觉脚本图文触发的本地服务
trigger_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
trigger_socket.bind(("127.0.0.1", 7000))
trigger_socket.setblocking(False)

audio = pyaudio.PyAudio()
FORMAT, CHANNELS, RATE, CHUNK, THRESHOLD = pyaudio.paInt16, 1, 16000, 1024, 4500

# 🌟 修复入口：需要传入全局锁 ws_lock
async def vision_trigger_listener(ws, ws_lock):
    """异步监听本地 7000 端口，一旦收到视觉情报，立刻插嘴！"""
    while True:
        try:
            data, _ = trigger_socket.recvfrom(65536)
            payload = json.loads(data.decode('utf-8'))
            print(f"\n🧠 [网关接收] 收到视觉中枢情报，正强行注入大模型...")
            
            msg = {
                "clientContent": {
                    "turns": [{
                        "role": "user",
                        "parts": [
                            {"text": payload["text"]},
                            {"inlineData": {"mimeType": "image/jpeg", "data": payload["image"]}}
                        ]
                    }],
                    "turnComplete": True
                }
            }
            # 🌟 核心修复 1：加锁发送！拿到绿灯才准发包，杜绝并发撞车！
            async with ws_lock:
                await ws.send(json.dumps(msg))
                print(f"🚀 [网关发送] 视觉图文包已成功越过封锁，发射至谷歌云端！")
                
        except BlockingIOError:
            pass 
        except Exception as e:
            # 🌟 核心修复 2：扒掉害死人的 pass 隐身衣，哪怕出错也要死得明明白白！
            print(f"❌ [网关内部发送报错]: {e}")
        await asyncio.sleep(0.1)

async def mic_stream(ws, ws_lock):
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    radar_counter = 0
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = audioop.rms(data, 2)
            if rms < THRESHOLD: 
                data = b'\x00' * len(data)
            else:
                # 🌟 增加这行日志，一旦你说话，终端必须刷出这句话！
                print(f"\r\033[K🎙️ [已检测到人声] 正在向云端推流...", end='', flush=True)
            radar_counter += 1
            if radar_counter % 10 == 0:
                print(f"\r\033[K📊 [雷达] 环境音量: {rms:4d}  (门槛: {THRESHOLD})", end='', flush=True)

            msg = {
                "realtimeInput": {
                    "audio": {
                        "mimeType": "audio/pcm;rate=16000",
                        "data": base64.b64encode(data).decode("utf-8")
                    }
                }
            }
            # 🌟 核心修复 3：麦克风也要加锁发送，遵守交通规则！
            async with ws_lock:
                await ws.send(json.dumps(msg))
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        stream.stop_stream()
        stream.close()

async def speaker_stream(ws):
    audio_buffer = bytearray()
    try:
        while True:
            response_raw = await ws.recv()
            response = json.loads(response_raw)
            
            # 照妖镜 2.0 逻辑：过滤无害的会话存档包
            if "serverContent" not in response:
                if "sessionResumptionUpdate" in response:
                    continue
                print(f"\n[⚠️ 云端警报/状态]: {json.dumps(response, ensure_ascii=False)}")
                continue

            server_content = response.get("serverContent", {})
            parts = server_content.get("modelTurn", {}).get("parts", [])
            
            for part in parts:
                if "text" in part:
                    print(f"\n🧠 [夸父]: {part['text']}", end="", flush=True)
                
                inline_data = part.get("inlineData", {})
                if inline_data.get("mimeType", "").startswith("audio/"):
                    pcm_data = base64.b64decode(inline_data.get("data"))
                    audio_buffer.extend(pcm_data)
                    
            if server_content.get("turnComplete"):
                if len(audio_buffer) > 0:
                    wav_path = "temp_gemini_reply.wav"
                    with wave.open(wav_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(audio_buffer)
                    
                    try:
                        with open(wav_path, "rb") as f:
                            url = f"http://{LOWER_COMPUTER_IP}:5000/play_audio"
                            requests.post(url, files={"file": f}, timeout=5)
                    except Exception as e:
                        print(f"\n❌ 发送音频到下位机失败: {e}")
                        
                    audio_buffer = bytearray()
                    
    except websockets.exceptions.ConnectionClosed:
        print("\n-> 🛑 谷歌空间站连接已断开。")

async def main():
    print("\n==================================================")
    print(" 🚀 Gemini 解耦网关启动，监听音频与视觉情报...")
    print("==================================================\n")
    
    async with websockets.connect(WS_URL) as ws:
        setup_msg = {
            "setup": {
                "model": "models/gemini-3.1-flash-live-preview",
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "systemInstruction": {"parts": [{"text": "你叫夸父，是由乐聚机器人研发的功能丰富的人形机器人。回答必须保持专业简洁、高情商。"}]}
            }
        }
        await ws.send(json.dumps(setup_msg))
        await ws.recv()
        print("-> ✅ 广域网 API 鉴权完毕，脑机接口通车！")
        
        # 🌟 核心修复 4：创建全局发包交通红绿灯 (Lock)
        ws_lock = asyncio.Lock()
        
        await asyncio.gather(
            asyncio.create_task(mic_stream(ws, ws_lock)),
            asyncio.create_task(speaker_stream(ws)),
            asyncio.create_task(vision_trigger_listener(ws, ws_lock))
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        audio.terminate()
        os._exit(0)
