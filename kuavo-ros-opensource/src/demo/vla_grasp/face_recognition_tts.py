# -*- encoding:utf-8 -*-
# 部署于: 下位机 Intel NUC (192.168.26.1)

import http.server
import json
import os
import numpy as np
import sherpa_onnx
import threading
import time

# 🌟 引入 ROS 基础设施，借道官方专有音频总线
import rospy
try:
    from kuavo_msgs.srv import playmusic, playmusicRequest
except ImportError:
    print("❌ 找不到 kuavo_msgs，请确保你在当前终端执行了 source devel/setup.bash ！")
    exit(1)

print("==================================================")
print("正在自动搜寻并加载下位机本地 TTS 模型...")

# 全自动地毯式搜寻解压出的模型文件
model_dir = None
for root, dirs, files in os.walk("/home/lab"):
    if "vits-aishell3.onnx" in files and "tokens.txt" in files:
        model_dir = root
        break

if not model_dir:
    print("[错误] 未在 /home/lab 目录下找到任何解压后的 TTS 模型文件！")
    model_dir = "/home/lab/vits-zh-aishell3"  # 默认备用路径
else:
    print(f"[成功] 自动寻找到模型目录: {model_dir}")

# 初始化原生 Python TTS 引擎
config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(model_dir, "vits-aishell3.onnx"),
            lexicon=os.path.join(model_dir, "lexicon.txt"),
            tokens=os.path.join(model_dir, "tokens.txt"),
        ),
        num_threads=4,
    )
)

try:
    tts = sherpa_onnx.OfflineTts(config)
    print("下位机 Python 原生 TTS 引擎初始化成功！")
except Exception as e:
    print(f"TTS引擎加载失败，请检查模型完整性: {e}")
print("==================================================")

def ros_audio_player_worker(wav_path):
    """
    🌟 借道官方 ROS 节点播放：
    通过官方已经霸占了声卡的 /play_music_node 代为播放 WAV 文件，100% 免疫声卡设备冲突锁死！
    """
    try:
        # 阻断等待官方音频服务挂起
        rospy.wait_for_service('/play_music', timeout=3.0)
        robot_music_play_client = rospy.ServiceProxy("/play_music", playmusic)
        
        # 封装标准官方播音请求
        request = playmusicRequest()
        request.music_number = wav_path  # 传入绝对文件路径
        request.volume = 95               # 霸气拉满音量，确保实验室听得清
        
        print(f"  [ROS 总线] 正在将音频交付给官方播放节点: {wav_path}")
        response = robot_music_play_client(request)
        if response.success_flag:
            print("  [ROS 总线] ✅ 官方音频节点代播成功。")
        else:
            print("  [ROS 总线] ❌ 官方音频节点拒绝播放。")
    except Exception as e:
        print(f"  [ROS 总线错误] 无法借道官方发声通道: {e}")


class TTSRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/tts":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode("utf-8"))
                text = data.get("text", "")

                print(f"-> 收到来自上位机的合成文本: {text}")

                # 1. 模型特征快速矩阵合成音频
                audio = tts.generate(text, sid=0, speed=1.0)
                samples = np.array(audio.samples)

                # 🌟 核心修复：数字暴力增益！
                # 将原始波形的振幅强行放大 3.0 倍 (如果觉得不够可以改 4.0，觉得太大可以改 2.0)
                samples = samples * 3.0 
                
                # 削峰填谷：切掉超出物理极值的波峰，防止音响滋啦滋啦破音
                samples = np.clip(samples, -1.0, 1.0)
                pcm_bytes = (samples * 32767).astype(np.int16).tobytes()

                # 2. 保存成标准的 WAV 缓存文件
                import wave
                wav_path = "/home/lab/reply.wav"
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(audio.sample_rate)
                    wf.writeframes(pcm_bytes)
                
                # 3. 斩断局域网超时：生成完毕立刻返回 HTTP 200，释放上位机堵塞
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "processing"}')
                print("-> ⚡ 已秒回上位机释放总线，正在向后台异步派发音频...")

                # 4. 启动独立后台 Daemon 线程去调用 ROS 播放
                threading.Thread(
                    target=ros_audio_player_worker, 
                    args=(wav_path,), 
                    daemon=True
                ).start()

            except Exception as e:
                print(f"[错误] 语音总线调度失败: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Server Error: {e}".encode())
        else:
            self.send_response(404)
            self.end_headers()

def run():
    # 必须开启 anonymous=True 防止与下位机其他节点重名
    rospy.init_node('kuavo_tts_offline_server', anonymous=True, disable_signals=True)
    server_address = ("", 5000)
    httpd = http.server.HTTPServer(server_address, TTSRequestHandler)
    print("\n🚀 【下位机】异步大功率离线语音解耦服务已在 5000 端口挂起通车...\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 [TTS系统] 接收到用户退出指令 (Ctrl+C)，正在释放5000端口...")
        httpd.server_close()
        print("👋 离线语音服务已安全退出！")

if __name__ == "__main__":
    run()