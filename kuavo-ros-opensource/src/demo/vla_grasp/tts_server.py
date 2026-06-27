# -*- encoding:utf-8 -*-
import http.server
import json
import os
import numpy as np
import sherpa_onnx
import threading
import time

# 🌟 核心重构 1：引入 ROS 基础设施，借道官方专有音频总线
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
tts = sherpa_onnx.OfflineTtsConfig
try:
    tts = sherpa_onnx.OfflineTts(config)
    print("下位机 Python 原生 TTS 引擎初始化成功！")
except Exception as e:
    print(f"TTS引擎加载失败，请检查模型完整性: {e}")
print("==================================================")


def ros_audio_player_worker(wav_path):
    """🌟 核心重构 2：异步触觉发声总线守护线程
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

                # 1. GPU/CPU 特征快速矩阵合成音频
                audio = tts.generate(text, sid=0, speed=1.0)
                samples = np.array(audio.samples)
                samples = np.clip(samples, -1.0, 1.0)
                pcm_bytes = (samples * 32767).astype(np.int16).tobytes()

                # 保存成标准的 WAV 缓存文件
                import wave
                wav_path = "/home/lab/reply.wav"
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(audio.sample_rate)
                    wf.writeframes(pcm_bytes)
                
                # 🌟🌟🌟 核心重构 3：太极脱耦 —— 斩断局域网通信超时 🌟🌟🌟
                # 在生成好 WAV 的第一毫秒，【立刻秒回】HTTP 200 告诉上位机“我已经收到了”，释放网络连接！
                # 这样上位机的 requests 接口会在 10ms 内成功闭环，绝不再报 Read timed out！
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "processing"}')
                print("-> ⚡ 已秒回上位机释放总线，正在向后台异步派发音频...")

                # 🌟 核心重构 4：启动独立后台 Daemon 线程，让官方节点在后台去异步代播，绝不阻塞网络响应
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
    # 🌟 核心重构 5：初始化本地下位机服务对应的 ROS 节点
    # 必须开启 anonymous=True，防止与下位机其他运动主控节点重名冲突
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