# -*- encoding:utf-8 -*-
# 部署于: 下位机 NUC 192.168.26.1
import os
import rospy
import threading
from flask import Flask, request, jsonify

try:
    from kuavo_msgs.srv import playmusic, playmusicRequest
except ImportError:
    print("❌ 找不到 kuavo_msgs，请 source setup.bash")
    exit(1)

app = Flask(__name__)

def ros_audio_player_worker(wav_path):
    try:
        rospy.wait_for_service('/play_music', timeout=3.0)
        client = rospy.ServiceProxy("/play_music", playmusic)
        req = playmusicRequest()
        req.music_number = wav_path
        req.volume = 100  # 拉满音量
        client(req)
        print(f"✅ 官方通道代播成功: {wav_path}")
    except Exception as e:
        print(f"❌ 官方通道播放失败: {e}")

@app.route('/play_audio', methods=['POST'])
def play_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
        
    file = request.files['file']
    wav_path = "/home/lab/gemini_reply.wav"
    file.save(wav_path)
    
    # 放入后台线程播放，立刻返回 200 释放网络，防阻塞
    threading.Thread(target=ros_audio_player_worker, args=(wav_path,), daemon=True).start()
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    rospy.init_node('kuavo_gemini_audio_receiver', anonymous=True, disable_signals=True)
    print("==================================================")
    print("🚀 【下位机】官方通道音频接收服务挂起于 5000 端口...")
    print("==================================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
