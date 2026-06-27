# -*- encoding:utf-8 -*-
# 运行在下位机 Intel NUC (192.168.26.1)
import socket
import pyaudio
import sys

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000  # 🌟 修复：完美对齐谷歌云端的 24kHz 高清输出采样率，杜绝磁带慢放效应
CHUNK = 1024

p = pyaudio.PyAudio()
# 自动咬住下位机默认物理扬声器
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

# 开启高速 UDP 监听，免去 TCP 握手与 ROS 序列化开销
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('0.0.0.0', 6000))

print("==================================================")
print(" 🔊【下位机】物理立体音响就绪，正在静默等待上位机音频流...")
print("==================================================")

try:
    while True:
        # 🌟 修复：将缓冲区拉满至物理极限 65536，防止大体积高清切片被网卡粗暴截断
        data, addr = server_socket.recvfrom(65536)
        if data:
            stream.write(data)  # 瞬间冲刷 NUC 声卡硬件 Buffer，无缝衔接
except KeyboardInterrupt:
    print("\n[!] 释放下位机物理声卡。")
    stream.stop_stream()
    stream.close()
    p.terminate()
