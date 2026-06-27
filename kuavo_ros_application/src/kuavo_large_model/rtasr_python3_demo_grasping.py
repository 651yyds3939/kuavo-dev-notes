import os
import sys
import time
import wave
import audioop
import pyaudio
import requests
import json
from faster_whisper import WhisperModel

# 🌟 引入 ROS 和 String 消息，用于跨网通信
import rospy
from std_msgs.msg import String

# ================= 1. 物理音频捕获层配置 =================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000     
CHUNK = 1024     

THRESHOLD = 5500         # 触发门槛：必须高于风扇底噪
SILENCE_DURATION = 0.5   # 🌟 核心防坑：VAD 肉身死等斩杀！从官方傲慢的 1.2s 强行压缩至 0.5s，实现秒回！
RECORD_SECONDS = 60      
INPUT_FILE = "input.wav" 

LOWER_COMPUTER_IP = "192.168.26.1"  # 远端下位机 IP
global_audio = pyaudio.PyAudio()

# 全局变量占位，用于后续向 ROS 发布 JSON 动作指令
vla_cmd_pub = None

print("==================================================")
print("正在加载本地 Faster-Whisper 语音识别模型...")

# ================= 2. GPU 算力引擎初始化 =================
whisper_model = WhisperModel(
    "/home/leju_kuavo/faster-whisper-small",  
    device="cuda",                            
    compute_type="float16",  # 使用半精度，为大模型节省极其宝贵的 Orin NX 显存带宽                 
    local_files_only=True                     
)
print("本地语音识别系统初始化成功！")
print("==================================================")


def chat():
    """麦克风阵列底层监听与 VAD（语音端点检测）逻辑"""
    stream = global_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("\n>>> 【夸父】正在倾听，请开始提问...")

    frames = []
    recording = False
    silence_start = None
    start_time = time.time()
    radar_counter = 0

    while not rospy.is_shutdown():  # 🌟 加入 ROS 节点存活判断，防止僵尸运行
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms = audioop.rms(data, 2)  
        
        radar_counter += 1
        # 🌟 7.1.2 极致雷达脱水输出：解决终端宽度 49 引发的无限折行刷屏惨剧
        if radar_counter % 5 == 0 and not recording:
            # \r 归位，\033[K 瞬间擦除残留，flush=True 零延迟输出，总长度严格控制在 25 字符以内
            print(f"\r\033[K🎙️ ASR-MIC: {rms:<4d} / {THRESHOLD}", end="", flush=True)
        
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
                # 只要静音超过 0.5 秒，立刻打断录音进入大模型分析，彻底终结肉身死等延迟！
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("-> 🛑 说话结束，正在分析文本...")
                    break

        if time.time() - start_time > RECORD_SECONDS:
            break

    stream.stop_stream()
    stream.close()

    # 将捕获的音频帧写入 WAV 文件供 Whisper 读取
    with wave.open(INPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(global_audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


def local_stt(file_path):
    """GPU 音频特征解码与幻觉斩杀层"""
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
            # 过滤掉置信度极低的无语音段，防止杂音干扰
            if segment.no_speech_prob <= 0.8:
                text += segment.text
                
        text = text.strip()
        
        # 🌟🌟 7.1.3 防幻觉斩杀组件：构建免思索长廊 🌟🌟
        # 解决 4000+ 风扇底噪逼迫 Whisper 脑补“字幕by索兰娅”的恶性算力污染事件
        ghost_words = ["字幕", "索兰娅", "谢谢", "八岁", "大家", "亭", "观看"]
        if any(word in text for word in ghost_words) or len(text.strip()) <= 1:
            print(f"\n🤫 [系统防御] 拦截到风扇底噪引发的语音幻觉: '{text}'，拒绝污染大模型，自动斩杀！")
            return "" # 返回空字符串，外层会直接跳过大模型推理，延迟瞬间归零！
            
        return text
    except Exception as e:
        print(f"【耳朵报错】: {e}")
        return ""


def local_offline_closed_loop_chat(text):
    """跨端 HTTP 神经调度中枢：升级为 VLA 具身动作架构"""
    global vla_cmd_pub  
    
    ollama_url = f"http://{LOWER_COMPUTER_IP}:11434/api/chat"
    
    # 严格的 Prompt 约束，强迫 LLM 输出标准的 JSON 字典
    vla_system_prompt = """
    你叫夸父，是乐聚机器人研发的人形具身智能机器人。
    你现在必须作为一个【JSON状态机】运行，严格根据人类指令输出动作意图。
    【输出格式约束】
    你必须且只能输出合法的 JSON 格式，绝不允许包含任何 Markdown 符号。
    JSON 包含：
    1. "thought": 你的推理过程。
    2. "action_cmd": {"action": "grab", "target": "目标物体名称"}。如果不需要动作，设为 {"action": "idle", "target": "none"}。
    3. "tts_reply": 你用来回答人类的话。字数在30字以内。
    【绝密发声指令】"tts_reply" 中绝对不能包含阿拉伯数字和英文字母，必须翻译为纯中文汉字。
    """

    # 单轮次状态管理：强行截断上下文，每次只传当前对话，防止 Orin NX 的 KV Cache 爆炸导致 OOM 熔断
    ollama_payload = {
        "model": "qwen2:7b",
        "format": "json", 
        "messages": [
            {"role": "system", "content": vla_system_prompt},
            {"role": "user", "content": text}
        ],
        "stream": False
    }

    try:
        print(f"  [人类指令]: {text}")
        
        # --- 1. 第一发突击：呼叫 NUC 上的 Qwen2-7B 大脑 ---
        ollama_response = requests.post(ollama_url, json=ollama_payload, timeout=120)
        ollama_response.raise_for_status() 
        raw_reply = ollama_response.json()["message"]["content"]
        
        # --- 2. JSON 解析与防御性兜底 ---
        try:
            brain_state = json.loads(raw_reply)
            thought = brain_state.get("thought", "思考异常")
            action_cmd = brain_state.get("action_cmd", {"action": "idle", "target": "none"})
            tts_reply = brain_state.get("tts_reply", "好的")
        except json.JSONDecodeError:
            print(f" [错误] 大模型 JSON 格式失控: {raw_reply}")
            action_cmd = {"action": "idle", "target": "none"}
            tts_reply = "系统认知模块出现异常，我需要重新思考。"

        print(f"  [夸父思维]: {thought}")
        print(f"  [机器指令]: 执行动作 -> {action_cmd['action']}, 目标 -> {action_cmd['target']}")
        print(f"  [语音反馈]: {tts_reply}")
        
        # --- 3. 第二发突击：将纯正中文汉字打给 NUC 上的 TTS 引擎 ---
        print("-> 🧠 正在驱动语音与底层运动模块...")
        tts_url = f"http://{LOWER_COMPUTER_IP}:5000/tts"
        requests.post(tts_url, json={"text": tts_reply}, timeout=60)

        # 🌟🌟🌟 7.1.4 软时域闭环下发：声学硬回音切除门络 🌟🌟🌟
        # 【坑点回顾】：如果不加阻塞，TTS刚发声，上方的 stream.read 会瞬间把机器人自己的声音录进去死循环！
        print("🤫 [声学防护] 正在执行语音输出，录音总线进入安全屏蔽期...")
        
        # 强行睡眠 3.5 秒，完美避开机器人的物理发声周期
        time.sleep(3.5)

        # 【致命崩溃排雷】：绝对不能在这里调用 stream.stop_stream()！此时 stream 已在 chat() 中 close！
        # 只要执行纯时间睡眠，并让外部的 while 循环重新调用 chat() 开启干净的流即可！
        print("\n>>> 🎙️ [声学防护] 屏蔽期结束，恢复干净倾听...")
        
        # 状态重置
        frames = []
        recording = False
        silence_start = None
        start_time = time.time()

        # --- 4. 神经总线下发：向 ROS 运动小脑发送触发信号 ---
        if action_cmd["action"] == "grab" and vla_cmd_pub is not None:
            print(f"-> ⚡ 触发物理抓取管线：目标 [{action_cmd['target']}]")
            # 将 JSON 机器指令序列化为字符串，打向 ROS 总线，穿透网段下发给 NUC！
            vla_cmd_pub.publish(json.dumps(action_cmd))

        print("-> ✅ 交互闭环完成！")
        
    except requests.exceptions.RequestException as err:
        print(f"【局域网通信失败】: {err}")
    except Exception as e:
        print(f"【发生未知错误】: {e}")


# ================= 主控制流 =================
if __name__ == '__main__':
    print("\n==================================================")
    print("      Kuavo 100% 本地全离线具身交互系统已就绪       ")
    print("==================================================\n")
    
    # 🌟 必须设置 disable_signals=True，防止 ROS 劫持终端的 Ctrl+C，避免进程假死！
    rospy.init_node('vla_brain_commander', disable_signals=True)
    vla_cmd_pub = rospy.Publisher('/vla/master_command', String, queue_size=1)
    
    try:
        # 采用非阻塞存活循环
        while not rospy.is_shutdown():
            chat()  # 进入录音与 VAD 阻塞
            sentence = local_stt(INPUT_FILE)  # 进入 GPU 听觉解码
            
            # 清理标点，防止全角标点绕过空判断
            clean_sentence = sentence.replace("。", "").replace("，", "").replace("！", "").replace("？", "").strip()
            
            if clean_sentence:
                local_offline_closed_loop_chat(sentence) # 推给大模型
            else:
                print("-> 📭 录音空白或环境杂音，重新倾听。")
                
    except KeyboardInterrupt:
        print("\n[!] 收到退出指令，安全关闭音频流。")
        global_audio.terminate()
        # 显式抛出关停信号，彻底干掉 ROS 驻留节点
        rospy.signal_shutdown("User KeyboardInterrupt")
        sys.exit(0)
