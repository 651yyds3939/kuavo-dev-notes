#!/bin/bash

# ==============================================================================
# 具身智能大小脑资产一键全自动迁移与反向软链接重组脚本 (kuavo_all 聚合路径版)
# ==============================================================================

# 核心绝对路径精准对齐
REPO_BASE="$HOME/Notes/kuavo-dev-notes"
LOW_WS="$HOME/kuavo_all/kuavo-ros-opensource"   # 🟢 已校准至 kuavo_all 子路径
UP_WS="$HOME/kuavo_all/kuavo_ros_application"     # 🟢 已校准至 kuavo_all 子路径

# 颜色高亮定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}[开始执行] 具身智能工作空间资产自动化盘点与无损平移...${NC}"

# ------------------------------------------------------------------------------
# 函数 1：处理【整包/整目录】自建的迁移与链接
# ------------------------------------------------------------------------------
move_and_link_dir() {
    local src_dir="$1"  # 物理机原始真实路径
    local dest_dir="$2" # 仓库目标影子路径
    
    if [ -d "$src_dir" ] && [ ! -L "$src_dir" ]; then
        echo -e "${YELLOW}[整包搬运] 正在平移自建目录: $(basename $src_dir)...${NC}"
        # 创建仓库目标上级目录
        mkdir -p "$(dirname "$dest_dir")"
        # 物理剪切平移
        mv "$src_dir" "$dest_dir"
        # 甩回反向软链接
        ln -s "$dest_dir" "$src_dir"
        echo -e "${GREEN}[成功] 整包 ${src_dir} -> ${dest_dir} 链条闭环建立！${NC}"
    elif [ -L "$src_dir" ]; then
        echo -e "[跳过] $src_dir 已经是软链接，无需重复处理。"
    else
        echo -e "[警告] 未找到原始目录: $src_dir ，请确认路径是否正确。"
    fi
}

# ------------------------------------------------------------------------------
# 函数 2：处理官方包内【单个魔改文件】的劫持与链接
# ------------------------------------------------------------------------------
move_and_link_file() {
    local src_file="$1"  # 物理机官方包内被修改的文件
    local dest_file="$2" # 仓库内对应的影子文件路径
    
    if [ -f "$src_file" ] && [ ! -L "$src_file" ]; then
        echo -e "${YELLOW}[单文劫持] 正在提取修改文件: $(basename $src_file)...${NC}"
        # 创建仓库影子层级
        mkdir -p "$(dirname "$dest_file")"
        # 物理平移文件
        mv "$src_file" "$dest_file"
        # 反向建立单文件链接
        ln -s "$dest_file" "$src_file"
        echo -e "${GREEN}[成功] 文件 ${src_file} 劫持成功！${NC}"
    elif [ -L "$src_file" ]; then
        echo -e "[跳过] $src_file 已经是软链接文件。"
    else
        echo -e "[提示] 路径未命中或已处理: $src_file"
    fi
}

# ==============================================================================
# 【第一阶段】下位机 kuavo-ros-opensource 资产重组
# ==============================================================================
echo -e "\n${BLUE}======== 正在处理下位机 (NUC) 资产 ========${NC}"

# 1.1 整包自建迁移
move_and_link_dir "$LOW_WS/src/demo/vla_grasp"                  "$REPO_BASE/kuavo-ros-opensource/src/demo/vla_grasp"
move_and_link_dir "$LOW_WS/src/kuavo_arm_moveit_config"        "$REPO_BASE/kuavo-ros-opensource/src/kuavo_arm_moveit_config"
move_and_link_dir "$LOW_WS/src/kuavo_arm_control"              "$REPO_BASE/kuavo-ros-opensource/src/kuavo_arm_control"
move_and_link_dir "$LOW_WS/src/demo/teleop"                     "$REPO_BASE/kuavo-ros-opensource/src/demo/teleop"

# 1.2 官方包内单文件魔改劫持
move_and_link_file "$LOW_WS/src/FAST_LIO/src/preprocess.cpp"    "$REPO_BASE/kuavo-ros-opensource/src/FAST_LIO/src/preprocess.cpp"
move_and_link_file "$LOW_WS/src/FAST_LIO/config/avia.yaml"      "$REPO_BASE/kuavo-ros-opensource/src/FAST_LIO/config/avia.yaml"
move_and_link_file "$LOW_WS/src/automatic_test/april_tag_recognition.py" "$REPO_BASE/kuavo-ros-opensource/src/automatic_test/april_tag_recognition.py"

# 资产配置文件原子级劫持
move_and_link_file "$LOW_WS/src/kuavo_assets/config/kuavo_v49/kuavo.json" "$REPO_BASE/kuavo-ros-opensource/src/kuavo_assets/config/kuavo_v49/kuavo.json"
move_and_link_file "$LOW_WS/src/kuavo_assets/models/biped_s49/urdf/biped_s49.urdf" "$REPO_BASE/kuavo-ros-opensource/src/kuavo_assets/models/biped_s49/urdf/biped_s49.urdf"
move_and_link_file "$LOW_WS/src/humanoid-control/h12pro_controller_node/config/customize_config.json" "$REPO_BASE/kuavo-ros-opensource/src/humanoid-control/h12pro_controller_node/config/customize_config.json"
move_and_link_file "$LOW_WS/src/humanoid-control/h12pro_controller_node/config/record_topics.json" "$REPO_BASE/kuavo-ros-opensource/src/humanoid-control/h12pro_controller_node/config/record_topics.json"


# ==============================================================================
# 【第二阶段】上位机 kuavo_ros_application 资产重组
# ==============================================================================
echo -e "\n${BLUE}======== 正在处理上位机 (Orin NX) 资产 ========${NC}"

# 2.1 整包自建迁移
move_and_link_dir "$UP_WS/src/kuavo_local_face_interaction"     "$REPO_BASE/kuavo_ros_application/src/kuavo_local_face_interaction"
move_and_link_dir "$UP_WS/src/kuavo_visual_follow"              "$REPO_BASE/kuavo_ros_application/src/kuavo_visual_follow"

# 2.2 官方大模型案例包内单文件魔改劫持
LARGE_MODEL_FILES=(
    "rtasr_python3_demo.py"
    "rtasr_python3_demo_grasping.py"
    "vlm_person_observer.py"
    "orin_gemini_gateway.py"
    "nuc_speaker_service.py"
)
for file in "${LARGE_MODEL_FILES[@]}"; do
    move_and_link_file "$UP_WS/src/kuavo_large_model/$file" "$REPO_BASE/kuavo_ros_application/src/kuavo_large_model/$file"
done

# 2.3 官方 YOLO 脚本及 Launch 补丁
move_and_link_file "$UP_WS/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros.py" "$REPO_BASE/kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros.py"
move_and_link_file "$UP_WS/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py" "$REPO_BASE/kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py"
move_and_link_file "$UP_WS/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_vision_with_observer_trigger.py" "$REPO_BASE/kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_vision_with_observer_trigger.py"
move_and_link_file "$UP_WS/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/models/yolov8n-seg.pt" "$REPO_BASE/kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/models/yolov8n-seg.pt"

move_and_link_file "$UP_WS/src/omni_robot/mini_omni_inference.py" "$REPO_BASE/kuavo_ros_application/src/omni_robot/mini_omni_inference.py"
move_and_link_file "$UP_WS/src/dynamic_biped/launch/orbbec_sensor_robot_enable.launch" "$REPO_BASE/kuavo_ros_application/src/dynamic_biped/launch/orbbec_sensor_robot_enable.launch"
move_and_link_file "$UP_WS/src/dynamic_biped/launch/orbbec_sensor_only_enable.launch" "$REPO_BASE/kuavo_ros_application/src/dynamic_biped/launch/orbbec_sensor_only_enable.launch"

echo -e "\n${GREEN}======== ✨ 恭喜老哥！kuavo_all 专属资产清洗重组圆满成功！ ✨ ========${NC}"
echo -e "${GREEN}代码实体已安全锁死在主目录下的 ~/Notes 仓库中，真实工作空间链接已无损刷新！${NC}"