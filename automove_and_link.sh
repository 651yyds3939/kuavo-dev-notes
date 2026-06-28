#!/bin/bash
# ==============================================================================
# kuavo-dev-notes 工作空间迁移、训练产物归档与软链接维护脚本
# ==============================================================================
#
# 【给 GitHub 访客 / clone 本仓库的人】
#   这是作者本机用的目录整理工具，默认路径为 ~/Notes/kuavo-dev-notes 与 ~/kuavo_all。
#   阅读笔记、魔改代码、跑官方流程时，通常不需要执行本脚本。
#   只有当你要复刻作者同一套「本仓库 + 终端软链接 + log 外置」目录布局时再使用。
#   说明文档：同目录 SYMLINK_LAYOUT.md
#
# 用法:
#   ./automove_and_link.sh help              显示帮助
#   ./automove_and_link.sh status            查看拓扑与体积
#   ./automove_and_link.sh verify            检查软链接是否断裂
#   ./automove_and_link.sh shadow            魔改影子目录/文件: kuavo_all → Notes
#   ./automove_and_link.sh link-repos        整仓入口: kuavo_all → Notes 子目录
#   ./automove_and_link.sh archive-logs      训练 logs 肉身 → _training_logs
#   ./automove_and_link.sh archive-videos    kuavo-robot-train/videos → _training_logs
#   ./automove_and_link.sh cleanup-train     删除 train 仓可再生产物
#   ./automove_and_link.sh git-untrack-logs  rl/wm 子仓停止跟踪 logs
#
# 重要: 各子命令幂等设计——已链接/已归档则跳过。不要随便改 REPO_BASE/ARCHIVE 后只挪一层。
# 拓扑说明见同目录 SYMLINK_LAYOUT.md
# ==============================================================================

set -euo pipefail

REPO_BASE="${REPO_BASE:-$HOME/Notes/kuavo-dev-notes}"
KUAVO_ALL="${KUAVO_ALL:-$HOME/kuavo_all}"
ARCHIVE="${ARCHIVE:-$KUAVO_ALL/_training_logs}"
LOW_WS="$KUAVO_ALL/kuavo-ros-opensource"
UP_WS="$KUAVO_ALL/kuavo_ros_application"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC} $*" >&2; }

usage() {
    echo "kuavo-dev-notes 工作空间维护脚本（作者本机专用，克隆者可忽略）"
    echo ""
    sed -n '12,24p' "$0" | sed 's/^# \?//'
    echo ""
    echo "说明文档: $REPO_BASE/SYMLINK_LAYOUT.md"
}

# ------------------------------------------------------------------------------
# 通用: 整目录迁移 + 反向软链接 (源码影子)
# ------------------------------------------------------------------------------
move_and_link_dir() {
    local src_dir="$1"
    local dest_dir="$2"

    if [ -d "$src_dir" ] && [ ! -L "$src_dir" ]; then
        warn "整包搬运: $(basename "$src_dir")"
        mkdir -p "$(dirname "$dest_dir")"
        mv "$src_dir" "$dest_dir"
        ln -s "$dest_dir" "$src_dir"
        ok "整包 ${src_dir} → ${dest_dir}"
    elif [ -L "$src_dir" ]; then
        info "跳过(已是软链接): $src_dir → $(readlink "$src_dir")"
    else
        warn "未找到实体目录: $src_dir"
    fi
}

move_and_link_file() {
    local src_file="$1"
    local dest_file="$2"

    if [ -f "$src_file" ] && [ ! -L "$src_file" ]; then
        warn "单文件劫持: $(basename "$src_file")"
        mkdir -p "$(dirname "$dest_file")"
        mv "$src_file" "$dest_file"
        ln -s "$dest_file" "$src_file"
        ok "文件 ${src_file} → ${dest_file}"
    elif [ -L "$src_file" ]; then
        info "跳过(已是软链接): $src_file"
    else
        info "未命中: $src_file"
    fi
}

# ------------------------------------------------------------------------------
# 通用: 训练产物归档 (logs/videos 肉身 → ARCHIVE，仓内留软链接)
# ------------------------------------------------------------------------------
archive_dir_to_training_logs() {
    local repo_rel_path="$1"   # 相对 REPO_BASE，如 leju_robot_rl/logs
    local archive_subpath="$2" # 相对 ARCHIVE，如 leju_robot_rl/logs

    local src="$REPO_BASE/$repo_rel_path"
    local dest="$ARCHIVE/$archive_subpath"

    if [ -L "$src" ]; then
        info "跳过(已是软链接): $src → $(readlink "$src")"
        return 0
    fi
    if [ ! -d "$src" ]; then
        warn "不存在，跳过: $src"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    mv "$src" "$dest"
    ln -s "$dest" "$src"
    ok "归档 ${src} → ${dest}，并已挂回软链接"
}

ensure_repo_symlink() {
    local name="$1"       # kuavo_all 下的名字
    local repo_subdir="$2" # Notes 下子目录

    local link="$KUAVO_ALL/$name"
    local target="$REPO_BASE/$repo_subdir"

    if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
        info "跳过(链接正确): $link"
        return 0
    fi
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        err "$link 是实体目录，请手动处理后重试"
        return 1
    fi
    ln -sfn "$target" "$link"
    ok "$link → $target"
}

check_link() {
    local path="$1"
    if [ -L "$path" ]; then
        local target
        target="$(readlink "$path")"
        if [ -e "$path" ]; then
            ok "链接有效: $path → $target"
        else
            err "链接断裂: $path → $target"
            return 1
        fi
    elif [ -d "$path" ]; then
        warn "仍是实体目录(未归档?): $path ($(du -sh "$path" | cut -f1))"
    else
        warn "不存在: $path"
    fi
}

# ==============================================================================
# 子命令
# ==============================================================================
cmd_shadow() {
    info "阶段 shadow: 下位机/上位机魔改影子 → Notes"
    echo

    info "======== 下位机 kuavo-ros-opensource ========"
    move_and_link_dir "$LOW_WS/src/demo/vla_grasp"           "$REPO_BASE/kuavo-ros-opensource/src/demo/vla_grasp"
    move_and_link_dir "$LOW_WS/src/kuavo_arm_moveit_config"  "$REPO_BASE/kuavo-ros-opensource/src/kuavo_arm_moveit_config"
    move_and_link_dir "$LOW_WS/src/kuavo_arm_control"         "$REPO_BASE/kuavo-ros-opensource/src/kuavo_arm_control"
    move_and_link_dir "$LOW_WS/src/demo/teleop"              "$REPO_BASE/kuavo-ros-opensource/src/demo/teleop"

    move_and_link_file "$LOW_WS/src/FAST_LIO/src/preprocess.cpp" \
        "$REPO_BASE/kuavo-ros-opensource/src/FAST_LIO/src/preprocess.cpp"
    move_and_link_file "$LOW_WS/src/FAST_LIO/config/avia.yaml" \
        "$REPO_BASE/kuavo-ros-opensource/src/FAST_LIO/config/avia.yaml"
    move_and_link_file "$LOW_WS/src/automatic_test/april_tag_recognition.py" \
        "$REPO_BASE/kuavo-ros-opensource/src/automatic_test/april_tag_recognition.py"
    move_and_link_file "$LOW_WS/src/kuavo_assets/config/kuavo_v49/kuavo.json" \
        "$REPO_BASE/kuavo-ros-opensource/src/kuavo_assets/config/kuavo_v49/kuavo.json"
    move_and_link_file "$LOW_WS/src/kuavo_assets/models/biped_s49/urdf/biped_s49.urdf" \
        "$REPO_BASE/kuavo-ros-opensource/src/kuavo_assets/models/biped_s49/urdf/biped_s49.urdf"
    move_and_link_file "$LOW_WS/src/humanoid-control/h12pro_controller_node/config/customize_config.json" \
        "$REPO_BASE/kuavo-ros-opensource/src/humanoid-control/h12pro_controller_node/config/customize_config.json"
    move_and_link_file "$LOW_WS/src/humanoid-control/h12pro_controller_node/config/record_topics.json" \
        "$REPO_BASE/kuavo-ros-opensource/src/humanoid-control/h12pro_controller_node/config/record_topics.json"

    echo
    info "======== 上位机 kuavo_ros_application ========"
    move_and_link_dir "$UP_WS/src/kuavo_local_face_interaction" \
        "$REPO_BASE/kuavo_ros_application/src/kuavo_local_face_interaction"
    move_and_link_dir "$UP_WS/src/kuavo_visual_follow" \
        "$REPO_BASE/kuavo_ros_application/src/kuavo_visual_follow"

    local file
    for file in rtasr_python3_demo.py rtasr_python3_demo_grasping.py vlm_person_observer.py \
                orin_gemini_gateway.py nuc_speaker_service.py; do
        move_and_link_file "$UP_WS/src/kuavo_large_model/$file" \
            "$REPO_BASE/kuavo_ros_application/src/kuavo_large_model/$file"
    done

    move_and_link_file "$UP_WS/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros.py" \
        "$REPO_BASE/kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros.py"
    move_and_link_file "$UP_WS/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py" \
        "$REPO_BASE/kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py"
    move_and_link_file "$UP_WS/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_vision_with_observer_trigger.py" \
        "$REPO_BASE/kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_vision_with_observer_trigger.py"
    move_and_link_file "$UP_WS/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/models/yolov8n-seg.pt" \
        "$REPO_BASE/kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/models/yolov8n-seg.pt"
    move_and_link_file "$UP_WS/src/omni_robot/mini_omni_inference.py" \
        "$REPO_BASE/kuavo_ros_application/src/omni_robot/mini_omni_inference.py"
    move_and_link_file "$UP_WS/src/dynamic_biped/launch/orbbec_sensor_robot_enable.launch" \
        "$REPO_BASE/kuavo_ros_application/src/dynamic_biped/launch/orbbec_sensor_robot_enable.launch"
    move_and_link_file "$UP_WS/src/dynamic_biped/launch/orbbec_sensor_only_enable.launch" \
        "$REPO_BASE/kuavo_ros_application/src/dynamic_biped/launch/orbbec_sensor_only_enable.launch"

    ok "shadow 完成"
}

cmd_link_repos() {
    info "整仓入口软链接: $KUAVO_ALL → $REPO_BASE 子目录"
    ensure_repo_symlink "leju_robot_rl"       "leju_robot_rl"
    ensure_repo_symlink "leju_robot_wm"       "leju_robot_wm"
    ensure_repo_symlink "kuavo-rl-opensource" "kuavo-rl-opensource"
    ok "link-repos 完成"
}

cmd_archive_logs() {
    info "训练 logs 归档 → $ARCHIVE"
    mkdir -p "$ARCHIVE/leju_robot_rl" "$ARCHIVE/leju_robot_wm" \
             "$ARCHIVE/kuavo-rl-opensource/kuavo-robot-train" \
             "$ARCHIVE/kuavo-rl-opensource/kuavo-robot-deploy"

    archive_dir_to_training_logs "leju_robot_rl/logs" \
        "leju_robot_rl/logs"
    archive_dir_to_training_logs "leju_robot_wm/logs" \
        "leju_robot_wm/logs"
    archive_dir_to_training_logs "kuavo-rl-opensource/kuavo-robot-train/logs" \
        "kuavo-rl-opensource/kuavo-robot-train/logs"

    # deploy 编译 log 常为 root 属主；默认删除，不归档
    local deploy_logs="$REPO_BASE/kuavo-rl-opensource/kuavo-robot-deploy/logs"
    if [ -d "$deploy_logs" ] && [ ! -L "$deploy_logs" ]; then
        if [ -w "$deploy_logs" ]; then
            rm -rf "$deploy_logs"
            ok "已删除 deploy 编译 logs: $deploy_logs"
        else
            warn "deploy/logs 无写权限(可能为 root)，请手动: sudo rm -rf $deploy_logs"
        fi
    fi

    ok "archive-logs 完成"
}

cmd_archive_videos() {
    info "kuavo-robot-train/videos 归档 → $ARCHIVE"
    mkdir -p "$ARCHIVE/kuavo-rl-opensource/kuavo-robot-train"
    archive_dir_to_training_logs "kuavo-rl-opensource/kuavo-robot-train/videos" \
        "kuavo-rl-opensource/kuavo-robot-train/videos"
    ok "archive-videos 完成"
}

cmd_cleanup_train() {
    local train="$REPO_BASE/kuavo-rl-opensource/kuavo-robot-train"
    info "清理可再生产物: $train"
    rm -rf "$train/humanoid.egg-info"
    find "$train" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    rm -f "$train/play.txt"
    ok "cleanup-train 完成"
}

cmd_git_untrack_logs() {
    for repo in leju_robot_rl leju_robot_wm; do
        local dir="$REPO_BASE/$repo"
        [ -d "$dir/.git" ] || { warn "无 .git，跳过: $repo"; continue; }
        info "处理 $repo ..."
        grep -qxF 'logs/' "$dir/.gitignore" 2>/dev/null || echo 'logs/' >> "$dir/.gitignore"
        grep -qxF 'logs'  "$dir/.gitignore" 2>/dev/null || echo 'logs'  >> "$dir/.gitignore"
        ( cd "$dir" && git ls-files -z logs 2>/dev/null | xargs -0 -r git rm --cached -f )
        ( cd "$dir" && git add .gitignore )
        ok "$repo: 已从索引移除 logs（请自行 git commit）"
    done
}

cmd_verify() {
    local failed=0
    info "检查整仓入口 ..."
    check_link "$KUAVO_ALL/leju_robot_rl"       || failed=1
    check_link "$KUAVO_ALL/leju_robot_wm"       || failed=1
    check_link "$KUAVO_ALL/kuavo-rl-opensource" || failed=1

    info "检查训练产物软链接 ..."
    check_link "$REPO_BASE/leju_robot_rl/logs"                              || failed=1
    check_link "$REPO_BASE/leju_robot_wm/logs"                              || failed=1
    check_link "$REPO_BASE/kuavo-rl-opensource/kuavo-robot-train/logs"      || failed=1
    check_link "$REPO_BASE/kuavo-rl-opensource/kuavo-robot-train/videos"   || failed=1

    info "穿透测试 ..."
    ls "$KUAVO_ALL/leju_robot_rl/logs/rsl_rl/Kuavo/s49/dance" >/dev/null 2>&1 \
        && ok "RL dance runs 可读" || { err "RL dance 路径不可读"; failed=1; }

    [ "$failed" -eq 0 ] && ok "verify 全部通过" || err "verify 有问题，见 SYMLINK_LAYOUT.md"
    return "$failed"
}

cmd_status() {
    echo -e "${BLUE}======== 路径常量 ========${NC}"
    echo "REPO_BASE = $REPO_BASE"
    echo "KUAVO_ALL = $KUAVO_ALL"
    echo "ARCHIVE   = $ARCHIVE"
    echo
    echo -e "${BLUE}======== 整仓入口 ========${NC}"
    ls -la "$KUAVO_ALL/leju_robot_rl" "$KUAVO_ALL/leju_robot_wm" "$KUAVO_ALL/kuavo-rl-opensource" 2>/dev/null || true
    echo
    echo -e "${BLUE}======== 训练产物链接 ========${NC}"
    ls -la "$REPO_BASE/leju_robot_rl/logs" "$REPO_BASE/leju_robot_wm/logs" \
           "$REPO_BASE/kuavo-rl-opensource/kuavo-robot-train/logs" \
           "$REPO_BASE/kuavo-rl-opensource/kuavo-robot-train/videos" 2>/dev/null || true
    echo
    echo -e "${BLUE}======== 体积 ========${NC}"
    du -sh "$ARCHIVE" 2>/dev/null || echo "ARCHIVE 不存在"
    du -sh "$ARCHIVE"/*/* 2>/dev/null | sort -hr || true
    echo
    du -sh --exclude='.git' "$REPO_BASE" 2>/dev/null || du -sh "$REPO_BASE"
}

cmd_setup_all() {
    info "新机器/重装后推荐顺序（已做过则自动跳过）"
    cmd_link_repos
    cmd_archive_logs
    cmd_archive_videos
    cmd_cleanup_train
    cmd_verify
}

# ==============================================================================
# 入口
# ==============================================================================
CMD="${1:-help}"
case "$CMD" in
    help|-h|--help) usage ;;
    status)         cmd_status ;;
    verify)         cmd_verify ;;
    shadow)         cmd_shadow ;;
    link-repos)     cmd_link_repos ;;
    archive-logs)   cmd_archive_logs ;;
    archive-videos) cmd_archive_videos ;;
    cleanup-train)  cmd_cleanup_train ;;
    git-untrack-logs) cmd_git_untrack_logs ;;
    setup-all)      cmd_setup_all ;;
    *)
        err "未知子命令: $CMD"
        usage
        exit 1
        ;;
esac
