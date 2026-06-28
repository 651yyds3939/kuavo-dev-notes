#!/bin/bash
xhost +

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
mkdir -p "$PARENT_DIR/.ccache"

CONTAINER_NAME="kuavo_rl_container"
IMAGE_NAME="kuavo_rl_img:0.1"

if [[ $(docker ps -aq -f ancestor=${IMAGE_NAME}) ]]; then
    echo "Container based on image '${IMAGE_NAME}' is already running."
    if [[ $(docker ps -aq -f status=exited -f name=${CONTAINER_NAME}) ]]; then
        echo "Restarting exited container '$CONTAINER_NAME' ..."
        docker start $CONTAINER_NAME
    fi
    echo "Exec into container '$CONTAINER_NAME' ..."
    docker exec -it $CONTAINER_NAME zsh
else
    docker run -it --rm --net host --gpus all \
        --name $CONTAINER_NAME \
        --privileged \
        -v /dev:/dev \
        -v "${HOME}/.ros:/root/.ros" \
        -v "$PARENT_DIR/.ccache:/root/.ccache" \
        -v "$PARENT_DIR:/root/kuavo_ws" \
        -v "${HOME}/.config/lejuconfig:/root/.config/lejuconfig" \
        --group-add=dialout \
        --ulimit rtprio=99 \
        --cap-add=sys_nice \
        -e DISPLAY=$DISPLAY \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        ${IMAGE_NAME} \
        zsh
fi
