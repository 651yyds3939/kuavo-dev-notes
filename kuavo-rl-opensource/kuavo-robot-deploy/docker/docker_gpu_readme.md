# Build docker image&container for Kuavo-MPC-WBC
## 1. Install Docker
Follow the instructions on the official Docker website to install Docker on your system. 
## 2. Build Docker Image for Kuavo-MPC-WBC
To build the dockerfile:
```bash
cd <path-to-kuavo-ros-control>
docker build -f docker/Dockerfile.GPU -t humanoid_control_img:noetic .
```
To run the docker container:
```bash
   docker run -it --rm --net host --gpus all \
        -v /dev:/dev \
        --privileged \
        --group-add=dialout \
        --ulimit rtprio=99 \
        --cap-add=sys_nice \
        -e DISPLAY=$DISPLAY \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        humanoid_control_img:noetic \
        bash
```