FROM osrf/ros:noetic-desktop-full

###################  first system wide configuration ###################################
# change timezone (this is very important otherwise many ROS topic time will be strange)
# TODO: please modify it to your timezone accordingly
ENV TZ=Asia/Shanghai
# ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && apt-get install ca-certificates -y && \
    sh -c 'echo "deb https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse\ndeb https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse\ndeb https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse\ndeb https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse" > /etc/apt/sources.list' && \
    apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
    apt-get install -y dirmngr gnupg2 && \
    sh -c 'echo "deb http://mirrors.ustc.edu.cn/ros/ubuntu/ focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 &&\
    rm -rf /var/lib/apt/lists/*

# >>> Tools <<<
ENV ZSH_CUSTOM=/root/.oh-my-zsh/custom
# zsh
RUN apt-get update -y \
    && apt-get install -y git wget \
    && git clone https://github.com/ohmyzsh/ohmyzsh.git /root/.oh-my-zsh \
    && git clone https://github.com/zsh-users/zsh-autosuggestions.git $ZSH_CUSTOM/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting \
    && sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t ys \
    -p git \
    -p zsh-syntax-highlighting \
    -p zsh-autosuggestions

# Gnome terminal support
RUN apt-get update -y \
    && apt-get install -y gnome-terminal \
    dbus-x11 \
    libcanberra-gtk-module \ 
    libcanberra-gtk3-module

# ros
RUN apt-get update -y \
    && apt-get install -y wget python3 python3-yaml python3-distro\
    && apt-get install -y ros-noetic-ros-base python3-rosdep python3-rosinstall \
    python3-rosinstall-generator python3-wstool python3-catkin-tools build-essential \
    && apt-get install ros-noetic-plotjuggler* -y \
    && apt-get install -y ros-noetic-interactive-markers ros-noetic-tf ros-noetic-tf2-ros ros-noetic-urdf \
    ros-noetic-kdl-parser ros-noetic-robot-state-publisher \
    && echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc \
    && echo 'source /opt/ros/noetic/setup.zsh' >> /root/.zshrc \
    && echo "chooses:\n" > fish_install.yaml \
    && echo "- {choose: 3, desc: '一键配置:rosdep(小鱼的rosdepc,又快又好用)'}\n" >> fish_install.yaml \
    && wget http://fishros.com/install  -O fishros && /bin/bash fishros \
    && rm -rf fish_install.yaml \
    && rm -rf /var/lib/apt/lists/*  /tmp/* /var/tmp/* \
    && apt clean && apt autoclean 

##### kuavo
RUN apt-get update -y && apt-get install libeigen3-dev libboost-all-dev -y \
    && apt-get install libyaml-cpp-dev libtinyxml2-dev -y \
    && apt-get install libncurses5-dev libncursesw5-dev -y \
    && apt-get install libmodbus-dev -y \
    && apt-get install liblcm-dev libgflags-dev libgoogle-glog-dev liblmdb-dev -y \
    && apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev -y \
    # set robot version
    && echo 'export ROBOT_VERSION=34' >> /root/.bashrc \
    && echo 'export ROBOT_VERSION=34' >> /root/.zshrc

# lcm lib
RUN apt-get update -y && apt-get install build-essential libglib2.0-dev git cmake -y \
    && git clone https://github.com/lcm-proj/lcm.git \
    && cd lcm # 编译并安装  \
    && mkdir build && cd build  \
    && cmake .. \
    && make -j$(nproc) \
    && make install
RUN ln -s /usr/lib/x86_64-linux-gnu/liblcm.so  /usr/local/lib/liblcm.so
# drake
RUN apt-get update -y && apt-get install --no-install-recommends ca-certificates gnupg lsb-release wget -y \
    && wget -qO- https://drake-apt.csail.mit.edu/drake.asc | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/drake.gpg >/dev/null \
    && echo "deb [arch=amd64] https://drake-apt.csail.mit.edu/$(lsb_release -cs) $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/drake.list >/dev/null \
    && apt-get update -y \
    && apt-get install --no-install-recommends drake-dev=1.19.0-1 -y \
    && echo 'export PATH="/opt/drake/bin${PATH:+:${PATH}}"' >> /root/.zshrc \
    && echo "export PYTHONPATH=\"/opt/drake/lib/python\$(python3 -c 'import sys; print(\"{0}.{1}\".format(*sys.version_info))')/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}\"" >> /root/.zshrc \
    && echo 'export PATH="/opt/drake/bin${PATH:+:${PATH}}"' >> /root/.bashrc \
    && echo "export PYTHONPATH=\"/opt/drake/lib/python\$(python3 -c 'import sys; print(\"{0}.{1}\".format(*sys.version_info))')/site-packages\${PYTHONPATH:+:\${PYTHONPATH}}\"" >> /root/.bashrc


# casadi
RUN apt-get update -y && apt-get install gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends -y \
    && git clone https://github.com/casadi/casadi.git \
    && cd casadi \
    && git checkout 3.5.5 \
    && mkdir build && cd build \
    && cmake .. -DWITH_EXAMPLES=OFF \
    && make -j$(nproc) \
    && make install

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# ocs2
RUN apt-get update -y && apt-get install git -y \
    && apt-get install liburdfdom-dev liboctomap-dev libassimp-dev ros-noetic-rqt-multiplot ros-noetic-grid-map-rviz-plugin -y \
    # Clone pinocchio and hpp-fcl
    && git clone --recurse-submodules https://github.com/leggedrobotics/hpp-fcl.git \
    && cd hpp-fcl \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && git clone --recurse-submodules https://github.com/leggedrobotics/pinocchio.git \
    && cd pinocchio \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install
    # && cd /root/kuavo_ws \
    # && catkin init \
    # && catkin config --extend /opt/ros/noetic \
    # && catkin config -DCMAKE_BUILD_TYPE=RelWithDebInfo \

# 其余库自行编译

# mujoco
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    libglfw3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
# ENV MUJOCO_NET https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz \
    && echo 'export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin' >> /root/.bashrc \
    && echo 'export PATH=$LD_LIBRARY_PATH:$PATH' >> /root/.bashrc \
    && echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> /root/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin' >> /root/.zshrc \
    && echo 'export PATH=$LD_LIBRARY_PATH:$PATH' >> /root/.zshrc \
    && echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> /root/.zshrc

# mujoco-py
RUN apt-get update -y && apt-get install python3-pip git -y \
    && git clone https://github.com/openai/mujoco-py.git \
    && cd mujoco-py \
    && pip3 install -U 'mujoco-py<2.2,>=2.1' \
    && pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install -r requirements.dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && python3 setup.py install \
    && pip3 install mujoco \
    && pip3 install pynput

COPY config_ccache.sh /root/config_ccache.sh
RUN chmod +x /root/config_ccache.sh && zsh -c /root/config_ccache.sh && rm /root/config_ccache.sh
RUN git config --global --add safe.directory /root/kuavo_ws

WORKDIR /root/kuavo_ws
# >>> crocoddyl <<<
# RUN pip3 install --user crocoddyl \
#     && pip3 install meshcat
