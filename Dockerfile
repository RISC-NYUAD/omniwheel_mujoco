FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///workspace/omniwheel_mujoco/cyclonedds.xml

RUN apt update && apt upgrade -y

RUN apt-get update && apt install -y \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-robot-state-publisher \
    python3-colcon-common-extensions \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libglfw3-dev \
    libxkbcommon-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    libyaml-cpp-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/omniwheel_mujoco

COPY dependencies/mujoco dependencies/mujoco
COPY mujoco mujoco

WORKDIR /workspace/omniwheel_mujoco/dependencies/mujoco/build
RUN cmake .. -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_EXAMPLES=OFF \
    && make -j$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
    && make install \
    && ldconfig

WORKDIR /workspace/omniwheel_mujoco
COPY simulate simulate
COPY cyclonedds.xml cyclonedds.xml

WORKDIR /workspace/ros2_ws
COPY src src

SHELL ["/bin/bash", "-c"]

RUN source /opt/ros/humble/setup.bash \
    && colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))

CMD source /opt/ros/humble/setup.bash \
    && source /workspace/ros2_ws/install/setup.bash \
    && ros2 launch omniwheel_mujoco_sim sim.launch.py
