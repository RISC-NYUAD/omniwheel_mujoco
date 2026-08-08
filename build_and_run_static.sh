#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="omniwheel_mujoco:humble"
CONTAINER_NAME="omniwheel_mujoco_static"

docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null
fi

DOCKER_ARGS=(
  --rm
  -it
  --name "$CONTAINER_NAME"
  --network host
  -e DISPLAY="${DISPLAY:-:0}"
  -e QT_X11_NO_MITSHM=1
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  -e CYCLONEDDS_URI=file:///workspace/omniwheel_mujoco/cyclonedds.xml
  -v /tmp/.X11-unix:/tmp/.X11-unix
  -v "$SCRIPT_DIR/simulate/config_static.yaml:/workspace/omniwheel_mujoco/simulate/config.yaml:ro"
  -v "$SCRIPT_DIR/mujoco:/workspace/omniwheel_mujoco/mujoco:ro"
  -v "$SCRIPT_DIR/cyclonedds.xml:/workspace/omniwheel_mujoco/cyclonedds.xml:ro"
)

if command -v nvidia-smi >/dev/null 2>&1; then
  DOCKER_ARGS+=(
    --gpus all
    -e NVIDIA_DRIVER_CAPABILITIES=all
  )
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME"
