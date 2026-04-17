#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_NAME="cudaq"
IMAGE_NAME="nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.13.0"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  docker pull "$IMAGE_NAME"
fi

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  # Container exists
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    # Container is already running
    docker attach "$CONTAINER_NAME"
  else
    # Container exists but is stopped
    docker start -i "$CONTAINER_NAME"
  fi
else
  # Container does not exist
  docker run -it --rm --name "$CONTAINER_NAME" \
    --mount type=bind,source="$SCRIPT_DIR",target=/home/"$CONTAINER_NAME"/"$(basename "$SCRIPT_DIR")" \
    -w /home/"$CONTAINER_NAME"/"$(basename "$SCRIPT_DIR")" \
    -h cudaq \
    "$IMAGE_NAME"
fi
