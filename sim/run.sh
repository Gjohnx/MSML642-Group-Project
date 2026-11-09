#!/usr/bin/env bash
set -eo pipefail

gz sim -s -r --headless-rendering -v 3 room.sdf &
sim_pid=$!
trap 'kill -TERM "$sim_pid" "$bridge_pid" 2>/dev/null || true; wait || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/project/sim/bridge.yaml &
bridge_pid=$!
wait -n "$sim_pid" "$bridge_pid"
