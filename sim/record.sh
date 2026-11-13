#!/usr/bin/env bash
# Record the running Gazebo GUI from viewer's virtual X display.
set -eo pipefail
output="${1:-/project/experiments/gazebo-demo.mp4}"
seconds="${2:-90}"
mkdir -p "$(dirname "$output")"
exec ffmpeg -y -f x11grab -video_size 1280x850 -framerate 20 -i :99.0   -t "$seconds" -c:v libx264 -preset ultrafast -crf 22 -pix_fmt yuv420p   -movflags +faststart "$output"
