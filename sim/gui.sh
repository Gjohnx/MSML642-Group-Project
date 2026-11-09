#!/usr/bin/env bash
set -eo pipefail

# Keep Gazebo's standard tools and entity tree while changing the opening view.
config="$(ros2 pkg prefix gz_sim_vendor)/opt/gz_sim_vendor/share/gz/gz-sim8/gui/gui.config"
preview="$(mktemp)"
trap 'rm -f "$preview"' EXIT
sed -e 's@<camera_pose>.*</camera_pose>@<camera_pose>-9 -10.5 12.75 0 0.74 0.86</camera_pose>@' \
    -e 's@<width>1000</width>@<width>1280</width>@' \
    -e 's@<height>845</height>@<height>850</height>@' "$config" > "$preview"

gz sim -g -v 3 --gui-config "$preview" &
gui_pid=$!
trap 'kill -TERM "$gui_pid" 2>/dev/null || true; wait || true; exit 143' TERM INT
wait "$gui_pid"
