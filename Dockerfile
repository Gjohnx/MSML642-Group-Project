FROM ros:jazzy-ros-base-noble AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
        libgl1-mesa-dri libegl-mesa0 \
        python3-numpy python3-yaml python3-pytest \
    && rm -rf /var/lib/apt/lists/*

FROM runtime AS viewer

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xvfb x11-utils x11vnc novnc websockify libqt5svg5 \
    && rm -rf /var/lib/apt/lists/*
