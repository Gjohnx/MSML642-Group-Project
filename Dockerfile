FROM ros:jazzy-ros-base-noble AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
        libgl1-mesa-dri libegl-mesa0 \
        python3-numpy python3-yaml python3-pytest python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

FROM runtime AS gui-build

COPY sim/gui /tmp/chase-gui
RUN bash -c 'source /opt/ros/jazzy/setup.bash && cmake -S /tmp/chase-gui -B /tmp/chase-build -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/chase-build -j2'

FROM runtime AS viewer

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xvfb x11-utils x11vnc novnc websockify libqt5svg5 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gui-build /tmp/chase-build/libChaseButton.so /opt/msml642/gui/libChaseButton.so
ENV GZ_GUI_PLUGIN_PATH=/opt/msml642/gui
