#!/usr/bin/env bash
set -eo pipefail

Xvfb "$DISPLAY" -screen 0 1280x850x24 -nolisten tcp &
display_pid=$!
trap 'kill -TERM "$display_pid" "${vnc_pid:-}" "${web_pid:-}" "${gui_pid:-}" 2>/dev/null || true; wait || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for attempt in {1..50}; do
    if xdpyinfo >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
x11vnc -display "$DISPLAY" -localhost -nopw -forever -shared &
vnc_pid=$!
websockify --web /usr/share/novnc 6080 127.0.0.1:5900 &
web_pid=$!
bash gui.sh &
gui_pid=$!
wait -n "$display_pid" "$vnc_pid" "$web_pid" "$gui_pid"
