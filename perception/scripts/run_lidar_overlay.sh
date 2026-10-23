#!/usr/bin/env bash
# One-command LiDAR -> OAK-D overlay (ROS 2 Humble, Livox Mid-360 + OAK-D-Lite).
# Starts missing drivers, waits for topics, launches the overlay, opens rqt_image_view.
# Ctrl+C stops everything this script started.
# (no set -u: ROS setup scripts reference unset vars)

PERCEPTION="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo clone location does not matter
export PSYCHE_OBJDETEC_ROOT="$PERCEPTION"                  # so sensor_bridges can import src.camera_stream
WS="$PERCEPTION/ros2_ws"
LIVOX_WS="$HOME/ros2_ws"
PKG_SRC="$WS/src/aurora_lidar_overlay"
INTR="$PKG_SRC/config/calibration/intrinsics.yaml"
EXTR="$PKG_SRC/config/calibration/extrinsics.yaml"   # source-tree copy so live-tuned saves persist
IMG=/oak/rgb/image_raw
INFO=/oak/rgb/camera_info
LIDAR=/livox/lidar
OUT=/aurora/lidar_overlay/image
LOGDIR="$(mktemp -d /tmp/lidar_overlay.XXXXXX)"
PIDS=()

cleanup() {
  trap - EXIT INT TERM
  echo "[run] stopping processes started by this script..."
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -INT -- "-$p" 2>/dev/null || kill -INT "$p" 2>/dev/null; done
  sleep 2
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -KILL -- "-$p" 2>/dev/null; done
}
trap cleanup EXIT INT TERM

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash || { echo "ERROR: cannot source /opt/ros/humble/setup.bash"; exit 1; }

# Build only if install is missing or older than the package source.
MARK="$WS/install/.overlay_built_stamp"   # touched after each successful build
if [ ! -f "$MARK" ] || [ ! -d "$WS/install/aurora_lidar_overlay" ] || [ -n "$(find "$PKG_SRC" -type f -newer "$MARK" -not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' -print -quit)" ]; then
  echo "[run] building aurora_lidar_overlay + sensor_bridges..."
  (cd "$WS" && colcon build --packages-select aurora_lidar_overlay sensor_bridges --symlink-install) || { echo "ERROR: colcon build failed"; exit 1; }
  touch "$MARK"
fi
# shellcheck disable=SC1091
source "$WS/install/setup.bash" || { echo "ERROR: cannot source $WS/install/setup.bash"; exit 1; }

has_pub() { ros2 topic info "$1" 2>/dev/null | grep -q "Publisher count: [1-9]"; }
# start <name> <cmd...> : run in its own process group, log to $LOGDIR/<name>.log
start() { local n=$1; shift; setsid bash -c "$*" >"$LOGDIR/$n.log" 2>&1 & PIDS+=("$!"); }

if has_pub "$IMG"; then echo "[run] camera already publishing, skipping OAK-D driver"; else
  echo "[run] starting OAK-D bridge"
  start oak "source /opt/ros/humble/setup.bash; source $WS/install/setup.bash; exec ros2 run sensor_bridges oak_image_publisher --ros-args -p intrinsics_yaml:=$INTR"
fi

if has_pub "$LIDAR"; then echo "[run] LiDAR already publishing, skipping LiDAR driver"; else
  echo "[run] starting Livox driver (PointCloud2, xfer_format=0)"
  D="$LIVOX_WS/install/livox_ros_driver2/share/livox_ros_driver2"
  [ -f "$D/launch_ROS2/msg_MID360_launch.py" ] || { echo "ERROR: livox_ros_driver2 not found in $LIVOX_WS"; exit 1; }
  # The stock launch file hardcodes xfer_format=1 (CustomMsg, for FAST-LIO). Use a temp copy with 0.
  sed -e 's/^xfer_format   = 1/xfer_format   = 0/' -e "s#^cur_config_path.*#cur_config_path = '$D/config'#" \
    "$D/launch_ROS2/msg_MID360_launch.py" > "$LOGDIR/livox_pc2_launch.py"
  start livox "source /opt/ros/humble/setup.bash; source $LIVOX_WS/install/setup.bash; exec ros2 launch $LOGDIR/livox_pc2_launch.py"
fi

# Wait until each topic delivers a message.
wait_topic() {
  local t=$1 label=$2 i
  for i in $(seq 1 20); do
    timeout 4 ros2 topic echo --once --no-arr --qos-profile sensor_data "$t" >/dev/null 2>&1 && { echo "[run] $label live: $t"; return 0; }
  done
  echo "ERROR: $label topic $t is not publishing (logs: $LOGDIR)"; return 1
}
wait_topic "$IMG" "camera image" || exit 1
wait_topic "$INFO" "camera_info" || exit 1
wait_topic "$LIDAR" "LiDAR" || exit 1

echo "[run] launching overlay (pinned parameters)"
start overlay "source /opt/ros/humble/setup.bash; source $WS/install/setup.bash; exec ros2 launch aurora_lidar_overlay overlay.launch.py \
  image_topic:=$IMG camera_info_topic:=$INFO lidar_topic:=$LIDAR output_topic:=$OUT \
  extrinsic_yaml:=$EXTR intrinsic_yaml:=$INTR invert_extrinsic:=false use_tf:=false color_by:=depth skip:=1"
sleep 5
WAIT_PID="${PIDS[-1]}"
if [ "${DASHBOARD:-0}" = 1 ]; then
  # Web dashboard (dashboard/run_dashboard.sh sets this): serves the overlay, detection and point-cloud panels.
  DASH="$PERCEPTION/../dashboard/server.py"
  PORT="${DASHBOARD_PORT:-8080}"
  echo "[run] starting dashboard on http://127.0.0.1:$PORT"
  start dashboard "source /opt/ros/humble/setup.bash; source $WS/install/setup.bash; exec python3 $DASH --port $PORT --overlay-topic $OUT ${DASHBOARD_ARGS:-}"
  WAIT_PID="${PIDS[-1]}"
  for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/api/metrics" >/dev/null 2>&1 && break; sleep 1; done
  if [ "${KIOSK:-1}" = 1 ]; then   # KIOSK=0 opens a normal browser window instead
    start browser "exec chromium-browser --kiosk --noerrdialogs --disable-infobars --user-data-dir=$HOME/snap/chromium/common/aurora-dashboard http://127.0.0.1:$PORT"
  elif [ "${NO_BROWSER:-0}" != 1 ]; then
    start browser "exec chromium-browser --user-data-dir=$HOME/snap/chromium/common/aurora-dashboard http://127.0.0.1:$PORT"
  fi
elif [ "${NO_VIEWER:-0}" != 1 ]; then
  start viewer "exec ros2 run rqt_image_view rqt_image_view $OUT"
fi
echo "[run] running. Logs in $LOGDIR. Ctrl+C to stop."
wait "$WAIT_PID"
