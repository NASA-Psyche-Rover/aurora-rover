#!/usr/bin/env bash
# One-command AURORA dashboard (OAK-D Lite + Livox Mid-360 + ROS 2 Humble).
# Starts the sensor drivers and the LiDAR overlay (via perception/scripts/run_lidar_overlay.sh),
# then the dashboard server, then opens it fullscreen on this machine's screen.
#
#   dashboard/run_dashboard.sh              # kiosk fullscreen; Alt+F4 closes the browser, Ctrl+C stops everything
#   KIOSK=0 dashboard/run_dashboard.sh      # normal browser window
#   NO_BROWSER=1 KIOSK=0 dashboard/run_dashboard.sh   # server only (open http://localhost:8080 yourself)
#   DASHBOARD_ARGS="--host 0.0.0.0" dashboard/run_dashboard.sh   # also reachable from other machines on the network
DASHBOARD=1 exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/perception/scripts/run_lidar_overlay.sh" "$@"
