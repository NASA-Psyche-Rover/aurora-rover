#!/bin/bash
# Double-click to open the 3D sim. Uses the venv in "NASA Psyche/venv".
cd "$(dirname "$0")"
VENV=".venv"                                   # the one-liner in README_v2.md makes this
[ -d "$VENV" ] || VENV="../../venv"
[ -d "$VENV" ] || VENV="$HOME/Desktop/Aayou/ASU/EPICS/NASA Psyche/venv"
source "$VENV/bin/activate"
python -c "import scipy" 2>/dev/null || pip install -q scipy
python -c "import cv2" 2>/dev/null || pip install -q opencv-python   # textures + sensor dashboard
# run the mjpython launcher through the venv python: its shebang path contains a space ("NASA Psyche"),
# which macOS cannot exec ("bad interpreter")
python "$VENV/bin/mjpython" psyche_cubli_v2.py "$@"
