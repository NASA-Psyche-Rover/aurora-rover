#!/bin/bash
# Double-click: records the demo mission (if needed), renders it in Blender, makes an mp4 and opens it.
# From Terminal you can pass a quality and a take:  ./"Render Film.command" final recordings/take_XXXX.npz
cd "$(dirname "$0")"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
if [ ! -x "$BLENDER" ]; then
  echo "Blender not found. Install it from https://www.blender.org/download/ (or: brew install --cask blender)"
  read -n 1 -s -r -p "Press any key to close"; exit 1
fi
VENV="../../venv"
[ -d "$VENV" ] || VENV="$HOME/Desktop/Aayou/ASU/EPICS/NASA Psyche/venv"
source "$VENV/bin/activate"
python -c "import scipy" 2>/dev/null || pip install -q scipy
QUALITY="${1:-preview}"
REC="${2:-recordings/demo.npz}"
[ -f "$REC" ] || python psyche_cubli_v2.py record --out "$REC"
NAME=$(basename "$REC" .npz)
"$BLENDER" -b -P render_blender.py -- --rec "$REC" --quality "$QUALITY" --intro
python psyche_cubli_v2.py video --frames "renders/$NAME/frames" --out "renders/${NAME}_${QUALITY}.mp4"
open "renders/${NAME}_${QUALITY}.mp4"
