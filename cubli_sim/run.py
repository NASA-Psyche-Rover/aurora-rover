#!/usr/bin/env python3
"""One command for any OS: opens the MuJoCo sim window and the sensor dashboard (the sim starts the dashboard).

    python run.py                       # default site
    python run.py --site north_pole     # any psyche_cubli_v2.py option works

macOS needs MuJoCo's `mjpython` for the 3D window, so there this re-runs itself through it (mjpython comes with
`pip install mujoco`, next to this Python). Windows / Linux run the sim directly.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "psyche_cubli_v2.py")


def main():
    args = sys.argv[1:]
    cmd = [sys.executable, SIM, *args]
    if sys.platform == "darwin":
        mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
        if not os.path.exists(mjpython):
            sys.exit("mjpython not found next to this Python: pip install mujoco (in the same environment)")
        # through this Python, not mjpython's own shebang: that path breaks when a folder name has a space
        cmd = [sys.executable, mjpython, SIM, *args]
    sys.exit(subprocess.call(cmd, cwd=HERE))


if __name__ == "__main__":
    main()
