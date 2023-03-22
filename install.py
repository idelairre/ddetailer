import sys
from launch import is_installed, run

if not is_installed("mediapipe"):
    python = sys.executable
    run(f'"{python}" -m pip install -U mediapipe',
        desc="Installing mediapipe", errdesc="Couldn't install mediapipe")