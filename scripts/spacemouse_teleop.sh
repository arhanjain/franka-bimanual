#!/usr/bin/env bash
set -euo pipefail

# Env-frame SpaceMouse teleop for the minimal EnvFrameFranka robot.
#
# Commands absolute EE poses in the sim env frame (world-aligned, shared origin).
# The launcher seeds each SpaceMouse from the robot's env-frame EE on startup so
# there is no jump. Mapping/scale tuning lives in _ENV_TUNING in the .py.
#
# Arms: defaults to both. Pass --arms r (or --arms l) to drive a single arm with
# a single SpaceMouse — only that arm is connected/commanded. Examples:
#   ./spacemouse_teleop.sh --arms r          # right arm only, /dev/hidraw3
#   ./spacemouse_teleop.sh --arms l          # left arm only,  /dev/hidraw2
#
# Camera grid is ON by default (both scene cams + one wrist per active arm);
# pass --no-cameras to disable it. (tyro flags: --cameras / --no-cameras.)
#
# --grippers references and drives the WSG grippers (one per active arm) from the
# SpaceMouse buttons: left button = close, right button = open. Off by default.
#   ./spacemouse_teleop.sh --grippers            # both arms + grippers
#   ./spacemouse_teleop.sh --arms r --grippers   # right arm + its gripper only
#
# Left SpaceMouse:  /dev/hidraw2   Right SpaceMouse: /dev/hidraw3

cd "$(dirname "$0")/.."

python scripts/spacemouse_teleop.py \
    --l-server-ip 192.168.3.10 --l-robot-ip 192.168.201.10 --l-port 18812 \
    --r-server-ip 192.168.3.11 --r-robot-ip 192.168.200.2 --r-port 18813 \
    --left-hidraw /dev/hidraw3 --right-hidraw /dev/hidraw2 \
    --fps 15 \
    "$@"
