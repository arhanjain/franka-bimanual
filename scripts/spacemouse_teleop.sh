#!/usr/bin/env bash
set -euo pipefail

# Env-frame SpaceMouse teleop for the minimal EnvFrameFranka robot.
#
# Commands absolute EE poses in the sim env frame (world-aligned, shared origin).
# The launcher seeds each SpaceMouse from the robot's env-frame EE on startup so
# there is no jump. Translation signs are tuned for the env frame (-1,1,1);
# rotation signs may still need tuning (--rotation-signs).
#
# Arms: defaults to both. Pass --arms r (or --arms l) to drive a single arm with
# a single SpaceMouse — only that arm is connected/commanded. Examples:
#   ./spacemouse_teleop.sh --arms r          # right arm only, /dev/hidraw3
#   ./spacemouse_teleop.sh --arms l          # left arm only,  /dev/hidraw2
#
# Left SpaceMouse:  /dev/hidraw2   Right SpaceMouse: /dev/hidraw3

cd "$(dirname "$0")/.."

python scripts/spacemouse_teleop.py \
    --l-server-ip 192.168.3.11 --l-robot-ip 192.168.200.2 --l-port 18813 \
    --r-server-ip 192.168.3.10 --r-robot-ip 192.168.201.10 --r-port 18812 \
    --left-hidraw /dev/hidraw2 --right-hidraw /dev/hidraw3 \
    --fps 30 \
    "$@"
