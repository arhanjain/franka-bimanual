#!/usr/bin/env bash
set -euo pipefail

# REAL half of the sim-vs-real drift harness. Owns the real bimanual
# EnvFrameFranka (+ cameras + WSG grippers) and serves it over a websocket so the
# sim rollout (real_vs_sim_rollout.py, sim venv) can mirror actions onto it.
#
# Run with the REAL venv active (third_party/franka-bimanual/.venv):
#   source ~/franka_ws/.venv/bin/activate   # or wherever the real venv lives
#   ./real_robot_server.sh --port 9001
#
# Pose-only (no grippers/cameras) for a quick check:
#   ./real_robot_server.sh --no-grippers --no-cameras
#
# Right arm only (luigi; e.g. while the left arm is down) -- skips the left
# wrist camera and never contacts mario. Pass the SAME --arms to the sim client.
#   ./real_robot_server.sh --arms r
#
# CONFIRM the --cam-* map in the .py against the physical mounting before
# trusting the bottom row of the grid.

cd "$(dirname "$0")/../.."

python scripts/testing/real_robot_server.py \
    --l-server-ip 192.168.3.10 --l-robot-ip 192.168.201.10 --l-port 18812 \
    --r-server-ip 192.168.3.11 --r-robot-ip 192.168.200.2 --r-port 18813 \
    --port 9001 \
    "$@"
