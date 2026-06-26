#!/usr/bin/env bash
set -euo pipefail

# SIM half of the sim-vs-real drift harness. Runs the IsaacLab LBM rollout and
# mirrors each action onto the real robot served by real_robot_server.py, showing
# a live sim(top)/real(bottom) camera grid and logging EE drift to drift.npz.
#
# Start real_robot_server.sh FIRST (in the real venv), then run this with the SIM
# venv active (sim-improvement/.venv), from the sim-improvement repo root:
#
#   ./real_vs_sim_rollout.sh \
#       --policy.client LbmOpenpi --policy.host <policy_host> --policy.port 8000 \
#       --instruction "put the kiwi on the saucer" \
#       --run_folder runs/real_vs_sim --overwrite \
#       --real-host 127.0.0.1 --real-port 9001
#
# Headless (no Isaac viewport) but keep the cv2 grid:  add --headless
# Disable the grid window (drift.npz only):            add --no-window
# Right arm only (match real_robot_server.sh --arms r): add --arms r
#   (sim still simulates both arms; only the right arm's actions reach the real
#    robot, and only the right arm's drift is logged/shown.)

# Resolve sim-improvement repo root (this file lives at
# <root>/third_party/franka-bimanual/scripts/testing/).
HERE="$(cd "$(dirname "$0")" && pwd)"
SIM_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$SIM_ROOT"

python "$HERE/real_vs_sim_rollout.py" \
    --environment LBM-Scenario-ImplicitIK-Vision \
    --scene-path ./envs/lbm_configs/3_cabot_breakfast/KiwiManip.json \
    --library-dir ./envs/lbm_usd_library \
    --policy.open_loop_horizon 8 \
    "$@"
