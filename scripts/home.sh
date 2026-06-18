#!/usr/bin/env bash

# Drive the arms to the built-in joint-space home pose via EnvFrameFranka.home()
# (a reset, not teleop). Joint space so both arms are symmetric (EE homing leaves
# the redundant elbow/wrist free).
#
# Usage: ./home.sh [l|r|lr]   (arm defaults to lr = both)
#
# The default target is the SIM reset pose (LEFT/RIGHT_PANDA_DEFAULT_JOINT_POS in
# src/sim_improvement/environments/lbm/robot.py). It is not a naive mirror -- the
# sim bakes in the asymmetric base mounting, so it looks symmetric where mirroring
# one arm does not.
#
# To override: guide BOTH arms to a pose and capture their real joints:
#   python home.py save home_pose            # --arm lr, no mirror
# (apply prefers home_poses/home_pose.json over the built-in default.)

set -euo pipefail

ARM="${1:-lr}"

python "$(dirname "$0")/home.py" apply home_pose \
    --arm="$ARM" \
    --l-server-ip=192.168.3.11 \
    --l-robot-ip=192.168.200.2 \
    --l-port=18813 \
    --r-server-ip=192.168.3.10 \
    --r-robot-ip=192.168.201.10 \
    --r-port=18812
