#!/usr/bin/env bash

# Script for rolling out a trained policy.
# $1 is policy repo id
# $2 is duration in seconds
#
# Display:
# Run local Rerun viewer on the robot/workstation host.
# Forward the viewer ports over SSH to view on your machine.

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <policy_repo_id> <duration_seconds>"
    exit 1
fi
lerobot-rollout \
    --strategy.type=base \
    --policy.path="$1" \
    --robot.type=bimanual_franka \
    --robot.l_server_ip=192.168.3.11 \
    --robot.l_robot_ip=192.168.200.2 \
    --robot.l_gripper_ip=192.168.2.21 \
    --robot.l_port=18813 \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.2.20 \
    --robot.r_port=18812 \
    --robot.use_ee_pos=false \
    --duration=$2
