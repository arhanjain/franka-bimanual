"""Print both arms' current joint angles + env-frame EE pose for symmetry checks.

Reads q directly from the RPyC kinematic snapshot (the env robot doesn't surface
joints in its observation). Mirror symmetry expectation for a symmetric pose:
the left and right joint vectors should match up to the per-joint sign pattern
of a left/right mirror (typically q1,q3,q5,q7 negate; q2,q4,q6 stay) -- so we
print l, r, and l+r / l-r to make the mismatch obvious.
"""

import numpy as np

from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig

np.set_printoptions(precision=4, suppress=True, sign=" ")


def main() -> None:
    robot = EnvFrameFranka(EnvFrameFrankaConfig(active_arms=("l", "r")))
    robot.connect()
    try:
        kin = robot.robot_manager.current_kinematic_state_batch(["l", "r"])
        ql, _, _, _ = kin["l"]
        qr, _, _, _ = kin["r"]
        print("l_q :", ql)
        print("r_q :", qr)
        print("l-r :", ql - qr, "  (~0 where joints track together)")
        print("l+r :", ql + qr, "  (~0 where joints mirror via sign flip)")
        env = robot.current_ee_pose_env()
        for arm in ("l", "r"):
            p, q = env[arm]
            print(f"{arm} env pos {np.round(p,4)}  quat {np.round(q,4)}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
