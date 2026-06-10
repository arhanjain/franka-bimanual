from lerobot_robot_bimanual_franka import SingleArmFranka, SingleArmFrankaConfig


def start_controller() -> SingleArmFranka:
    config = SingleArmFrankaConfig(
        r_server_ip="192.168.3.10",
        r_robot_ip="192.168.201.10",
        r_gripper_ip="192.168.201.10",
        r_port=18812,
        use_ee_pos=True,
    )
    robot = SingleArmFranka(config)
    robot.connect()
    return robot
