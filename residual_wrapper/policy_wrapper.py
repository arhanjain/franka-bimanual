from lerobot.types import RobotObservation
import env_wrapper

def process_obs(obs: RobotObservation) -> RobotObservation:
    pass

def process_chunk(chunk):
    pass

def main() -> None:
    # start controller
    controller = env_wrapper.start_controller()

    # instantiate base policy
    # instantiate residual

    # get obs
    obs = controller.get_observation()
    # extract depth and remove it from original obs
    
    # pass to base policy
    # chunk = base_policy.infer(obs)

    # process
    obs = process_obs(obs)
    # res_chunk = process_chunk(chunk)
    
    # pass to residual policy
    # res = residual.infer(res_chunk + obs)

    # set controller residual
    controller.cache_delta(dpos, drot)
    # execute on controller
    # controller.send_action(chunk)


if __name__ == "__main__":
    main()