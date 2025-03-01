import gymnasium as gym
from tqdm import tqdm
import numpy as np

from dual_xarms_sim.fix_dual_xarms_sim import DoubleInsertDualXarmsGymEnv
from dual_xarms_sim.relative_frame import RelativeFrame
from bc_flowmatch.env_wrappers.frame_stack_wrapper import FrameStackWrapperEnv

if __name__ == '__main__':
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: FrameStackWrapperEnv(
                RelativeFrame(
                    DoubleInsertDualXarmsGymEnv(
                        render_mode='rgb_array'
                    )
                ), n_frames=1, gap=29
            ) for _ in range(10)
        ]
    )
    bar = tqdm(range(envs.get_attr("MAX_STEPS")[0]), smoothing=1)

    try:
        obses, infos = envs.reset()
        while True:
            actions = envs.action_space.sample() * 0
            obses, rewards, terminates, truncates, infos = envs.step(actions)
            bar.update(1)
            if terminates.all() or truncates.all():
                break

    except Exception as e:
        envs.close()
        raise e

    # env = DoubleInsertDualXarmsGymEnv(render_mode='rgb_array')
    # env = RelativeFrame(env)
    # env = FrameStackWrapperEnv(env, n_frames=1, gap=29)
    # obs, info = env.reset()

    # obs_space_shape = 0
    # for k,v in env.observation_space["state"].items():
    #     obs_space_shape += np.prod(v.shape)

    # obs_shape = 0
    # try:
    #     for k,v in obs["state"].items():
    #         obs_shape += np.prod(v.shape)
    # except Exception as e:
    #     print(k, v)
    #     env.close()
    #     raise e
    # assert obs_space_shape == obs_shape
    # import ipdb; ipdb.set_trace()
