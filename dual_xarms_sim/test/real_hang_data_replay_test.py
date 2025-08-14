import numpy as np
import h5py
from loop_rate_limiters import RateLimiter
from tqdm import tqdm
import time

from dual_xarms_sim.rmp_env import RMPDualXArmsEnv
from dual_xarms_sim.relative_frame import RelativeFrame, WristRelativeTo
from dual_xarms_sim.oculus_intervention import OculusIntervention

if __name__ == "__main__":
    '''
    DANGEROUS: This script is for testing the real hang data replay.
    For the buggy data, PLEASE monitor closely, and stop either the script or E-STOP
    if the robot is not moving as expected.
    '''
    np.random.seed(42)
    env = RMPDualXArmsEnv(control_freq=60)
    env = OculusIntervention(env, freq=60)
    env = RelativeFrame(env)
    env = WristRelativeTo(env)

    human_rate = RateLimiter(60, name="Human Rate", warn=False)
    bar = tqdm(total=env.unwrapped.MAX_STEPS, desc="Env steps")

    # dataset replay actions
    import h5py
    eps_id = 0

    # DANGEROUS: double check which dataset you are replaying from!!!

    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0419_hdf5/episode_{eps_id}.hdf5"
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0419_fixed_hdf5/episode_{eps_id}.hdf5"
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0423_hdf5/episode_{eps_id}.hdf5"
    dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/replay_test_hdf5/episode_{eps_id}.hdf5"

    with h5py.File(dataset_path, "r") as root:
        rel_actions = root["actions"]["relative_action"][()]
        abs_actions = root["actions"]["global_action"][()]
    t = 0

    # latency_freq = 60//2

    try:
        done, truncated = False, False
        obs, _ = env.reset()

        while not (done or truncated):
            # print(np.abs(rel_actions[t] - abs_actions[t]).sum())
            action = rel_actions[t] # rel actions
            # action = abs_actions[t] # abs actions

            obs, rew, done, _, info = env.step(action)
            if "intervene_action" in info:
                action = info["intervene_action"]

            # if t % latency_freq == 0:
            #     time.sleep(1/5)

            t += 1
            bar.update(1)
            human_rate.sleep()

        env.close()
    except KeyboardInterrupt:
        env.close()
