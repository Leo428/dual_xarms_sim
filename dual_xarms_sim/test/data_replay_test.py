import numpy as np
import h5py
from loop_rate_limiters import RateLimiter
from tqdm import tqdm

from dual_xarms_sim.fix_dual_xarms_sim import DoubleInsertDualXarmsGymEnv
from dual_xarms_sim.relative_frame import RelativeFrame
from dual_xarms_sim.oculus_intervention import OculusIntervention

if __name__ == "__main__":
    np.random.seed(42)
    env = DoubleInsertDualXarmsGymEnv(control_freq=60, render_mode="human")
    # env = DoubleInsertDualXarmsGymEnv(control_freq=60, render_mode="rgb_array")

    human_rate = RateLimiter(60, name="Human Rate", warn=False)
    bar = tqdm(total=env.MAX_STEPS, desc="Env steps")

    # dataset replay actions
    import h5py
    eps_id = 40
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0304_fixed_hdf5/episode_{eps_id}.hdf5"
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0311_fixed_hdf5/episode_{eps_id}.hdf5"
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318_fixed_hdf5/episode_{eps_id}.hdf5"
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_action_coord_frame_bugfix_hdf5/episode_{eps_id}.hdf5"
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_0226_hdf5/episode_{eps_id}.hdf5"
    dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_hdf5/episode_{eps_id}.hdf5"
    with h5py.File(dataset_path, "r") as root:
        rel_actions = root["actions"]["relative_action"][()]
        abs_actions = root["actions"]["global_action"][()]
    t = 0

    try:
        # env = OculusIntervention(env, freq=60)
        env = RelativeFrame(env) # uncomment this line to test absolute actions

        done, truncated = False, False
        obs, _ = env.reset()

        while not (done or truncated):
            # print(np.abs(rel_actions[t] - abs_actions[t]).sum())
            action = rel_actions[t] # rel actions
            # action = abs_actions[t] # abs actions

            obs, rew, done, _, info = env.step(action)
            if "intervene_action" in info:
                action = info["intervene_action"]
            if rew > 0:
                print(rew)

            t += 1
            bar.update(1)
            human_rate.sleep()

        env.close()
    except KeyboardInterrupt:
        env.close()
