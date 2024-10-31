from tqdm import tqdm
import numpy as np
import time

from dual_xarms_sim.dual_xarms_gym import DualXarmsGymEnv
from dual_xarms_sim.relative_frame import RelativeFrame
from dual_xarms_sim.oculus_intervention import OculusIntervention
from dual_xarms_sim.utils.network import get_oculus_reading

if __name__ == "__main__":
    task_name = "sim_dual_xarms_cube_handover"
    env = DualXarmsGymEnv(render_mode="human")
    env = OculusIntervention(env, freq=10)
    env = RelativeFrame(env)

    episodes_progress_bar = tqdm(range(10), desc="Episodes")
    step_progress_bar = tqdm(range(1000), desc="Steps")

    while episodes_progress_bar.n < episodes_progress_bar.total:
        try:
            oculus_data = get_oculus_reading()
            if oculus_data is None:
                print("Failed to get Oculus data")
                time.sleep(1)
                continue

            # if A button is pressed, begin an episode
            if oculus_data["left_a_button"] or oculus_data["right_a_button"]:
                print("Starting episode...")
                step_progress_bar.reset()
                obs, info = env.reset()

                done = False
                obses, actions, rews, dones, truncated, infos = [], [], [], [], [], []

                while not done:
                    action = np.zeros(env.action_space.shape)
                    obs, rew, done, _, info = env.step(action)

                    obses.append(obs)
                    actions.append(action)
                    rews.append(rew)
                    dones.append(done)
                    truncated.append(False)
                    infos.append(info)

                    step_progress_bar.update(1)

                is_save_data = input("Finished episode. Save data? (y/n): ")
                if is_save_data.lower() == "y":
                    file_name = f"{task_name}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
                    print(f"Saving data to {file_name}")
                    with open(file_name, "wb") as f:
                        np.savez(f, 
                            obses=obses, actions=actions, rews=rews, dones=dones, truncated=truncated, infos=infos,
                            allow_pickle=True
                        )
                    episodes_progress_bar.update(1)

            else:
                episodes_progress_bar.desc = "Waiting for A button press..."
                time.sleep(1)

        except KeyboardInterrupt:
            env.close()
            break
