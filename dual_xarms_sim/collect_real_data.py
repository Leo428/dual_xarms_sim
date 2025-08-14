from tqdm import tqdm
import numpy as np
import time
import cv2

from dual_xarms_sim.rmp_env import RMPDualXArmsEnv
from dual_xarms_sim.relative_frame import RelativeFrame, WristRelativeTo
from dual_xarms_sim.oculus_intervention import OculusIntervention
from dual_xarms_sim.utils.network import get_oculus_reading

if __name__ == "__main__":
    task_name = "real_dual_xarms_hang"
    env = RMPDualXArmsEnv(control_freq=60)
    env = OculusIntervention(env, freq=60)
    env = RelativeFrame(env)
    env = WristRelativeTo(env)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # tried as low as 20, seems fine

    episodes_progress_bar = tqdm(range(60), desc="Episodes")
    step_progress_bar = tqdm(range(60 * 60 * 3), desc="Steps")

    while episodes_progress_bar.n < episodes_progress_bar.total:
        try:
            oculus_data = get_oculus_reading(timeout=0.01)
            if oculus_data is None:
                print("Failed to get Oculus data")
                # time.sleep(1)
                continue

            # if A button is pressed, begin an episode
            if oculus_data["left_a_button"] or oculus_data["right_a_button"]:
                print("Starting episode...")
                step_progress_bar.reset()
                obs, info = env.reset()

                truncated = False
                done = False
                obses, actions, rews, dones, truncateds, infos = [], [], [], [], [], []

                while not truncated:
                    action = env.action_space.sample() * 0
                    obs, rew, done, truncated, info = env.step(action)
                    if "intervene_action" in info:
                        action = info["intervene_action"]

                    for name, img in obs["images"].items():
                        result, encoded_image = cv2.imencode('.jpg', img, encode_param)
                        obs["images"][name] = encoded_image

                    obses.append(obs)
                    actions.append(action)
                    rews.append(rew)
                    dones.append(done)
                    truncateds.append(truncated)
                    infos.append(info)

                    step_progress_bar.update(1)
                    if done:
                        step_progress_bar.desc = "Task Completed"

                obs, info = env.reset()
                is_save_data = input("Finished episode. Save data? (y/n): ")
                if "y" in is_save_data.lower():
                    file_name = f"{task_name}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
                    file_name = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round0_zheyuan_0814_hdf5/" + file_name
                    print(f"Saving data to {file_name}")
                    with open(file_name, "wb") as f:
                        np.savez(f, 
                            obses=obses, actions=actions, rews=rews, dones=dones, truncateds=truncateds, infos=infos,
                            allow_pickle=True
                        )
                    episodes_progress_bar.update(1)
                else:
                    print("Data Discarded!!!!")

            else:
                episodes_progress_bar.desc = "Waiting for A button press..."
                time.sleep(0.01)

        except KeyboardInterrupt:
            env.close()
            break
        except Exception as e:
            print(e)
            env.close()
            break
