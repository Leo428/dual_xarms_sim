import numpy as np
import imageio

# data = np.load("obses.npy", allow_pickle=True)
data = np.load("/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_cube_1101/sim_dual_xarms_cube_handover_20241101_162055.npz", allow_pickle=True)

frames = []
for d in data["obses"]:
    images = d["images"]
    frame = [images["left/top"], images["right/top"], images["left/wrist"], images["right/wrist"]]
    # frame = [images["left/top"], images["right/top"]]
    frame = np.concatenate(frame, axis=1)
    frames.append(frame)

imageio.mimsave("data_test.mp4", frames, fps=20)