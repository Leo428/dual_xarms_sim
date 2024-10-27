import numpy as np
import imageio

data = np.load("obses.npy", allow_pickle=True)
frames = []
for d in data:
    images = d["images"]
    # frame = [images["left/top"], images["right/top"], images["left/wrist"], images["right/wrist"]]
    frame = [images["left/top"], images["right/top"]]
    frame = np.concatenate(frame, axis=1)
    frames.append(frame)

imageio.mimsave("obses_2.mp4", frames, fps=20)
