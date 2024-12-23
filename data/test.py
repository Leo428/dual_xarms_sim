import numpy as np
import imageio
import cv2

# data = np.load("obses.npy", allow_pickle=True)
# data = np.load("/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_cube_1101/sim_dual_xarms_cube_handover_20241101_162055.npz", allow_pickle=True)
# data = np.load("/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_1126/real_dual_xarms_hang_20241126_232933.npz", allow_pickle=True)
data = np.load("/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_1212/real_dual_xarms_hang_20241212_163620.npz", allow_pickle=True)


frames = []
for d in data["obses"]:
    # import ipdb; ipdb.set_trace()
    images = d["images"]
    decompressed_images = {}
    for k, v in images.items():
        decompressed_image = cv2.imdecode(v, 1)
        decompressed_images[k] = np.array(decompressed_image)
    frame = []
    for k,v in decompressed_images.items():
        frame.append(v)

    frame = np.concatenate(frame, axis=1)
    frames.append(frame[..., ::-1])

imageio.mimsave("data_test_real_1212.mp4", frames, fps=60)