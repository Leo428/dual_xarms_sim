import numpy as np
import imageio
import cv2
import h5py
from tqdm import tqdm

for eps_id in tqdm(range(20)):
    dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_robyn_0228_hdf5/episode_{eps_id}.hdf5"
    with h5py.File(dataset_path, "r") as root:
        frames = []
        decompressed_images = {}
        for cam_name in ["obses/images/left/top", "obses/images/right/top", "obses/images/left/wrist", "obses/images/right/wrist"]:
            compressed_images = root[cam_name][()]
            decompressed_frames = []
            for compressed_img in compressed_images:
                decompressed_image = cv2.imdecode(compressed_img, 1)
                decompressed_frames.append(np.array(decompressed_image))
            decompressed_images[cam_name] = decompressed_frames

        horizon = len(decompressed_images["obses/images/left/top"])
        for idx in range(horizon):
            frame = [
                decompressed_images["obses/images/left/top"][idx],
                decompressed_images["obses/images/right/top"][idx],
                decompressed_images["obses/images/left/wrist"][idx],
                decompressed_images["obses/images/right/wrist"][idx],
            ]
            frame = np.concatenate(frame, axis=1)
            # no need to flip colors for sim data
            frames.append(frame)
            # frames.append(frame[..., ::-1])

        imageio.mimsave(f"sim_double_insert_robyn_hdf5_ep{eps_id}.mp4", frames, fps=60)