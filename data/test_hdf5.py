import numpy as np
import imageio
import cv2
import h5py

dataset_path = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_1126_hdf5/episode_21.hdf5"
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
        frames.append(frame[..., ::-1])

    imageio.mimsave("data_test_real_hdf5.mp4", frames, fps=50)