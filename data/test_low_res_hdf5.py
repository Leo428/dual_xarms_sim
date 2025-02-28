import torch
import numpy as np
import imageio
import cv2
import h5py
from torchvision.transforms.v2 import Resize, RandomCrop

dataset_path = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_jasmine_0226_hdf5/episode_11.hdf5"
resize = Resize((128, 128), antialias=False)
with h5py.File(dataset_path, "r") as root:
    frames = []
    decompressed_images = {}
    for cam_name in ["obses/images/left/top", "obses/images/right/top", "obses/images/left/wrist", "obses/images/right/wrist"]:
        compressed_images = root[cam_name][()]
        decompressed_frames = []
        for compressed_img in compressed_images:
            decompressed_image = cv2.imdecode(compressed_img, 1)
            decompressed_frames.append(np.array(decompressed_image))

        decompressed_frames = torch.from_numpy(np.stack(decompressed_frames, axis=0)).permute(0, 3, 1, 2)
        decompressed_frames = resize(decompressed_frames)
        decompressed_frames = decompressed_frames.permute(0, 2, 3, 1).numpy()
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

    imageio.mimsave("resized128_sim_double_insert_hdf5_jasmine_1.mp4", frames, fps=60)