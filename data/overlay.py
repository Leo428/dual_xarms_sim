import numpy as np
import imageio
import cv2
import h5py
from tqdm import tqdm
import torch
from torchvision.transforms.v2 import CenterCrop, GaussianBlur, Compose, Resize, ToDtype, RandomApply
from einops import rearrange

MAX_EPISODES = 50

BGR=False
total_size = 0
overlay_images = {
    "obses/images/right/top": np.zeros((224, 224, 3), dtype=np.float32), # H, W, C, rgba
}

for eps_id in tqdm(range(MAX_EPISODES)):
    dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_0226_hdf5/episode_{eps_id}.hdf5"
    with h5py.File(dataset_path, "r") as root:
        episode_length = root["obses/images/left/top"].shape[0]
        total_size += episode_length

print(f"Total size: {total_size} images")

for eps_id in tqdm(range(MAX_EPISODES)):
    dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_0226_hdf5/episode_{eps_id}.hdf5"
    with h5py.File(dataset_path, "r") as root:
        for cam_name in overlay_images.keys():
            compressed_images = root[cam_name][()]
            for compressed_img in tqdm(compressed_images, desc=f"Decompressing {cam_name}", leave=False):
                decompressed_image = cv2.imdecode(compressed_img, 1)
                overlay_images[cam_name] += decompressed_image.astype(np.float32)

for cam_name in overlay_images.keys():
    overlay_images[cam_name] /= total_size
    overlay_images[cam_name] = np.clip(overlay_images[cam_name], 0, 255).astype(np.uint8)
    imageio.imwrite(f"overlay_{cam_name.replace("/", "_")}.png", overlay_images[cam_name])

