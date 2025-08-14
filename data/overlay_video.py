import numpy as np
import imageio
import cv2
import h5py
from tqdm import tqdm
import torch
from torchvision.transforms.v2 import CenterCrop
from einops import rearrange

BGR=False
OVERLAY_PATH = "heatmap_overlay.png"

# 1) Load your RGBA overlay once
overlay_rgba = imageio.imread(OVERLAY_PATH)
overlay_rgb  = overlay_rgba[...,:3].astype(np.float32)
alpha_map    = overlay_rgba[...,3].astype(np.float32) / 255.0
# expand alpha to (H,W,1)
alpha_map    = np.expand_dims(alpha_map, axis=-1)

for eps_id in tqdm(range(1)):
    dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_0520_v2_hdf5/episode_{eps_id}.hdf5"
    with h5py.File(dataset_path, "r") as root:
        decompressed_images = {}
        for cam_name in [
            "obses/images/left/top",
            "obses/images/right/top",
            "obses/images/left/wrist",
            "obses/images/right/wrist"
        ]:
            buf = root[cam_name][()]
            frames = []
            for b in buf:
                img = cv2.imdecode(b, cv2.IMREAD_COLOR) # BGR uint8
                if BGR:
                    img = img[..., ::-1] # to RGB
                frames.append(img)
            decompressed_images[cam_name] = np.stack(frames, axis=0)  # [T,H,W,3]

        horizon = decompressed_images["obses/images/left/top"].shape[0]
        out_frames = []
        for idx in range(horizon):
            # 2) Blend overlay onto the *right/top* view only
            rt = decompressed_images["obses/images/right/top"][idx].astype(np.float32)
            blended = (1 - alpha_map) * rt + alpha_map * overlay_rgb
            blended = blended.clip(0,255).astype(np.uint8)
            decompressed_images["obses/images/right/top"][idx] = blended

            # 3) Concatenate all four cams
            quad = np.concatenate([
                decompressed_images["obses/images/left/top"][idx],
                blended,
                decompressed_images["obses/images/left/wrist"][idx],
                decompressed_images["obses/images/right/wrist"][idx],
            ], axis=1)

            out_frames.append(quad[..., ::-1] if BGR else quad)

        # 4) Write out the video
        imageio.mimsave(
            f"sim_double_insert_round1_with_heat_{eps_id}.mp4",
            out_frames,
            fps=60,
        )
