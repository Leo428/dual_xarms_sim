import numpy as np
import imageio
import cv2
import h5py
from tqdm import tqdm
import torch
from torchvision.transforms.v2 import CenterCrop, GaussianBlur, Compose, Resize, ToDtype
from einops import rearrange

from dual_xarms_sim.utils.blur import gaussian_2d_smoothing

BGR=True
crop = CenterCrop((360,360))
image_aug = Compose([
    CenterCrop((360,360)),
    Resize((224,224)),
    ToDtype(torch.float32, scale=True)
    # Resize((140,140)),
    # CenterCrop((128,128)),
])
to_uint8 = ToDtype(torch.uint8, scale=True)

camera_names = [
    # "obses/images/left/top",
    "obses/images/right/top",
    "obses/images/left/wrist",
    "obses/images/right/wrist"
]

for eps_id in tqdm(range(1)):
    dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_0325_hdf5/episode_{eps_id}.hdf5"
    with h5py.File(dataset_path, "r") as root:
        frames = []
        decompressed_images = []
        for cam_name in camera_names:
            compressed_images = root[cam_name][()]
            decompressed_frames = []
            for compressed_img in compressed_images:
                decompressed_image = torch.tensor(cv2.imdecode(compressed_img, 1), dtype=torch.uint8)
                decompressed_frames.append(
                    image_aug(rearrange(decompressed_image, "h w c -> c h w"))
                )

            decompressed_frames = torch.stack(decompressed_frames)
            # decompressed_frames = rearrange(decompressed_frames, "t c h w -> t h w c")
            decompressed_images.append(decompressed_frames) # B, K, C, H, W

        decompressed_images = torch.stack(decompressed_images, dim=1)

        B, K = decompressed_images.shape[:2]
        apply_blur_mask = torch.rand(B) <= 0.1  # shape: (B,)
        rand_k = torch.randint(0, K, (B,))
        for k in range(K):
            # Get mask of which batch elements chose camera k
            mask = (rand_k == k) & apply_blur_mask # shape: (B,)
            if mask.any():
                # Select images to blur: shape (N, C, H, W)
                selected = decompressed_images[mask, k]
                blurred = gaussian_2d_smoothing(selected, scale=5.0)
                # Put back blurred images into copy
                decompressed_images[mask, k] = blurred

        decompressed_images = rearrange(decompressed_images, "b k c h w -> b k h w c")
        decompressed_images = rearrange(decompressed_images, "b k h w c -> b h (k w) c")
        decompressed_images = to_uint8(decompressed_images).numpy()
        if BGR:
            decompressed_images = decompressed_images[..., ::-1]

        imageio.mimsave(f"blur_{eps_id}.mp4", decompressed_images, fps=60)