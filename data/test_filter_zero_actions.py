import numpy as np
import imageio
import cv2
import h5py
from tqdm import tqdm
import torch
from torchvision.transforms.v2 import CenterCrop
from einops import rearrange

BGR=True
crop = CenterCrop((360,360))

for eps_id in tqdm(range(15)):
    dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_jasmine_0331_hdf5/episode_{eps_id}.hdf5"
    with h5py.File(dataset_path, "r") as root:
        frames = []
        decompressed_images = {}
        for cam_name in ["obses/images/left/top", "obses/images/right/top", "obses/images/left/wrist", "obses/images/right/wrist"]:
            compressed_images = root[cam_name][()]
            decompressed_frames = []
            for compressed_img in compressed_images:
                decompressed_image = np.array(cv2.imdecode(compressed_img, 1))
                decompressed_frames.append(decompressed_image)
            decompressed_frames = np.stack(decompressed_frames)
            decompressed_frames = torch.from_numpy(decompressed_frames)
            decompressed_frames = crop(rearrange(decompressed_frames, "t h w c -> t c h w"))
            decompressed_frames = rearrange(decompressed_frames, "t c h w -> t h w c")
            decompressed_images[cam_name] = decompressed_frames

        horizon = len(decompressed_images["obses/images/left/top"])
        # actions = root["actions"]["relative_action"][()]
        left_tcp_vel = root["obses"]["state"]["left"]["wrist_tcp_vel"][()]
        right_tcp_vel = root["obses"]["state"]["right"]["wrist_tcp_vel"][()]

        tol = 1e-2
        filtered_zero_counts = 0
        for idx in range(horizon):
            if np.all(
                    np.logical_and(
                        np.abs(left_tcp_vel[idx]) < tol, np.abs(right_tcp_vel[idx]) < tol
                    )
                ):
                filtered_zero_counts += 1
                continue

            frame = [
                decompressed_images["obses/images/left/top"][idx],
                decompressed_images["obses/images/right/top"][idx],
                decompressed_images["obses/images/left/wrist"][idx],
                decompressed_images["obses/images/right/wrist"][idx],
            ]
            # frame = [
            #     crop(decompressed_images["obses/images/left/top"][idx]),
            #     crop(decompressed_images["obses/images/right/top"][idx]),
            #     crop(decompressed_images["obses/images/left/wrist"][idx]),
            #     crop(decompressed_images["obses/images/right/wrist"][idx]),
            # ]
            frame = np.concatenate(frame, axis=1)
            if BGR:
                frames.append(frame[..., ::-1])
            else:
                frames.append(frame)

        print(f"Filtered zero actions: {filtered_zero_counts} for episode {eps_id}")
        filtered_zero_counts = 0
        imageio.mimsave(f"filtered_crop_real_hang_jasmine_0331_ep{eps_id}.mp4", frames, fps=60)