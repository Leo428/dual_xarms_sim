import numpy as np
import imageio
import cv2
import h5py
from tqdm import tqdm
import torch
from torchvision.transforms.v2 import CenterCrop, RandomCrop
from einops import rearrange

BGR=True
INTERVENTION=False
crop = CenterCrop((360,360))
# crop = RandomCrop((200, 200))
# crop = CenterCrop((200,200))
# crop = CenterCrop((224,224))

cam_names = [
    # "obses/images/left/top",
    "obses/images/right/top",
    # "obses/images/left/wrist",
    # "obses/images/right/wrist",
]

for eps_id in tqdm(range(28, 100)):
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round9_0720_hdf5/episode_{eps_id}.hdf5"
    # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round10_0801_hdf5/add/episode_{eps_id}.hdf5"
    dataset_path = f"/media/huzheyuan/data0/real_robot_data/real_lid_full_success_r9_hdf5/add/episode_{eps_id}.hdf5"
    # dataset_path = f"/media/huzheyuan/data0/real_robot_data/real_hang_full_success_r6_hdf5/episode_{eps_id}.hdf5"
    # dataset_path = f"/media/huzheyuan/data0/real_robot_data/real_burger_full_success_r6_hdf5/episode_{eps_id}.hdf5"
    with h5py.File(dataset_path, "r") as root:
        frames = []
        decompressed_images = {}
        if INTERVENTION:
            intervention_steps = root["metadata/interventions"][()]
        for cam_name in cam_names:
            compressed_images = root[cam_name][()]
            decompressed_frames = []
            step = 0
            for compressed_img in compressed_images:
                if INTERVENTION and step not in intervention_steps:
                    step += 1
                    continue
                step += 1

                decompressed_image = np.array(cv2.imdecode(compressed_img, 1))
                decompressed_image = torch.from_numpy(decompressed_image)
                decompressed_image = rearrange(decompressed_image, "h w c -> c h w")
                decompressed_image = crop(decompressed_image)
                decompressed_image = rearrange(decompressed_image, "c h w -> h w c")
                decompressed_frames.append(decompressed_image)
            decompressed_frames = np.stack(decompressed_frames)
            decompressed_images[cam_name] = decompressed_frames

        horizon = len(decompressed_images[cam_names[0]])
        for idx in range(horizon):
            frame = [
                decompressed_images[cam_name][idx] for cam_name in cam_names
            ]
            frame = np.concatenate(frame, axis=1)
            if BGR:
                frames.append(frame[..., ::-1])
            else:
                frames.append(frame)

        imageio.mimsave(f"real_hang_full_success_round6_rtop_{eps_id}.mp4", frames, fps=60)
        # break