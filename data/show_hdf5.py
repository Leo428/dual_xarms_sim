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

total_len = 0
total_intervention = 0

try:
    for eps_id in tqdm(range(0, 100)):
        # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round9_0720_hdf5/episode_{eps_id}.hdf5"
        # dataset_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round10_0801_hdf5/episode_{eps_id}.hdf5"
        # dataset_path = f"/media/huzheyuan/data0/real_robot_data/real_lid_full_success_r9_hdf5/episode_{eps_id}.hdf5"
        # dataset_path = f"/media/huzheyuan/data0/real_robot_data/real_hang_full_success_r6_hdf5/episode_{eps_id}.hdf5"
        dataset_path = f"/media/huzheyuan/data0/real_robot_data/real_burger_full_success_r8_hdf5/episode_{eps_id}.hdf5"
        with h5py.File(dataset_path, "r") as root:
            if INTERVENTION:
                intervention_steps = root["metadata/interventions"][()]
                total_intervention += len(intervention_steps)
                print(f"Episode {eps_id} has {len(intervention_steps)} intervention steps")
            ep_len = root["metadata/horizon"][()]
            total_len += ep_len
except Exception as e:
    print(f"Error processing episode {eps_id}: {e}")
    print(f"Total episodes processed: {eps_id}")
    print(f"Total length: {total_len}")
    print(f"Total intervention steps: {total_intervention}")
