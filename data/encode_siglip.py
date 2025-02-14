import os
import glob
import numpy as np
import h5py
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, Compose, Normalize
from einops import rearrange
from transformers import AutoConfig, SiglipImageProcessor, SiglipVisionModel

#########################################
# PRETRAINED IMAGE ENCODER (SIGLIP) 
#########################################
class SiglipVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "google/siglip-so400m-patch14-384"
        self.image_processor = SiglipImageProcessor.from_pretrained(self.model_name)
        self.cfg_only = AutoConfig.from_pretrained(self.model_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            self.model_name,
            attn_implementation="sdpa",
        )

    def forward(self, x):
        # x is expected to be (B, C, H, W)
        outs = self.vision_tower(x, interpolate_pos_encoding=True)
        image_features = outs.pooler_output  
        return image_features

#########################################
# DATASET (Modified to return meta info)
#########################################
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_dir, camera_names, norm_stats, episode_ids, episode_len,
            action_chunk, image_resize=(96,96), image_compressed=False,
            skip_last_n=0, obs_history_len=1, history_gap=9
        ):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.action_chunk = action_chunk
        self.episode_len = episode_len
        self.episode_len = [l - skip_last_n for l in self.episode_len]
        self.skip_last_n = skip_last_n
        self.cumulative_len = np.cumsum(self.episode_len)
        self.image_resize = image_resize
        self.image_compressed = image_compressed
        self.obs_history_len = obs_history_len
        self.history_gap = history_gap  # e.g. (0, 9)

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1], f"Index {index} is out of bounds"
        episode_index = np.argmax(self.cumulative_len > index)  # first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        # Determine which episode and time stamp
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        # Build list of time indices for observation history.
        time_indices = []
        for h_idx in reversed(range(self.obs_history_len)):
            step = start_ts - h_idx * self.history_gap
            step = max(step, 0)  # clamp negative indices to 0
            time_indices.append(step)

        with h5py.File(dataset_path, 'r') as root:
            # Load images for each camera at each requested time index.
            img_list_per_cam = {cam: [] for cam in self.camera_names}
            for t_idx in time_indices:
                for cam_name in self.camera_names:
                    img_data = root[f'/obses/images/{cam_name}'][t_idx]
                    if self.image_compressed:
                        img_data = cv2.imdecode(img_data, 1)  # decode to BGR
                        img_data = img_data[..., ::-1]       # convert BGR -> RGB
                    img_list_per_cam[cam_name].append(img_data)

        all_cam_images = []
        for cam_name in self.camera_names:
            imgs_for_cam = np.stack(img_list_per_cam[cam_name], axis=0)  # (obs_history_len, H, W, C)
            all_cam_images.append(imgs_for_cam)
        # Stack to get (obs_history_len, num_cams, H, W, C)
        all_cam_images = np.stack(all_cam_images, axis=1)

        # Convert to torch tensor and rearrange to channel-first:
        image_data = torch.from_numpy(all_cam_images)  # (s, num_cams, H, W, C)
        image_data = torch.einsum('s k h w c -> s k c h w', image_data)
        image_data = Resize(self.image_resize, antialias=None)(image_data)
        image_data = image_data / 255.0  # scale to [0,1]

        sample = {
            "observation.image": image_data,  # shape: (obs_history_len, num_cams, C, H, W)
            # Add meta information for later saving:
            "episode_id": episode_id,
            "time_indices": np.array(time_indices)  # shape: (obs_history_len,)
        }
        return sample

#########################################
# Normalization & DataLoader Functions
#########################################
def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_episode_len = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = np.concatenate((
                root["obses/state/left/relative2_tcp_pose"][()],
                root["obses/state/left/relative2_tcp_vel"][()],
            ), axis=-1)
        all_qpos_data.append(qpos)
        all_episode_len.append(len(qpos))
    return all_episode_len

def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, action_chunk, 
              image_resize=(96,96), image_compressed=False, skip_last_n=0, obs_history_len=1, history_gap=9):
    print(f'\nData from: {dataset_dir}\n')
    print(f'Will train using image sizes of {image_resize} pixels')
    train_episode_ids = np.arange(num_episodes)
    val_episode_ids = np.arange(num_episodes)
    print(f'\n- Train on {train_episode_ids} episodes\n- Val on {val_episode_ids} episodes\n')
    all_episode_len = get_norm_stats(dataset_dir, num_episodes)
    train_episode_len = [all_episode_len[i] for i in train_episode_ids]
    val_episode_len = [all_episode_len[i] for i in val_episode_ids]
    train_dataset = EpisodicDataset(dataset_dir, camera_names, {}, train_episode_ids, train_episode_len, action_chunk,
                                    image_resize=image_resize, image_compressed=image_compressed, skip_last_n=skip_last_n,
                                    obs_history_len=obs_history_len, history_gap=history_gap)
    val_dataset = EpisodicDataset(dataset_dir, camera_names, {}, val_episode_ids, val_episode_len, action_chunk,
                                  image_resize=image_resize, image_compressed=image_compressed, skip_last_n=skip_last_n,
                                  obs_history_len=obs_history_len, history_gap=history_gap)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                num_workers=16, prefetch_factor=2, persistent_workers=True)
    return val_dataloader, all_episode_len

#########################################
# EXAMPLE USAGE
#########################################
if __name__ == "__main__":
    # Paths and parameters:
    dataset_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_1212_hdf5/"
    num_episodes = 55
    camera_names = ['left/top', 'right/top', 'left/wrist', 'right/wrist']
    action_chunk = 60
    batch_size = 64
    image_resize = (180, 320)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation transform for the encoder: note we resize and normalize
    eval_img_transform = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataloaders.
    val_dl, all_episode_len = load_data(dataset_dir, num_episodes, camera_names, 
                                               64, 64, action_chunk,
                                               image_resize=image_resize,
                                               image_compressed=True, skip_last_n=0,
                                               obs_history_len=1,
                                               history_gap=9)
    val_iter = iter(val_dl)
    encoder = torch.compile(SiglipVisionTower().to(device))
    torch.set_float32_matmul_precision('high')
    encoder.eval()

    for ep_id, ep_len in tqdm(enumerate(all_episode_len)):
        images = []
        for _ in tqdm(range(ep_len), desc="decompressing images"):
            batch = next(val_iter)
            images.append(batch["observation.image"])

        # round up to nearest batch size
        batch_nums = -(-len(images) // batch_size)
        encoded_features = []
        for i in tqdm(range(batch_nums), desc="batch encoding"):
            images_batch = torch.cat(images[i*batch_size:(i+1)*batch_size], dim=0).to(device)
            B, S, K = images_batch.shape[:3]
            with torch.inference_mode():
                images_batch = eval_img_transform(images_batch)
                images_batch = rearrange(images_batch, 'b s k ... -> (b s k) ...')
                image_features = encoder(images_batch)
                image_features = rearrange(image_features, '(b s k) ... -> b s k ...', b=B, s=S, k=K)
                encoded_features.append(image_features)
        encoded_features = torch.cat(encoded_features, dim=0)
        assert encoded_features.shape[0] == len(images) == ep_len, f"Mismatch in number of encoded features for episode {ep_id}"

        try:
            filename = os.path.join(dataset_dir, f"episode_{ep_id}.hdf5")
            with h5py.File(filename, "r+") as root:
                for cam_idx, cam_name in enumerate(camera_names):
                    if f"obses/images/{cam_name}_encoded" in root:
                        root[f"obses/images/{cam_name}_encoded"][...] = encoded_features[:, :, cam_idx, :].cpu().numpy()
                    else:
                        root.create_dataset(
                            f"obses/images/{cam_name}_encoded", data=encoded_features[:, :, cam_idx, :].cpu().numpy()
                        )
                root.flush()
        except Exception as e:
            print(f"Failed to update file {filename}")
            break

        print(f"Updated file {filename} successfully")
