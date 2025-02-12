import numpy as np
import h5py
from tqdm import tqdm
import os
import glob

from dual_xarms_sim.utils.transformation import compute_relative_poses_batch, compute_relative_velocities_batch

if __name__ == "__main__":
    hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_1212_hdf5/"
    hdf5_files = glob.glob(os.path.join(hdf5_dir, "*.hdf5"))

    for filename in tqdm(hdf5_files):
        try:
            print(f"Processing file: {filename}")
            with h5py.File(filename, "r+") as root:
                left_group = root["obses/state/left"]
                right_group = root["obses/state/right"]

                left_tcp_pose = root["obses/state/left/tcp_pose"][()]
                right_tcp_pose = root["obses/state/right/tcp_pose"][()]
                left_tcp_vel = root["obses/state/left/tcp_vel"][()]
                right_tcp_vel = root["obses/state/right/tcp_vel"][()]

                # compute relative velocities
                computed_left_rel2_right_pose = compute_relative_poses_batch(left_tcp_pose, right_tcp_pose)
                computed_right_rel2_left_pose = compute_relative_poses_batch(right_tcp_pose, left_tcp_pose)
                computed_left_rel2_right_vel = compute_relative_velocities_batch(left_tcp_vel, right_tcp_vel, left_tcp_pose, right_tcp_pose)
                computed_right_rel2_left_vel = compute_relative_velocities_batch(right_tcp_vel, left_tcp_vel, right_tcp_pose, left_tcp_pose)

                if "relative2_tcp_pose" in left_group:
                    # raise RuntimeError("Dataset 'relative2_tcp_pose' already exists in the left group. Skipping creation.")
                    left_group["relative2_tcp_pose"][...] = computed_left_rel2_right_pose
                else:
                    left_group.create_dataset(
                        "relative2_tcp_pose", data=computed_left_rel2_right_pose, dtype=np.float32
                    )

                if "relative2_tcp_pose" in right_group:
                    # raise RuntimeError("Dataset 'relative2_tcp_pose' already exists in the right group. Skipping creation.")
                    right_group["relative2_tcp_pose"][...] = computed_right_rel2_left_pose
                else:
                    right_group.create_dataset(
                        "relative2_tcp_pose", data=computed_right_rel2_left_pose, dtype=np.float32
                    )

                if "relative2_tcp_vel" in left_group:
                    # raise RuntimeError("Dataset 'relative2_tcp_vel' already exists in the left group. Skipping creation.")
                    left_group["relative2_tcp_vel"][...] = computed_left_rel2_right_vel
                else:
                    left_group.create_dataset(
                        "relative2_tcp_vel", data=computed_left_rel2_right_vel, dtype=np.float32
                    )

                if "relative2_tcp_vel" in right_group:
                    # raise RuntimeError("Dataset 'relative2_tcp_vel' already exists in the right group. Skipping creation.")
                    right_group["relative2_tcp_vel"][...] = computed_right_rel2_left_vel
                else:
                    right_group.create_dataset(
                        "relative2_tcp_vel", data=computed_right_rel2_left_vel, dtype=np.float32
                    )

                # Make sure to flush changes to disk
                root.flush()

        except Exception as e:
            print(f"Failed to process file: {filename}")
            print(e)
            # continue