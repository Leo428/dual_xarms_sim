import numpy as np
import h5py
from tqdm import tqdm
import os
import glob

from dual_xarms_sim.utils.transformation import compute_relative_poses_batch, compute_relative_velocities_batch

if __name__ == "__main__":
    hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_riya_0312_hdf5"
    hdf5_files = glob.glob(os.path.join(hdf5_dir, "*.hdf5"))

    for filename in tqdm(hdf5_files):
        try:
            print(f"Processing file: {filename}")
            with h5py.File(filename, "r+") as root:
                left_group = root["obses/state/left"]
                right_group = root["obses/state/right"]

                left_gripper_pos = root["obses/state/left/gripper_pos"][()]
                right_gripper_pos = root["obses/state/right/gripper_pos"][()]

                if left_gripper_pos.ndim == 1 and "gripper_pos" in left_group:
                    print(f"fixed {filename} left gripper shape")
                    del left_group["gripper_pos"]
                    left_group.create_dataset("gripper_pos", data=left_gripper_pos[..., None])

                if right_gripper_pos.ndim == 1 and "gripper_pos" in right_group:
                    print(f"fixed {filename} right gripper shape")
                    del right_group["gripper_pos"]
                    right_group.create_dataset("gripper_pos", data=right_gripper_pos[..., None])

                # Make sure to flush changes to disk
                root.flush()

        except Exception as e:
            print(f"Failed to process file: {filename}")
            raise e
            # continue