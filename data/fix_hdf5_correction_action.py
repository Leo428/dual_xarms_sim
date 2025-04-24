import os
import glob
import h5py
import numpy as np
from tqdm import tqdm

from dual_xarms_sim.utils.transformation import construct_adjoint_matrix

# SRC_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0304_hdf5"
# DST_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0304_fixed_hdf5/"

# SRC_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0311_hdf5"
# DST_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0311_fixed_hdf5/"

# SRC_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318_hdf5"
# DST_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318_fixed_hdf5/"

SRC_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0419_hdf5/"
DST_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0419_fixed_hdf5/"

# Dataset paths inside each HDF5 file — change these if your structure differs:
LEFT_POSE_PATH  = "obses/state/left/tcp_pose"
RIGHT_POSE_PATH = "obses/state/right/tcp_pose"
ACTIONS_PATH    = "actions/relative_action"
ABS_ACTIONS_PATH = "actions/global_action"

os.makedirs(DST_DIR, exist_ok=True)

for src_fp in glob.glob(os.path.join(SRC_DIR, "*.hdf5")):
    file_name = os.path.basename(src_fp)
    dst_file_path = os.path.join(DST_DIR, file_name)
    print(f"Processing {file_name} → {dst_file_path}")

    with h5py.File(src_fp, "r") as f_src, h5py.File(dst_file_path, "w") as f_dst:
        # 1) copy everything except the actions dataset
        for name in f_src:
            if name == "actions":
                continue
            f_src.copy(name, f_dst, name)

        # 2) load poses and actions
        left_poses = f_src[LEFT_POSE_PATH ][()] # shape (T, 7)
        right_poses = f_src[RIGHT_POSE_PATH][()] # shape (T, 7)
        actions = f_src[ACTIONS_PATH   ][()] # shape (T, 14)
        intervention_steps = f_src["metadata/interventions"][()] # should just fail if not present

        # 3) create a mask for the intervention steps
        T = actions.shape[0]
        intervene_mask = np.zeros(T, dtype=bool)
        intervene_mask[intervention_steps] = True

        # 4) undo the bug timestep‑by‑timestep
        fixed_actions = np.empty_like(actions)
        for t in range(T):
            if intervene_mask[t]:
                fixed_actions[t] = actions[t] # already in wrist frame
            else:
                A_l = construct_adjoint_matrix(left_poses[t])
                A_r = construct_adjoint_matrix(right_poses[t])
                rel = actions[t].copy()
                rel[:6]   = np.linalg.inv(A_l) @ actions[t, :6]
                rel[7:13] = np.linalg.inv(A_r) @ actions[t, 7:13]
                fixed_actions[t] = rel

        f_src.copy(ABS_ACTIONS_PATH, f_dst, ABS_ACTIONS_PATH)
        # 5) write corrected actions back into the new HDF5
        f_dst.create_dataset(
            ACTIONS_PATH,
            data=fixed_actions,
        )

    print(f"  → Done.")