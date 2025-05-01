import os
import glob
import h5py
import numpy as np
from tqdm import tqdm

# DST_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0304_fixed_hdf5/"

# DST_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0311_fixed_hdf5/"

DST_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318_fixed_hdf5/"

# DST_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0419_fixed_hdf5/"

for file_name in tqdm(glob.glob(os.path.join(DST_DIR, "*.hdf5"))):
    print(f"Processing {file_name}")
    try:
        with h5py.File(file_name, "r+") as root:
            intervention_steps = root["metadata/interventions"][()] # should just fail if not present
            actions = root["actions"]["relative_action"][()]
            # 3) create a mask for the intervention steps
            T = actions.shape[0]
            intervene_mask = np.zeros(T, dtype=bool)
            intervene_mask[intervention_steps] = True

            root.create_dataset(
                "obses/state/is_intervention",
                data=intervene_mask,
            )
            root.flush()
    except Exception as e:
        raise e
