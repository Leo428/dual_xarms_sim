import numpy as np
import h5py
import os  # New import for directory handling
import glob  # New import for file pattern matching
from tqdm import tqdm

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_cube_1101"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_cube_1101_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_0226"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_0226_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_jasmine_0226"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_jasmine_0226_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_0228"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_robyn_0228_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0304"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0304_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0311"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0311_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_riya_0312"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_riya_0312_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_robyn_0228"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_robyn_0228_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_robyn_0324"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_robyn_0324_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_0325"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_0325_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_riya_0327"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_riya_0327_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_jasmine_0331"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_jasmine_0331_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_ood_0403"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_ood_0403_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0419"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0419_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0423"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_zheyuan_correction_0423_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_action_coord_frame_bugfix"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_action_coord_frame_bugfix_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318_fixed"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318_fixed_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round2"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round2_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round3"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round3_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round4"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round4_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round5"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round5_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_0508"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_0508_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round6/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round6_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_0520/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_0520_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_0520_v2/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_0520_v2_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_0521_v2/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round1_0521_v2_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round2_0522/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round2_0522_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round3_0605/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round3_0605_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round4_0606/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round4_0606_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round5_0812_hdf5/fs/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_round5_0812_hdf5/fs/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round0_jasmine/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round0_jasmine_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round1_0612/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round1_0612_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round2_0614/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round2_0614_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round3v0_0806_hdf5/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round3v0_0806_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round4v0_0809_hdf5/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round4v0_0809_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round5v0_0811_hdf5/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round5v0_0811_hdf5/"

npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round6v0_0813_hdf5/fs/"
hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round6v0_0813_hdf5/fs/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round1_0620/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round1_0620_hdf5/"

# npz_directory = "/media/huzheyuan/data0/real_robot_data/real_hang_round1_0623/"
# hdf5_dir = "/media/huzheyuan/data0/real_robot_data/real_hang_round1_0623_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round2_0621/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round2_0621_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round2_0624/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round2_0624_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round3_0703/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round3_0703_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round4_0705/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round4_0705_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round5_0709/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round5_0709_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round6_0711/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round6_0711_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round6_0717/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round6_0717_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round7_0722/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round7_0722/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round0/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round0_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round1_0619/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round1_0619_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round1_0623/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round1_0623_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round1_0626/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round1_0626_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round2_0627/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round2_0627_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round3_0701/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round3_0701_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round4_0703/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round4_0703_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round5_0707/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round5_0707_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round6_0708/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round6_0708_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round7_0710/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round7_0710_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round8_0712/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round8_0712_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round6_0716/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round6_0716_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round7_0717/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round7_0717_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round7_0718/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round7_0718_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round8_0723_hdf5/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round8_0723_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round9_0728_hdf5/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round9_0728_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round10_0801_hdf5/add/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_lid_round10_0801_hdf5/add/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round0/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round0_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round1_0624/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round1_0624_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round2_0626/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round2_0626_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round3_0627/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round3_0627_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round4_0630/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round4_0630_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round5_0702/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round5_0702_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round6_0707/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round6_0707_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round7_0709/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round7_0709_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round8_0716/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round8_0716_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round9_0720/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round9_0720_hdf5/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round9_0720_hdf5/add/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_burger_round9_0720_hdf5/add/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/replay_test/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/replay_test_hdf5/"

# npz_directory = "/media/huzheyuan/data0/real_robot_data/real_lid_full_success_r9_hdf5/add/"
# hdf5_dir = "/media/huzheyuan/data0/real_robot_data/real_lid_full_success_r9_hdf5/add/"

# npz_directory = "/media/huzheyuan/data0/real_robot_data/round7_burger_full_success/"
# hdf5_dir = "/media/huzheyuan/data0/real_robot_data/round7_burger_full_success/"

# npz_directory = "/media/huzheyuan/data0/real_robot_data/round6_real_hang_full_success/"
# hdf5_dir = "/media/huzheyuan/data0/real_robot_data/round6_real_hang_full_success/"

# npz_directory = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round7_0722/fs/"
# hdf5_dir = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_round7_0722/fs/"

# npz_directory = "/media/huzheyuan/data0/real_robot_data/real_burger_full_success_r8_hdf5/add/"
# hdf5_dir = "/media/huzheyuan/data0/real_robot_data/real_burger_full_success_r8_hdf5/add/"

npz_files = glob.glob(os.path.join(npz_directory, "*.npz"))
episode_id = 0 # the starting episode id

IS_INTERVENTION = False #True or False
total_intervention = 0
total_steps = 0

for filename in tqdm(npz_files):
    try:
        print(f"Processing file: {filename}")
        with np.load(filename, allow_pickle=True) as data:
            # Create an HDF5 file to store the data
            if IS_INTERVENTION:
                interventions = []
                for step, info in enumerate(data['infos']):
                    has_intervention = 'intervene_action' in info
                    if step >= 60 and has_intervention:
                        interventions.append(step)
                print(f"episode {episode_id} has {len(interventions)} interventions")
                total_intervention += len(interventions)
                if len(interventions) == 0:
                    print(f"episode {episode_id} has no interventions")
                    continue
            hdf5_filename = f"episode_{episode_id}.hdf5"
            with h5py.File(hdf5_dir + hdf5_filename, 'w') as h5f:
                # Save metadata
                metadata = h5f.create_group('metadata')
                metadata['task'] = 'real_xarms_shirt_hang_variations'
                metadata['og_filename'] = filename
                metadata['horizon'] = len(data['rews'])
                total_steps += len(data['rews'])
                if IS_INTERVENTION:
                    metadata['interventions'] = np.array(interventions)

                # Process observations
                obses_list = data["obses"]  # List of observations
                relative_action = data["actions"]  # array of actions (horizon, action_dim)
                # Create groups for 'state' and 'images'
                state_group = h5f.create_group('obses/state')
                images_group = h5f.create_group('obses/images')
                action_group = h5f.create_group('actions')
                rewards_group = h5f.create_group('rewards')
                dones_group = h5f.create_group('dones')
                truncateds_group = h5f.create_group('truncateds')

                # Get all keys from 'state' and 'images' dictionaries
                state_keys = list(obses_list[0]['state'].keys())
                state_keys.remove('left/og_action')
                state_keys.remove('right/og_action')
                images_keys = list(obses_list[0]['images'].keys())
                action_absolute_keys = ['left/og_action', 'right/og_action']

                # Initialize dictionaries to collect data
                state_data = {k: [] for k in state_keys}
                images_data = {k: [] for k in images_keys}
                images_len = {k: [] for k in images_keys}
                global_action_data = []
                # Iterate over each observation
                for obs in obses_list:
                    state = obs['state']
                    images = obs['images']
                    for k in state_keys:
                        state_data[k].append(state[k])
                    for k in images_keys:
                        images_data[k].append(images[k])
                        images_len[k].append(len(images[k]))

                    global_action_data.append(np.concat([state['left/og_action'], state['right/og_action']]))

                # Convert lists to numpy arrays and save to HDF5
                for k in state_keys:
                    state_array = np.array(state_data[k], dtype=np.float32)
                    state_group.create_dataset(k, data=state_array)

                for k in images_keys:
                    compressed_len = np.array(images_len[k])
                    padded_size = compressed_len.max()
                    padded_compressed_image_list = []
                    for compressed_image in images_data[k]:
                        padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                        image_len = len(compressed_image)
                        padded_compressed_image[:image_len] = compressed_image
                        padded_compressed_image_list.append(padded_compressed_image)

                    images_array = np.array(padded_compressed_image_list)
                    # images_array = np.array(images_data[k])
                    images_group.create_dataset(k, data=images_array, dtype=np.uint8)

                global_action_array = np.array(global_action_data, dtype=np.float32)
                action_group.create_dataset('global_action', data=global_action_array)
                relative_action_array = np.array(relative_action, dtype=np.float32)
                action_group.create_dataset('relative_action', data=relative_action_array)

                rewards_group.create_dataset('rewards', data=np.array(data['rews'], dtype=np.float32))

                dones_group.create_dataset('dones', data=np.array(data['dones']))

                truncateds_group.create_dataset('truncateds', data=np.array(data['truncateds']))

        print(f"File {filename} processed successfully, saved as {hdf5_filename}")
        episode_id += 1

    except Exception as e:
        print(f"Failed to process file: {filename}")
        raise e

print(f"Total episodes processed: {episode_id}")
print(f"Total steps: {total_steps}")
print(f"Total interventions: {total_intervention}")