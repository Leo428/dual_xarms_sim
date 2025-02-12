import numpy as np
import h5py

from dual_xarms_sim.utils.transformation import compute_relative_poses_batch, compute_relative_velocities_batch

if __name__ == "__main__":
    dataset_path = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_cube_1101_hdf5/episode_0.hdf5"
    # dataset_path = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_1212_hdf5/episode_0.hdf5"

    with h5py.File(dataset_path, "r") as root:
        print(root["obses/state/left"].keys())
        left_ee_pose = root["obses/state/left/ego_tcp_pose"][()]
        right_ee_pose = root["obses/state/right/ego_tcp_pose"][()]
        left_rel2_right_pose = root["obses/state/left/relative2_tcp_pose"][()]
        right_rel2_left_pose = root["obses/state/right/relative2_tcp_pose"][()]
        computed_left_rel2_right_pose = compute_relative_poses_batch(left_ee_pose, right_ee_pose)
        computed_right_rel2_left_pose = compute_relative_poses_batch(right_ee_pose, left_ee_pose)
        print(np.allclose(computed_left_rel2_right_pose, left_rel2_right_pose, atol=1e-7))
        print(np.allclose(computed_right_rel2_left_pose, right_rel2_left_pose, atol=1e-7))

        left_ee_vel = root["obses/state/left/ego_tcp_vel"][()]
        right_ee_vel = root["obses/state/right/ego_tcp_vel"][()]
        left_rel2_right_vel = root["obses/state/left/relative2_tcp_vel"][()]
        right_rel2_left_vel = root["obses/state/right/relative2_tcp_vel"][()]
        # compute relative velocities and compare with ground truth
        computed_left_rel2_right_vel = compute_relative_velocities_batch(left_ee_vel, right_ee_vel, left_ee_pose, right_ee_pose)
        computed_right_rel2_left_vel = compute_relative_velocities_batch(right_ee_vel, left_ee_vel, right_ee_pose, left_ee_pose)
        print(np.allclose(computed_left_rel2_right_vel, left_rel2_right_vel, atol=1e-7))
        print(np.allclose(computed_right_rel2_left_vel, right_rel2_left_vel, atol=1e-7))
