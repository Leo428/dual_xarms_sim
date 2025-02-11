import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R

dataset_path = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_cube_1101_hdf5/episode_0.hdf5"
# dataset_path = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_1212_hdf5/episode_0.hdf5"


def compute_relative_poses_batch(left_poses, right_poses):
    """
    Computes the left end-effector poses relative to the right end-effector poses in a batched manner.

    Parameters:
        left_poses  : np.ndarray of shape (B, 7) where each row is [x, y, z, qx, qy, qz, qw]
        right_poses : np.ndarray of shape (B, 7) where each row is [x, y, z, qx, qy, qz, qw]

    Returns:
        relative_poses: np.ndarray of shape (B, 7) where each row is the relative pose [x, y, z, qx, qy, qz, qw]
    """
    # Extract translations and quaternions for both left and right poses
    t_left = left_poses[:, :3]    # Shape: (B, 3)
    quat_left = left_poses[:, 3:]  # Shape: (B, 4)

    t_right = right_poses[:, :3]    # Shape: (B, 3)
    quat_right = right_poses[:, 3:]  # Shape: (B, 4)

    # Create batch Rotation objects
    R_left = R.from_quat(quat_left)   # Each row is treated as one rotation.
    R_right = R.from_quat(quat_right)

    # Compute the inverse rotations of the right poses (batched)
    R_right_inv = R_right.inv()

    # Compute the relative rotation: R_relative = R_right_inv * R_left
    # This performs elementwise multiplication for the batch.
    R_relative = R_right_inv * R_left
    quat_relative = R_relative.as_quat()  # Shape: (B, 4)

    # Compute the relative translation:
    # First, find the difference in translations (in world frame)
    delta_t = t_left - t_right  # Shape: (B, 3)
    # Rotate this difference into the right arm's frame.
    t_relative = R_right_inv.apply(delta_t)  # Shape: (B, 3)

    # Concatenate the relative translation and rotation to get the full relative pose
    relative_poses = np.hstack([t_relative, quat_relative], dtype=np.float32)  # Shape: (B, 7)

    return relative_poses

def compute_relative_velocities_batch(left_vel, right_vel, left_pose, right_pose):
    """
    Computes the left end-effector velocities relative to the right end-effector frame 
    in a batched (vectorized) manner.

    Parameters:
        left_vel : np.ndarray of shape (B, 6)
                   Each row is [vx, vy, vz, wx, wy, wz] for the left end-effector in world frame.
        right_vel: np.ndarray of shape (B, 6)
                   Each row is [vx, vy, vz, wx, wy, wz] for the right end-effector in world frame.
        left_pose: np.ndarray of shape (B, 7)
                   Each row is [x, y, z, qx, qy, qz, qw] for the left end-effector in world frame.
        right_pose: np.ndarray of shape (B, 7)
                   Each row is [x, y, z, qx, qy, qz, qw] for the right end-effector in world frame.

    Returns:
        relative_vel: np.ndarray of shape (B, 6)
                      Each row is the left end-effector's velocity relative to the right end-effector,
                      expressed in the right end-effector's frame as [vx, vy, vz, wx, wy, wz].
    """
    # Extract positions from the poses
    t_left = left_pose[:, :3]    # Shape: (B, 3)
    t_right = right_pose[:, :3]  # Shape: (B, 3)

    # Extract rotations from the right pose (we need these to transform into right's frame)
    quat_right = right_pose[:, 3:]  # Shape: (B, 4) in [qx, qy, qz, qw] order.
    R_right = R.from_quat(quat_right)
    R_right_inv = R_right.inv()     # This will rotate vectors from world into the right TCP's frame.

    # Extract linear and angular velocity components
    v_left    = left_vel[:, :3]   # Linear velocity of left TCP in world frame.
    omega_left = left_vel[:, 3:]  # Angular velocity of left TCP in world frame.

    v_right    = right_vel[:, :3]   # Linear velocity of right TCP in world frame.
    omega_right = right_vel[:, 3:]  # Angular velocity of right TCP in world frame.

    # Compute the difference in positions (from right to left)
    delta_t = t_left - t_right  # Shape: (B, 3)

    # Account for the velocity of a point displaced by delta_t due to the right's angular velocity.
    # (i.e. if the right arm rotates, a point offset from its origin would have an extra velocity)
    v_due_to_right_rotation = np.cross(omega_right, delta_t)

    # Compute the relative linear velocity in world frame:
    v_rel_world = v_left - (v_right + v_due_to_right_rotation)

    # Rotate the relative linear velocity into the right TCP's frame:
    v_rel_right = R_right_inv.apply(v_rel_world)

    # For angular velocity, the relative angular velocity in world frame is simply:
    omega_rel_world = omega_left - omega_right
    # Rotate into the right TCP's frame:
    omega_rel_right = R_right_inv.apply(omega_rel_world)

    # Concatenate linear and angular parts to get the full relative velocity (6D)
    relative_vel = np.hstack([v_rel_right, omega_rel_right]).astype(np.float32)

    return relative_vel

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
    import ipdb; ipdb.set_trace()