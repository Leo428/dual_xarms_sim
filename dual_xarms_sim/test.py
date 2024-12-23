from scipy.spatial.transform import Rotation as R
import numpy as np

ee_pose = np.array([0.0136,  0.6683,  0.4373, -0.2653,  0.9537,  0.1296, -0.0565])
tcp_pose = np.array([-0.03139479, 0.79367876, 0.219799, -0.26534453, 0.9537316, 0.12959905, -0.05650368])
target_ee = np.array([0.0136,  0.6684,  0.4372, -0.2653,  0.9538,  0.1295, -0.0564])
target_tcp = np.array([0.00364719, 0.53564703, 0.21970052, -0.26527676, 0.95376605, 0.12953007, -0.05639593])

# transform from tcp to ee
tcp_to_ee = np.array([0.0, 0.0, -0.255]) # translation

tcp_to_ee_rot = R.from_quat(tcp_pose[3:], scalar_first=True)
trans = tcp_to_ee_rot.apply(tcp_to_ee)

new_ee_pose = tcp_pose
new_ee_pose[:3] += trans
print(new_ee_pose)
print(ee_pose)
