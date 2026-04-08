import gymnasium as gym
from gymnasium import spaces
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from typing import Any, Literal, Tuple, Dict
import zmq
import json
import time
from tqdm import tqdm
import cv2
import queue
import threading
from copy import deepcopy

from dual_xarms_sim.oculus_intervention import OculusIntervention

_HOME_JOINT_QPOS = np.array([0, -0.25891, -0.00020, 1.03223, 0, 1.31830, 0, 0, -0.25891, -0.00020, 1.03223, 0, 1.31830, 0])
_LEFT_HOME_TCP_POSE = np.array([7.4233063e-02, 3.8093147e-01, 1.9769229e-01, -2.7911010e-05, 9.9990791e-01, -9.1831549e-05, -1.3579541e-02])
_RIGHT_HOME_TCP_POSE = np.array([7.4233063e-02, -3.8093147e-01, 1.9769229e-01, -2.7911010e-05, 9.9990791e-01, -9.1831549e-05, -1.3579541e-02])
_LEFT_CARTESIAN_BOUNDS = np.array([[-0.3185 + 0.1, 0.381-0.381 - 0.25, 0], [0.381 + 0.03, 0.381 + 0.381, 0.6]])
_RIGHT_CARTESIAN_BOUNDS = np.array([[-0.3185 + 0.1, -0.381-0.381, 0], [0.381 + 0.03, -0.381 + 0.381 + 0.25, 0.6]])

class ImageDisplayer(threading.Thread):
    def __init__(self, queue: queue.Queue, heatmap_path: str = None):
        super().__init__(daemon=True)
        self.queue = queue
        self.show_overlay = False  # start with overlay hidden

        # these will all be 360×360 if heatmap_path is given
        self.heatmap_bgr    = None
        self.expanded_alpha = None

        if heatmap_path:
            heatmap = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
            if heatmap is None:
                raise RuntimeError(f"Failed to load heatmap from {heatmap_path}")

            # split out alpha if present
            if heatmap.shape[2] == 4:
                hm_bgr   = heatmap[..., :3]
                hm_alpha = heatmap[..., 3:] / 255.0
            else:
                hm_bgr   = heatmap
                hm_alpha = np.ones(hm_bgr.shape[:2] + (1,), dtype=np.float32)

            # First, get a 2D alpha map
            alpha2d = hm_alpha.squeeze(-1)         # now shape (H, W)
            # Resize that to 360×360 (still 2D)
            resized_alpha2d = cv2.resize(
                alpha2d,
                (360, 360),
                interpolation=cv2.INTER_LINEAR
            )
            # Now re-add the singleton channel dimension
            self.expanded_alpha = 0.7 * resized_alpha2d[..., np.newaxis]  # -> (360,360,1)
            # resize both to 360×360
            self.heatmap_bgr    = cv2.resize(hm_bgr,   (360, 360), interpolation=cv2.INTER_LINEAR)

            print(f"Loaded and resized heatmap to {self.heatmap_bgr.shape}")

    def set_overlay_visible(self, visible: bool):
        self.show_overlay = visible

    @staticmethod
    def _draw_vertical_dashed_line(img, x, color=(0,255,0), thickness=3, dash_length=10, gap_length=10):
        h, _ = img.shape[:2]
        y = 0
        while y < h:
            y_end = min(y + dash_length, h)
            cv2.line(img, (x, y), (x, y_end), color, thickness)
            y += dash_length + gap_length

    def run(self):
        while True:
            try:
                cam_list = self.queue.get()
                processed = []

                for name, bgr in cam_list:
                    # only overlay on the 'right/top' view, if heatmap is loaded and flagged
                    if (
                        self.show_overlay
                        and name == "right/top"
                        and self.heatmap_bgr is not None
                    ):
                        h_hm, w_hm = self.heatmap_bgr.shape[:2]
                        h_img, w_img = bgr.shape[:2]

                        # compute offsets to center the 360×360 heatmap on the 360×640 image
                        x0 = (w_img - w_hm) // 2
                        y0 = (h_img - h_hm) // 2  # this will be 0 if heights match

                        # blend in the region of interest
                        roi       = bgr[y0:y0+h_hm, x0:x0+w_hm].astype(np.float32)
                        hm_bgr    = self.heatmap_bgr.astype(np.float32)
                        alpha     = self.expanded_alpha.astype(np.float32)

                        blended   = (roi * (1 - alpha) + hm_bgr * alpha).astype(np.uint8)
                        # copy result back
                        out       = bgr.copy()
                        out[y0:y0+h_hm, x0:x0+w_hm] = blended

                        self._draw_vertical_dashed_line(out, x0)
                        self._draw_vertical_dashed_line(out, x0 + w_hm)

                        processed.append(out)
                    else:
                        processed.append(bgr)

                canvas = np.vstack((np.hstack(processed[:2]), np.hstack(processed[2:])))
                cv2.imshow("ZED Cameras (RGB)", canvas)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Error in ImageDisplayer: {e}")
                break


class RMPDualXArmsEnv(gym.Env):
    def __init__(self,
        seed: int = 0,
        control_freq: int = 60, # Hz
        time_limit: int = 3 * 60, # 3 minutes
        max_linear_velocity: float = 1.0, # m/s
        max_angular_velocity: float = np.pi/3, # rad/s
        overlay_heatmap: str = None,
    ):
        super().__init__()
        self.control_freq = control_freq
        self._MAX_LINEAR_VELOCITY = max_linear_velocity / control_freq
        self._MAX_ANGULAR_VELOCITY = max_angular_velocity / control_freq
        self.gym_rate = RateLimiter(control_freq, warn=False)

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict(
                {
                    "left/tcp_pose": spaces.Box( # world frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/tcp_vel": spaces.Box( # world frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "left/gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                    "left/joint_qpos": spaces.Box(
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/tcp_pose": spaces.Box( # world frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/tcp_vel": spaces.Box( # world frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "right/gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                    "right/joint_qpos": spaces.Box(
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/target_tcp_pose": spaces.Box( # world frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/target_gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                    "right/target_tcp_pose": spaces.Box( # world frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/target_gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                }
            ),
            "images": gym.spaces.Dict(
                {
                    "left/top": gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                    "left/wrist": gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                    "right/top": gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                    "right/wrist": gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                }
            )
        })

        # left tcp pos delta, left tcp euler delta, left gripper pos,
        # right tcp pos delta, right tcp euler delta, right gripper pos
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.zmq_context = zmq.Context()
        self.action_cmd_pub = self.zmq_context.socket(zmq.PUB)
        self.action_cmd_pub.setsockopt(zmq.CONFLATE, 1)  # Ensure only the latest message is kept
        self.action_cmd_pub.bind("tcp://127.0.0.1:5002")

        self.robot_state_sub = self.zmq_context.socket(zmq.SUB)
        self.robot_state_sub.setsockopt(zmq.RCVHWM, 1)  # Receive the latest message
        self.robot_state_sub.connect("tcp://127.0.0.1:5005")
        self.robot_state_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self.robot_states = {}
        self.images = {}
        self.left_target_tcp_pose = None
        self.right_target_tcp_pose = None
        self.left_target_gripper_pos = 0
        self.right_target_gripper_pos = 0

        self.latency_running_avg = 0.0
        self.bar = tqdm(total=100000000, desc="freq:")
        self.step_count = 0
        self.MAX_STEPS = time_limit * control_freq

        self.frames_queue = queue.Queue(maxsize=10)
        self._displayer = ImageDisplayer(self.frames_queue, overlay_heatmap)
        self._displayer.start()

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs = self._compute_observation()
        self.left_target_tcp_pose = _LEFT_HOME_TCP_POSE
        self.right_target_tcp_pose = _RIGHT_HOME_TCP_POSE
        self.left_target_gripper_pos = 840
        self.right_target_gripper_pos = 840
        count = 0
        while count < 1000:
            left_pos_diff = self.left_target_tcp_pose[:3] - self.robot_states["left/tcp_pose"][:3]
            right_pos_diff = self.right_target_tcp_pose[:3] - self.robot_states["right/tcp_pose"][:3]
            left_rot_diff = R.from_quat(self.left_target_tcp_pose[3:7], scalar_first=True) * R.from_quat(self.robot_states["left/tcp_pose"][3:7], scalar_first=True).inv()
            right_rot_diff = R.from_quat(self.right_target_tcp_pose[3:7], scalar_first=True) * R.from_quat(self.robot_states["right/tcp_pose"][3:7], scalar_first=True).inv()
            left_euler_diff = left_rot_diff.as_euler("xyz")
            right_euler_diff = right_rot_diff.as_euler("xyz")
            if np.linalg.norm(left_pos_diff) < 0.01 and np.linalg.norm(right_pos_diff) < 0.01 and \
                np.linalg.norm(left_euler_diff) < 0.01 and np.linalg.norm(right_euler_diff) < 0.01:
                print("Reset done!")
                break
            # limit the diff
            left_pos_diff = self.limit_offset_norm(left_pos_diff, self._MAX_LINEAR_VELOCITY * 0.8)
            right_pos_diff = self.limit_offset_norm(right_pos_diff, self._MAX_LINEAR_VELOCITY * 0.8)
            left_euler_diff = self.limit_offset_norm(left_euler_diff, self._MAX_ANGULAR_VELOCITY)
            right_euler_diff = self.limit_offset_norm(right_euler_diff, self._MAX_ANGULAR_VELOCITY)

            left_target_pose = np.concatenate([
                self.robot_states["left/tcp_pose"][:3] + left_pos_diff[:3],
                (
                    R.from_euler("xyz", left_euler_diff) * \
                    R.from_quat(self.robot_states["left/tcp_pose"][3:7], scalar_first=True)
                ).as_quat(scalar_first=True),
            ])
            right_target_pose = np.concatenate([
                self.robot_states["right/tcp_pose"][:3] + right_pos_diff[:3],
                (
                    R.from_euler("xyz", right_euler_diff) * \
                    R.from_quat(self.robot_states["right/tcp_pose"][3:7], scalar_first=True)
                ).as_quat(scalar_first=True),
            ])
            self.action_cmd_pub.send_json(
                {
                    "timestamp": time.time(),
                    "action": np.concatenate([
                        left_target_pose, [self.left_target_gripper_pos],
                        right_target_pose, [self.right_target_gripper_pos],
                    ]).tolist(),
                }
            )
            self.gym_rate.sleep()
            obs = self._compute_observation()

        obs = self._compute_observation()
        # while self.latency_running_avg > 0.02:
        #     obs = self._compute_observation() # this should wait for the first observation after reset
        #     time.sleep(0.001)

        self.left_target_tcp_pose = self.robot_states["left/tcp_pose"]
        self.right_target_tcp_pose = self.robot_states["right/tcp_pose"]
        self.bar.reset()
        self.step_count = 0
        return obs, {}

    def _compute_observation(self) -> Dict[str, np.ndarray]:
        while True:
            try:
                msg = self.robot_state_sub.recv_json(flags=zmq.NOBLOCK)
                timestamp = msg["timestamp"]
                robot_state = msg["robot_state"]
                metadata = msg["metadata"]
                for cam in metadata["cameras"]:
                    binary_data = self.robot_state_sub.recv()
                    self.images[cam] = np.frombuffer(binary_data, dtype=np.uint8).reshape(360, 640, 3)
                for k, v in robot_state.items():
                    self.robot_states[k] = np.array(v, dtype=np.float32)
                # IMPORTANT: record the target tcp pose and gripper pos as well
                self.robot_states["left/target_tcp_pose"] = self.left_target_tcp_pose
                self.robot_states["left/target_gripper_pos"] = np.array(
                    (self.left_target_gripper_pos,), dtype=np.float32
                )
                self.robot_states["right/target_tcp_pose"] = self.right_target_tcp_pose
                self.robot_states["right/target_gripper_pos"] = np.array(
                    (self.right_target_gripper_pos,), dtype=np.float32
                )

                self.frames_queue.put([
                    ("left/top", self.images["left/top"]),
                    ("right/top", self.images["right/top"]),
                    ("left/wrist", self.images["left/wrist"]),
                    ("right/wrist", self.images["right/wrist"]),
                ], block=False)

                self.latency_running_avg = 0.1 * self.latency_running_avg + \
                    0.9 * (time.time() - timestamp)
                # self.latency_running_avg = (time.time() - timestamp)
                self.bar.desc = f"avg latency: {self.latency_running_avg * 1000:.2f} ms"
                # return both states and images
                if self.latency_running_avg < 0.03:
                    return {
                        "state": deepcopy(self.robot_states),
                        "images": deepcopy(self.images),
                    }
            except zmq.Again:
                continue

    def limit_offset_norm(self, offset, max_offset):
        # scale offset such that the max norm of offset is max_offset
        norm = np.linalg.norm(offset)
        if norm > max_offset:
            offset = offset / norm * max_offset
        return offset

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = action.astype(np.float32)

        left_xyz = action[:3]
        left_rpy = action[3:6]
        left_gripper = action[6]
        right_xyz = action[7:10]
        right_rpy = action[10:13]
        right_gripper = action[13]

        left_d_xyz = self.limit_offset_norm(left_xyz - self.left_target_tcp_pose[0:3], self._MAX_LINEAR_VELOCITY)
        self.left_target_tcp_pose[0:3] = np.clip(
            self.left_target_tcp_pose[0:3] + left_d_xyz, _LEFT_CARTESIAN_BOUNDS[0], _LEFT_CARTESIAN_BOUNDS[1]
        )

        right_d_xyz = self.limit_offset_norm(right_xyz - self.right_target_tcp_pose[0:3], self._MAX_LINEAR_VELOCITY)
        self.right_target_tcp_pose[0:3] = np.clip(
            self.right_target_tcp_pose[0:3] + right_d_xyz, _RIGHT_CARTESIAN_BOUNDS[0], _RIGHT_CARTESIAN_BOUNDS[1]
        )

        left_current_rot = R.from_quat(self.left_target_tcp_pose[3:7], scalar_first=True)
        left_rot_delta = R.from_euler("xyz", left_rpy) * left_current_rot.inv()
        left_rot_delta_clamped = R.from_rotvec(self.limit_offset_norm(left_rot_delta.as_rotvec(), self._MAX_ANGULAR_VELOCITY))
        self.left_target_tcp_pose[3:7] = (left_rot_delta_clamped * left_current_rot).as_quat(scalar_first=True)

        right_current_rot = R.from_quat(self.right_target_tcp_pose[3:7], scalar_first=True)
        right_rot_delta = R.from_euler("xyz", right_rpy) * right_current_rot.inv()
        right_rot_delta_clamped = R.from_rotvec(self.limit_offset_norm(right_rot_delta.as_rotvec(), self._MAX_ANGULAR_VELOCITY))
        self.right_target_tcp_pose[3:7] = (right_rot_delta_clamped * right_current_rot).as_quat(scalar_first=True)

        # gripper action is global absolute position in [80, 840]
        left_target_gripper_pos = np.clip(np.array((left_gripper,), dtype=np.float32), 80, 840)
        right_target_gripper_pos = np.clip(np.array((right_gripper,), dtype=np.float32), 80, 840)

        # Send action command to central server
        target_cmd = np.concat([
            self.left_target_tcp_pose, left_target_gripper_pos,
            self.right_target_tcp_pose, right_target_gripper_pos,
        ])
        self.action_cmd_pub.send_json(
            {
                "timestamp": time.time(),
                "action": target_cmd.tolist(),
            }
        )

        self.step_count += 1
        reward = 0.0  # Placeholder
        done = False  # Placeholder
        truncated = self.step_count >= self.MAX_STEPS
        info = {}  # Placeholder

        self.gym_rate.sleep()
        self.bar.update(1)
        obs = self._compute_observation()
        return obs, reward, done, truncated, {}

    def close(self):
        self.action_cmd_pub.close()
        self.robot_state_sub.close()
        self._displayer.join()

    def seed(self, seed=None):
        pass


if __name__ == "__main__":
    from dual_xarms_sim.relative_frame import RelativeFrame, WristRelativeTo
    try:
        env = RMPDualXArmsEnv(
                control_freq=60,
                overlay_heatmap="/home/huzheyuan/dual_xarms/dual_xarms_sim/dual_xarms_sim/real_hang_r0.png"
            )
        env = OculusIntervention(env, freq=60)
        env = RelativeFrame(env)
        env = WristRelativeTo(env)

        obs, _ = env.reset()
        done = False
        # import ipdb; ipdb.set_trace()

        # For tracking intervention state to toggle overlay
        prev_intervention_active = False

        while not done:
            action = env.action_space.sample() * 0
            obs, reward, done, truncated, info = env.step(action)

            # Check if intervention state changed
            intervention_active = "intervene_action" in info
            if intervention_active != prev_intervention_active and hasattr(env.unwrapped, '_displayer') and env.unwrapped._displayer:
                prev_intervention_active = intervention_active
                # Toggle overlay visibility based on intervention state
                env.unwrapped._displayer.set_overlay_visible(intervention_active)
            # print(info)
            # print(obs["state"].keys())
            # print(obs["images"].keys())

    except Exception as e:
        print(e)
        env.close()