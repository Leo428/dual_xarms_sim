from pathlib import Path
from typing import Any, Literal, Tuple, Dict
import time
import threading

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mink
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from dual_xarms_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from dual_xarms_sim.ik_controller import IKController

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "ufactory_xarm7" / "dual_scene.xml"

LEFT_HOME = np.asarray([-0.35, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0])
RIGHT_HOME = np.asarray([0.35, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0])
LEFT_CARTESIAN_BOUNDS = np.asarray([[-0.7, 0.2, 0], [0.1, 0.6, 0.3]])
# LEFT_EULER_BOUNDS = np.asarray([[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]])
RIGHT_CARTESIAN_BOUNDS = np.asarray([[-0.1, 0.2, 0], [0.7, 0.6, 0.3]])
# RIGHT_EULER_BOUNDS = np.asarray([[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]])
_SAMPLING_BOUNDS = np.asarray([[0, 0.3], [0.2, 0.5]])

# Define joint names based on the xarm7 structure from your model
_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
# All joints on xarm7 are assumed to have similar velocity limits
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}
_HOME_JOINT_QPOS = np.array([0, -0.25844, -0.00013, 1.03062, -0.00006, 1.31739, 0, 0, -0.25844, -0.00013, 1.03062, 0.00006, 1.31739, 0])
_HOME_JOINT_CTRL = np.array([0.785398163, -0.247, 0, 0.909, 0, 1.15644, 0, 0])
_MAX_LINEAR_VELOCITY = 0.75 # m/s
_MAX_ANGULAR_VELOCITY = np.pi/3 # rad/s

class DualXarmsGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_freq: int = 10, # 10 Hz
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(height=224, width=224),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
    ):
        self.control_freq = control_freq
        self.MAX_LINEAR_VELOCITY = _MAX_LINEAR_VELOCITY / control_freq
        self.MAX_ANGULAR_VELOCITY = _MAX_ANGULAR_VELOCITY / control_freq
        self._action_scale = action_scale
        self.gym_rate = RateLimiter(frequency=control_freq)

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=1 / control_freq,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(control_freq),
        }

        self.render_mode = render_mode
        self.camera_names = ["left/top", "left/wrist", "right/top", "right/wrist"]
        self.image_obs = image_obs

        joint_names = []
        self.velocity_limits = {}
        for prefix in ["left", "right"]:
            for n in _JOINT_NAMES:
                name = f"{prefix}/{n}"
                joint_names.append(name)
                self.velocity_limits[name] = _VELOCITY_LIMITS[n]
        self.arm_dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.arm_actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

        gripper_names = [f"{prefix}/gripper" for prefix in ["left", "right"]]
        self._gripper_ctrl_ids = [self._model.actuator(gripper_name).id for gripper_name in gripper_names]
        self._tcp_site_ids = [self._model.site(f"{side}/link_tcp").id for side in ["left", "right"]]
        # self._block_z = self._model.geom("block").size[2]

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict(
                {
                    "left/tcp_pose": spaces.Box( # world frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/tcp_vel": spaces.Box( # world frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "left/ego_tcp_pose": spaces.Box( # head camera frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/ego_tcp_vel": spaces.Box( # head camera frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "left/gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                    "right/tcp_pose": spaces.Box( # world frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/tcp_vel": spaces.Box( # world frame
                        -np.inf, np.inf, shape=(6,), dtype=np.float32 # linear + angular euler
                    ),
                    "right/ego_tcp_pose": spaces.Box( # head camera frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/ego_tcp_vel": spaces.Box( # head camera frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "right/gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                    # "block_pose": spaces.Box( # world frame, pos + quat
                    #     -np.inf, np.inf, shape=(7,), dtype=np.float32
                    # ),
                    # "block_pose_ego": spaces.Box( # head camera frame, pos + quat
                    #     -np.inf, np.inf, shape=(7,), dtype=np.float32
                    # ),
                    # "block_pose_wrist": spaces.Box( # wrist frame, pos + quat
                    #     -np.inf, np.inf, shape=(7,), dtype=np.float32
                    # ),
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

        if self.render_mode == "human":
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=True, show_right_ui=True)

        self._renderer = mujoco.Renderer(self.model, width=render_spec.width, height=render_spec.height)

        self.ik_configuration = mink.Configuration(self.model)
        # Task definitions using mink library
        self.l_ee_task = mink.FrameTask(
            frame_name="left/link_tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.r_ee_task = mink.FrameTask(
            frame_name="right/link_tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1e-4)
        self.tasks = [self.l_ee_task, self.r_ee_task, self.posture_task]
        # Fetch geometry IDs for collision avoidance
        l_wrist_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("left/link7").id)
        r_wrist_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("right/link7").id)
        l_upper_arm_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("left/link1").id)
        r_upper_arm_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("right/link1").id)

        # Define geometry IDs for the environment if needed (example: table or frames)
        # You would need to define these based on your actual environment setup
        table_geoms = ["floor"]  # Placeholder, replace with actual geom ID(s)

        # Define collision pairs
        collision_pairs = [
            # (l_wrist_geoms, r_wrist_geoms),  # Avoid collisions between the left and right end-effectors
            (l_upper_arm_geoms + r_upper_arm_geoms, table_geoms),  # Avoid collisions between arms and the table
            (l_upper_arm_geoms, r_upper_arm_geoms),  # Avoid collisions between the left and right arms
        ]
        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.01,
            collision_detection_distance=0.05,
        )
        self.ik_limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(self.model, self.velocity_limits),
            collision_avoidance_limit,
        ]
        self.ik_rate = RateLimiter(frequency=100.0)
        self.ik_controller = IKController(
            model=self._model, data=self._data,
            configuration=self.ik_configuration,
            actuator_ids=self.arm_actuator_ids, dof_ids=self.arm_dof_ids,
            tasks=self.tasks, l_ee_task=self.l_ee_task, r_ee_task=self.r_ee_task,
            ik_solver="quadprog", ik_limits=self.ik_limits,
            ik_max_iters=2, pos_threshold=1e-2, ori_threshold=1e-2,
            damping=1e-5, rate=self.ik_rate, human_viewer=self._viewer,
        )
        self.ik_thread = threading.Thread(target=self.ik_controller.run_ik, daemon=True)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self.arm_dof_ids] = _HOME_JOINT_QPOS
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        self._data.mocap_pos[0], self._data.mocap_quat[0] = LEFT_HOME[:3], LEFT_HOME[3:]
        self._data.mocap_pos[1], self._data.mocap_quat[1] = RIGHT_HOME[:3], RIGHT_HOME[3:]
        mujoco.mj_forward(self._model, self._data)

        # Sample a new block position.
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, 0.02)
        mujoco.mj_forward(self._model, self._data)

        with self.ik_controller.lock:
            self.ik_configuration.update(self._data.qpos)
            self.posture_task.set_target_from_configuration(self.ik_configuration)
            self.ik_controller.set_targets(
                LEFT_HOME[:3], LEFT_HOME[3:], RIGHT_HOME[:3], RIGHT_HOME[3:]
            )
            if not self.ik_controller.running:
                self.ik_thread.start()

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray
            dimensions are: left tcp pos delta, left tcp euler delta, left gripper pos,
                right tcp pos delta, right tcp euler delta, right gripper pos

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        left_tcp_pos_delta = action[:3]
        left_tcp_euler_delta = action[3:6]
        right_tcp_pos_delta = action[7:10]
        right_tcp_euler_delta = action[10:13]

        # # Set the mocap position.
        left_pos = self._data.mocap_pos[0].copy()
        left_dpos = self.limit_offset_norm(
            left_tcp_pos_delta * self.MAX_LINEAR_VELOCITY, self.MAX_LINEAR_VELOCITY
        )
        left_npos = np.clip(left_pos + left_dpos, *LEFT_CARTESIAN_BOUNDS)

        left_quat = self._data.mocap_quat[0].copy()
        left_dquat = R.from_euler("xyz", self.limit_offset_norm(
            left_tcp_euler_delta*self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)
        )
        left_nquat = (left_dquat * R.from_quat(left_quat, scalar_first=True)).as_quat(scalar_first=True)

        right_pos = self._data.mocap_pos[1].copy()
        right_dpos = self.limit_offset_norm(
            right_tcp_pos_delta*self.MAX_LINEAR_VELOCITY, self.MAX_LINEAR_VELOCITY
        )
        right_npos = np.clip(right_pos + right_dpos, *RIGHT_CARTESIAN_BOUNDS)
        right_quat = self._data.mocap_quat[1].copy()
        right_dquat = R.from_euler("xyz", self.limit_offset_norm(
            right_tcp_euler_delta*self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)
        )
        right_nquat = (right_dquat * R.from_quat(right_quat, scalar_first=True)).as_quat(scalar_first=True)

        self.ik_controller.set_targets(left_npos, left_nquat, right_npos, right_nquat)

        # Set gripper grasp.
        left_g = self._data.ctrl[self._gripper_ctrl_ids[0]] / 255
        left_dg = action[6] * 0.1
        left_ng = np.clip(left_g + left_dg, 0.0, 1.0)
        right_g = self._data.ctrl[self._gripper_ctrl_ids[1]] / 255
        right_dg = action[13] * 0.1
        right_ng = np.clip(right_g + right_dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_ids[0]] = left_ng * 255
        self._data.ctrl[self._gripper_ctrl_ids[1]] = right_ng * 255

        obs = self._compute_observation()
        # rew = self._compute_reward()
        # terminated = self.time_limit_exceeded()

        self.gym_rate.sleep()
        return obs, 0, False, False, {}

    def render(self):
        rendered_frames = []
        for cam_name in self.camera_names:
            self._renderer.update_scene(self.data, camera=cam_name)
            rendered_frames.append(self._renderer.render())
        return rendered_frames

    def _compute_observation(self) -> dict:
        # IMPORTANT NOTE:
        # in observation, the quat from mujoco is scalar first, but we should keep it scalar last
        obs = {}
        obs["state"] = {}

        with self.ik_controller.lock:
            for side in ["left", "right"]:
                # in world frame
                tcp_pos = self._data.sensor(f"{side}/tcp_pos").data
                tcp_quat = np.roll(self._data.sensor(f"{side}/tcp_quat").data, -1)
                obs["state"][f"{side}/tcp_pose"] = np.concatenate([tcp_pos, tcp_quat]).astype(np.float32)
                tcp_vel = self._data.sensor(f"{side}/tcp_vel").data
                tcp_angvel = self._data.sensor(f"{side}/tcp_angvel").data
                obs["state"][f"{side}/tcp_vel"] = np.concatenate([tcp_vel, tcp_angvel]).astype(np.float32)

                # in head camera frame
                ego_tcp_pos = self._data.sensor(f"{side}/ego_tcp_pos").data
                ego_tcp_quat = np.roll(self._data.sensor(f"{side}/ego_tcp_quat").data, -1)
                obs["state"][f"{side}/ego_tcp_pose"] = np.concatenate([ego_tcp_pos, ego_tcp_quat]).astype(np.float32)
                ego_tcp_vel = self._data.sensor(f"{side}/ego_tcp_vel").data
                ego_tcp_angvel = self._data.sensor(f"{side}/ego_tcp_angvel").data
                obs["state"][f"{side}/ego_tcp_vel"] = np.concatenate([ego_tcp_vel, ego_tcp_angvel]).astype(np.float32)

            # gripper pos
            obs["state"]["left/gripper_pos"] = np.array(
                self._data.ctrl[self._gripper_ctrl_ids[0]] / 255, dtype=np.float32)
            obs["state"]["right/gripper_pos"] = np.array(
                self._data.ctrl[self._gripper_ctrl_ids[1]] / 255, dtype=np.float32)

        if self.image_obs:
            obs["images"] = {}
            images = self.render()
            for cam_name in self.camera_names:
                obs["images"][cam_name] = images.pop(0)

        # else:
        #     block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        #     obs["state"]["block_pos"] = block_pos

        return obs

    def _compute_reward(self) -> float:
        # block_pos = self._data.sensor("block_pos").data
        # tcp_pos = self._data.sensor("2f85/pinch_pos").data
        # dist = np.linalg.norm(block_pos - tcp_pos)
        # r_close = np.exp(-20 * dist)
        # r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        # r_lift = np.clip(r_lift, 0.0, 1.0)
        # rew = 0.3 * r_close + 0.7 * r_lift
        # return rew
        return 0

    def close(self):
        if self.render_mode == "human":
            self._viewer.close()
        self.ik_controller.stop()
        self.ik_thread.join()
        super().close()

    def limit_offset_norm(self, offset, max_offset):
        # scale offset such that the max norm of offset is max_offset
        norm = np.linalg.norm(offset)
        if norm > max_offset:
            offset = offset / norm * max_offset
        return offset

from tqdm import tqdm

if __name__ == "__main__":
    env = DualXarmsGymEnv(render_mode="human")
    from dual_xarms_sim.relative_frame import RelativeFrame
    from dual_xarms_sim.oculus_intervention import OculusIntervention

    env = OculusIntervention(env, freq=10)
    env = RelativeFrame(env)

    obs, _ = env.reset()
    obses = [obs]

    for i in tqdm(range(100000)):
        action = env.action_space.sample() * 0
        obs, _, _, _, _ = env.step(action)

    #     obses.append(obs)

    # with open("obses.npy", "wb") as f:
    #     np.save(f, obses, allow_pickle=True)

    env.ik_controller.human_viewer.close()
    env.ik_controller.stop()
    env.ik_thread.join()
