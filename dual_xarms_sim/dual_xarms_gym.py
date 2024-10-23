from pathlib import Path
from typing import Any, Literal, Tuple, Dict
import time
import threading

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
import mink
from loop_rate_limiters import RateLimiter

from dual_xarms_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from dual_xarms_sim.ik_controller import IKController

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "ufactory_xarm7" / "dual_scene.xml"

# _PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
LEFT_CARTESIAN_BOUNDS = np.asarray([[-0.7, 0.2, 0], [0.2, 0.6, 0.3]])
RIGHT_CARTESIAN_BOUNDS = np.asarray([[-0.2, 0.2, 0], [0.7, 0.6, 0.3]])
# _SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])

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
_HOME_JOINT_QPOS = np.array([0, -0.247, 0, 0.909, 0, 1.15644, 0, 0, -0.247, 0, 0.909, 0, 1.15644, 0])
_HOME_JOINT_CTRL = np.array([0.785398163, -0.247, 0, 0.909, 0, 1.15644, 0, 0])

class DualXarmsGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.2, 1]),
        seed: int = 0,
        control_dt: float = 0.01, # 0.005, #200Hz
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        # self.camera_id = (0, 1)
        # self.image_obs = image_obs

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

        self.observation_space = gym.spaces.Dict(
            {
                # TODO: add tcp eulers for both sides
                "state": gym.spaces.Dict(
                    {
                        "left/tcp_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "left/tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "left/gripper_pos": spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float32
                        ),
                        "right/tcp_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "right/tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "right/gripper_pos": spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float32
                        ),
                        # "block_pos": spaces.Box(
                        #     -np.inf, np.inf, shape=(3,), dtype=np.float32
                        # ),
                    }
                ),
            }
        )

        # left tcp pos delta, left tcp euler delta, left gripper pos,
        # right tcp pos delta, right tcp euler delta, right gripper pos
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        # from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        # self._viewer = MujocoRenderer(
        #     self.model,
        #     self.data,
        # )
        # self._viewer.render(self.render_mode)
        if self.render_mode == "human":
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)

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
        self.ik_solver = "quadprog"
        self.pos_threshold = 1e-2
        self.ori_threshold = 1e-2
        self.ik_max_iters = 2
        self.ik_rate = RateLimiter(frequency=200.0)
        self.ik_controller = IKController(
            self._model,
            self._data,
            self.ik_configuration,
            self.arm_actuator_ids,
            self.arm_dof_ids,
            self.tasks,
            self.l_ee_task,
            self.r_ee_task,
            self.ik_solver,
            self.ik_limits,
            self.ik_max_iters,
            self.pos_threshold,
            self.ori_threshold,
            damping=1e-5,
            rate=self.ik_rate,
            human_viewer=self._viewer,
        )
        self.ik_thread = threading.Thread(target=self.ik_controller.run_ik, daemon=True)
        self.gym_rate = RateLimiter(frequency=10.0)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self.arm_dof_ids] = _HOME_JOINT_QPOS
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        self._data.mocap_pos[0] = self._data.sensor("left/tcp_pos").data
        self._data.mocap_quat[0] = self._data.sensor("left/tcp_quat").data
        self._data.mocap_pos[1] = self._data.sensor("right/tcp_pos").data
        self._data.mocap_quat[1] = self._data.sensor("right/tcp_quat").data
        mujoco.mj_forward(self._model, self._data)

        # Sample a new block position.
        # block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        # self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        # mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        # self._z_init = self._data.sensor("block_pos").data[2]
        # self._z_success = self._z_init + 0.2

        with self.ik_controller.lock:
            self.ik_configuration.update(self._data.qpos)
            self.posture_task.set_target_from_configuration(self.ik_configuration)
            self.ik_controller.set_targets(
                mink.SE3.from_mocap_name(self.model, self.data, "left/target"),
                mink.SE3.from_mocap_name(self.model, self.data, "right/target"),
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
        left_gripper_pos = action[6]
        right_tcp_pos_delta = action[7:10]
        right_tcp_euler_delta = action[10:13]
        right_gripper_pos = action[13]

        # # Set the mocap position.
        left_pos = self._data.mocap_pos[0].copy()
        left_dpos = np.asarray(left_tcp_pos_delta) * self._action_scale[0]
        left_npos = np.clip(left_pos + left_dpos, *LEFT_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = left_npos

        right_pos = self._data.mocap_pos[1].copy()
        right_dpos = np.asarray(right_tcp_pos_delta) * self._action_scale[0]
        right_npos = np.clip(right_pos + right_dpos, *RIGHT_CARTESIAN_BOUNDS)
        self._data.mocap_pos[1] = right_npos

        # Update task targets based on current mocap positions
        self.ik_controller.set_targets(
            mink.SE3.from_mocap_name(self._model, self._data, "left/target"),
            mink.SE3.from_mocap_name(self._model, self._data, "right/target")
        )

        # # Set gripper grasp.
        # g = self._data.ctrl[self._gripper_ctrl_id] / 255
        # dg = grasp * self._action_scale[1]
        # ng = np.clip(g + dg, 0.0, 1.0)
        # self._data.ctrl[self._gripper_ctrl_id] = ng * 255
        # self._data.ctrl[self._panda_ctrl_ids] = tau

        # self.data.ctrl[self.arm_actuator_ids] = self.ik_configuration.q[self.arm_dof_ids]
        # mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        # rew = self._compute_reward()
        # terminated = self.time_limit_exceeded()

        self.gym_rate.sleep()
        # return obs, rew, terminated, False, {}

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        with self.ik_controller.lock:
            # TODO: add tcp eulers for both sides
            tcp_pos = self._data.sensor("left/tcp_pos").data
            obs["state"]["left/tcp_pos"] = tcp_pos.astype(np.float32)
            tcp_quat = self._data.sensor("left/tcp_quat").data
            # obs["state"]["left/tcp_euler"] = tcp_quat.astype(np.float32)
            tcp_vel = self._data.sensor("left/tcp_vel").data
            obs["state"]["left/tcp_vel"] = tcp_vel.astype(np.float32)
            obs["state"]["left/gripper_pos"] = np.array(
                self._data.ctrl[self._gripper_ctrl_ids[0]] / 255, dtype=np.float32
            )

            tcp_pos = self._data.sensor("right/tcp_pos").data
            obs["state"]["right/tcp_pos"] = tcp_pos.astype(np.float32)
            tcp_quat = self._data.sensor("right/tcp_quat").data
            # obs["state"]["right/tcp_euler"] = tcp_quat.astype(np.float32)
            tcp_vel = self._data.sensor("right/tcp_vel").data
            obs["state"]["right/tcp_vel"] = tcp_vel.astype(np.float32)
            obs["state"]["right/gripper_pos"] = np.array(
                self._data.ctrl[self._gripper_ctrl_ids[1]] / 255, dtype=np.float32
            )

        # if self.image_obs:
        #     obs["images"] = {}
        #     obs["images"]["front"], obs["images"]["wrist"] = self.render()
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
        self.ik_controller.stop()
        self.ik_thread.join()
        super().close()


import requests

def get_controller_velocity(controller_key):
    url = f"http://127.0.0.1:8000/velocity/{controller_key}"
    # start_time = time.time()
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        velocity_data = response.json()
        # end_time = time.time()
        # print(f"Time taken to get velocity data: {end_time - start_time}")
        return velocity_data
    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"OOps: Something Else: {err}")

from tqdm import tqdm
import logging

# Set the logging level to ERROR, which ignores WARNING messages
# logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
    env = DualXarmsGymEnv(render_mode="human")
    env.reset()

    for i in tqdm(range(100000000)):
        action = env.action_space.sample() * 0
        # left_data = get_controller_velocity("left")
        # left_xyz = np.array([left_data["x"], left_data["y"], left_data["z"]])
        # right_data = get_controller_velocity("right")
        # right_xyz = np.array([right_data["x"], right_data["y"], right_data["z"]])

        # left_gripper = left_data["left_trigger"]
        # right_gripper = right_data["right_trigger"]

        # action[:3] = left_xyz * 0
        # action[7:10] = right_xyz * 0
        env.step(action)
