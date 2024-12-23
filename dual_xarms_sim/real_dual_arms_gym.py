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
from scipy.spatial.transform import Rotation as R

from xarm.wrapper import XArmAPI

from dual_xarms_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from dual_xarms_sim.real_ik_controller import RealIKController

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "ufactory_xarm7" / "dual_scene.xml"

# 75.5-18 = 57.5 / 2 = 28.75
LEFT_HOME = np.asarray([-0.2875, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0])
RIGHT_HOME = np.asarray([0.2875, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0])
LEFT_CARTESIAN_BOUNDS = np.asarray([[-0.2875 - 0.3, 0.2, 0], [-0.2875 + 0.3, 0.6, 0.5]])
# LEFT_EULER_BOUNDS = np.asarray([[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]])
RIGHT_CARTESIAN_BOUNDS = np.asarray([[0.2875 - 0.3, 0.2, 0], [0.2875 + 0.3, 0.6, 0.5]])
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
_VELOCITY_LIMITS = {k: np.pi/4 for k in _JOINT_NAMES}
_VELOCITY_LIMITS["joint7"] = np.pi/2

_HOME_JOINT_QPOS = np.array([0, -0.25891, -0.00020, 1.03223, 0, 1.31830, 0, 0, -0.25891, -0.00020, 1.03223, 0, 1.31830, 0])
_MAX_LINEAR_VELOCITY = 0.75 # m/s
_MAX_ANGULAR_VELOCITY = np.pi/4 # rad/s

class RealDualXarmsGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_freq: int = 20, # 20 Hz
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(height=224, width=224),
        render_mode: Literal["rgb_array", "human"] = "human",
        image_obs: bool = True, run_ik: bool = True,
    ):
        self.control_freq = control_freq
        self.MAX_LINEAR_VELOCITY = _MAX_LINEAR_VELOCITY / control_freq
        self.MAX_ANGULAR_VELOCITY = _MAX_ANGULAR_VELOCITY / control_freq
        self._action_scale = action_scale
        self.gym_rate = RateLimiter(frequency=control_freq, name="gym_rate")

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
                    "left/relative2_tcp_pose": spaces.Box( # relative to right tcp, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/relative2_tcp_vel": spaces.Box( # relative to right tcp, linear + angular euler
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
                    "right/tcp_vel": spaces.Box( # world frame
                        -np.inf, np.inf, shape=(6,), dtype=np.float32 # linear + angular euler
                    ),
                    "right/ego_tcp_pose": spaces.Box( # head camera frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/ego_tcp_vel": spaces.Box( # head camera frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "right/relative2_tcp_pose": spaces.Box( # relative to left tcp, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/relative2_tcp_vel": spaces.Box( # relative to left tcp, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "right/gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                    "right/joint_qpos": spaces.Box(
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
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

        # self._renderer = Renderer(self.model, width=render_spec.width, height=render_spec.height)

        # initialize robot arms
        self.left_arm_ip = "192.168.1.221"
        self.left_arm = XArmAPI(port=self.left_arm_ip, is_radian=True)
        self.right_arm_ip = "192.168.1.199"
        self.right_arm = XArmAPI(port=self.right_arm_ip, is_radian=True)

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
        self.posture_task = mink.PostureTask(self.model, cost=1e-2)
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
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )
        self.ik_limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(self.model, self.velocity_limits),
            collision_avoidance_limit,
        ]
        self.ik_rate = 200.0
        self.ik_controller = RealIKController(
            left_arm=self.left_arm, right_arm=self.right_arm, max_angular_vel=_MAX_ANGULAR_VELOCITY,
            model=self._model, data=self._data,
            configuration=self.ik_configuration,
            actuator_ids=self.arm_actuator_ids, dof_ids=self.arm_dof_ids,
            tasks=self.tasks, l_ee_task=self.l_ee_task, r_ee_task=self.r_ee_task,
            ik_solver="quadprog", ik_limits=self.ik_limits,
            ik_max_iters=2, pos_threshold=1e-2, ori_threshold=1e-2,
            damping=1e-5, frequency=self.ik_rate, human_viewer=self._viewer,
        )
        self.ik_thread = threading.Thread(target=self.ik_controller.run_ik, daemon=True)
        self.run_ik = run_ik
        self.is_ik_thread_running = False

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        time.sleep(0.1)
        self.left_arm.motion_enable(enable=True)
        self.right_arm.motion_enable(enable=True)
        self.left_arm.set_mode(0)
        self.right_arm.set_mode(0)
        self.left_arm.set_state(state=0)
        self.right_arm.set_state(state=0)
        self.left_arm.set_gripper_mode(0)
        self.right_arm.set_gripper_mode(0)
        self.left_arm.set_gripper_enable(True)
        self.right_arm.set_gripper_enable(True)
        self.left_arm.set_gripper_position(840, wait=True, speed=8000)
        self.right_arm.set_gripper_position(840, wait=True, speed=8000)
        time.sleep(0.1)
        self.left_arm.set_servo_angle(angle=_HOME_JOINT_QPOS[:7], speed=0.2, is_radian=True, wait=True)
        self.right_arm.set_servo_angle(angle=_HOME_JOINT_QPOS[-7:], speed=0.2, is_radian=True, wait=True)
        time.sleep(0.1)
        self.left_arm.set_mode(1) # 1: servo joint position mode
        self.right_arm.set_mode(1) # 1: servo joint position mode
        self.left_arm.set_state(state=0)
        self.right_arm.set_state(state=0)
        time.sleep(0.1)

        """Reset the environment."""
        self.ik_controller.stop()
        time.sleep(0.1)

        super().reset(seed=seed, **kwargs)
        with self.ik_controller.lock:
            mujoco.mj_resetData(self._model, self._data)

            # Reset arm to home position.
            # self._data.qpos[self.arm_dof_ids] = _HOME_JOINT_QPOS
            status, left_qpos = self.left_arm.get_servo_angle(is_radian=True)
            if status == 0:
                self._data.qpos[self.arm_dof_ids[:7]] = left_qpos
            else:
                print(f"Failed to get left arm servo angle: {status}")
            status, right_qpos = self.right_arm.get_servo_angle(is_radian=True)
            if status == 0:
                self._data.qpos[self.arm_dof_ids[-7:]] = right_qpos
            else:
                print(f"Failed to get right arm servo angle: {status}")
            self.ik_configuration.update(self.data.qpos)
            mujoco.mj_forward(self._model, self._data)

            # Reset mocap body to home position.
            self._data.mocap_pos[0], self._data.mocap_quat[0] = LEFT_HOME[:3], LEFT_HOME[3:]
            self._data.mocap_pos[1], self._data.mocap_quat[1] = RIGHT_HOME[:3], RIGHT_HOME[3:]
            mujoco.mj_forward(self._model, self._data)

            # Sample a new block position.
            block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            self._data.jnt("block").qpos[:3] = (*block_xy, 0.02)
            mujoco.mj_forward(self._model, self._data)

            self.ik_configuration.update(self._data.qpos)
            self.posture_task.set_target_from_configuration(self.ik_configuration)
            self.ik_controller.set_targets(
                LEFT_HOME[:3], LEFT_HOME[3:], RIGHT_HOME[:3], RIGHT_HOME[3:]
            )
            if not self.is_ik_thread_running and self.run_ik:
                self.ik_thread.start()
                self.is_ik_thread_running = True

        self.ik_controller.start()
        time.sleep(0.1)

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
        # gripper range on the real arms is between 80 and 840, 0 is open, 1 is closed
        self.left_arm.set_gripper_position(int(80 + (1-left_ng) * (840 - 80)), wait=False, speed=8000)
        self.right_arm.set_gripper_position(int(80 + (1-right_ng) * (840 - 80)), wait=False, speed=8000)

        obs = self._compute_observation()
        rew = self._compute_reward()
        # terminated = self.time_limit_exceeded()
        done = True if rew == 4.0 else False

        self.gym_rate.sleep()
        # time.sleep(0.005)
        return obs, rew, done, False, {}

    def render(self):
        rendered_frames = []
        # for cam_name in self.camera_names:
        #     self._renderer.update_scene(self.data, camera=cam_name)
        #     rendered_frames.append(self._renderer.render())
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

                # relative to the other side tcp
                wrt2other_tcp_pos = self._data.sensor(f"{side}/relative2_tcp_pos").data
                wrt2other_tcp_quat = np.roll(self._data.sensor(f"{side}/relative2_tcp_quat").data, -1)
                obs["state"][f"{side}/relative2_tcp_pose"] = np.concatenate([wrt2other_tcp_pos, wrt2other_tcp_quat]).astype(np.float32)
                wrt2other_tcp_vel = self._data.sensor(f"{side}/relative2_tcp_vel").data
                wrt2other_tcp_angvel = self._data.sensor(f"{side}/relative2_tcp_angvel").data
                obs["state"][f"{side}/relative2_tcp_vel"] = np.concatenate([wrt2other_tcp_vel, wrt2other_tcp_angvel]).astype(np.float32)

                # joint qpos
                joint_qpos = self._data.qpos[self.arm_dof_ids]
                obs["state"][f"{side}/joint_qpos"] = joint_qpos[:7] if side == "left" else joint_qpos[7:]

            # gripper pos
            obs["state"]["left/gripper_pos"] = np.array(
                self._data.ctrl[self._gripper_ctrl_ids[0]] / 255, dtype=np.float32)
            obs["state"]["right/gripper_pos"] = np.array(
                self._data.ctrl[self._gripper_ctrl_ids[1]] / 255, dtype=np.float32)

        if self.image_obs:
            obs["images"] = {}
            # images = self.render()
            # for cam_name in self.camera_names:
            #     obs["images"][cam_name] = images.pop(0)

        # else:
        #     block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        #     obs["state"]["block_pos"] = block_pos

        return obs

    def _compute_reward(self) -> float:
        # Check if the block is in contact with the gripper
        all_contact_pairs = []
        with self.ik_controller.lock:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                contact_pair = (self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name)
                all_contact_pairs.append(contact_pair)

        cube_held_left = (("left/left_pad", "cube") in all_contact_pairs or \
                            ("left/left_pad_lower", "cube") in all_contact_pairs) and \
                            (("left/right_pad", "cube") in all_contact_pairs or \
                            ("left/right_pad_lower", "cube") in all_contact_pairs)
        cube_held_right = (("right/left_pad", "cube") in all_contact_pairs or \
                            ("right/left_pad_lower", "cube") in all_contact_pairs) and \
                            (("right/right_pad", "cube") in all_contact_pairs or \
                            ("right/right_pad_lower", "cube") in all_contact_pairs)
        cube_on_floor = ("floor", "cube") in all_contact_pairs

        if cube_held_right and (not cube_held_left):
            if cube_on_floor:
                return 1.0
            return 2.0
        elif cube_held_right and cube_held_left and (not cube_on_floor):
            return 3.0
        elif cube_held_left and not cube_held_right and (not cube_on_floor):
            
            return 4.0

        return 0.0

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
    env = RealDualXarmsGymEnv(control_freq=50, render_mode="human")
    from dual_xarms_sim.relative_frame import RelativeFrame
    from dual_xarms_sim.oculus_intervention import OculusIntervention

    try:
        env = OculusIntervention(env, freq=50)
        # env = RelativeFrame(env)

        obs, _ = env.reset()
        # obses = [obs]

        for i in tqdm(range(100000)):
            action = env.action_space.sample() * 0
            obs, rew, done, _, info = env.step(action)
            if "intervene_action" in info:
                action = info["intervene_action"]
            # print(rew, done)

    except KeyboardInterrupt:
        env.close()
