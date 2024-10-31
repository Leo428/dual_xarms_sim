from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from copy import deepcopy

from dual_xarms_sim.utils.transformation import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "left/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                "right/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 14 DoF action space with (l_dx, l_dy, l_dz, l_drx, l_dry, l_drz, l_dgrip, ...).
    By convention, the 7th and 14th dimension of the action space is used for the gripper.

    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrix = {
            "left": np.zeros((6, 6)),
            "right": np.zeros((6, 6)),
        }

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = {
                "left": np.zeros((4, 4)),
                "right": np.zeros((4, 4)),
            }
        
        self.observation_space = deepcopy(env.observation_space)
        for side in ["left", "right"]:
            if self.include_relative_pose:
                # Add relative pose to observation space
                self.observation_space["state"][f"{side}/wrist_tcp_pose"] = spaces.Box(
                    -np.inf, np.inf, shape=(7,)
                )
            self.observation_space["state"][f"{side}/wrist_tcp_vel"] = spaces.Box(
                -np.inf, np.inf, shape=(6,)
            )

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)

        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        # Update adjoint matrix
        self.adjoint_matrix = {
            "left": construct_adjoint_matrix(obs["state"]["left/tcp_pose"]),
            "right": construct_adjoint_matrix(obs["state"]["right/tcp_pose"]),
        }

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for side in ["left", "right"]:
            obs["state"][f"{side}/wrist_tcp_pose"] = obs["state"][f"{side}/tcp_pose"]
            obs["state"][f"{side}/wrist_tcp_vel"] = obs["state"][f"{side}/tcp_vel"]

            # Update adjoint matrix
            self.adjoint_matrix[side] = construct_adjoint_matrix(
                obs["state"][f"{side}/wrist_tcp_pose"]
            )   
            if self.include_relative_pose:
                # Update transformation matrix from the reset pose's relative frame to base frame
                self.T_r_o_inv[side] = np.linalg.inv(
                    construct_homogeneous_matrix(obs["state"][f"{side}/wrist_tcp_pose"])
                )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        for side in ["left", "right"]:
            adjoint_inv = np.linalg.inv(self.adjoint_matrix[side])
            obs["state"][f"{side}/wrist_tcp_vel"] = adjoint_inv @ obs["state"][f"{side}/tcp_vel"]

            if self.include_relative_pose:
                T_b_o = construct_homogeneous_matrix(obs["state"][f"{side}/tcp_pose"])
                T_b_r = self.T_r_o_inv[side] @ T_b_o

                # Reconstruct transformed tcp_pose vector
                p_b_r = T_b_r[:3, 3]
                theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
                obs["state"][f"{side}/wrist_tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix
        """
        # left arm
        action[:6] = self.adjoint_matrix["left"] @ action[:6]
        # right arm
        action[7:13] = self.adjoint_matrix["right"] @ action[7:13]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action[:6] = np.linalg.inv(self.adjoint_matrix["left"]) @ action[:6]
        action[7:13] = np.linalg.inv(self.adjoint_matrix["right"]) @ action[7:13]
        return action
