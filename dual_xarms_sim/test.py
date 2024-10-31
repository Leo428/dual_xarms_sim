from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "ufactory_xarm7" / "dual_scene.xml"

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

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Adjust joint names for both left and right arms
    joint_names = []
    velocity_limits = {}
    for prefix in ["left", "right"]:
        for n in _JOINT_NAMES:
            name = f"{prefix}/{n}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[n]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    # Task definitions using mink library
    tasks = [
        l_ee_task := mink.FrameTask(
            frame_name="left/link_tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        r_ee_task := mink.FrameTask(
            frame_name="right/link_tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-4),
    ]

    # Fetch geometry IDs for collision avoidance
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/link7").id)  # Left end-effector
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/link7").id)  # Right end-effector
    l_upper_arm_geoms = mink.get_subtree_geom_ids(model, model.body("left/link1").id)  # Left upper arm
    r_upper_arm_geoms = mink.get_subtree_geom_ids(model, model.body("right/link1").id)  # Right upper arm

    # Define geometry IDs for the environment if needed (example: table or frames)
    # You would need to define these based on your actual environment setup
    table_geoms = ["floor"]  # Placeholder, replace with actual geom ID(s)

    # Define collision pairs
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),  # Avoid collisions between the left and right end-effectors
        (l_upper_arm_geoms + r_upper_arm_geoms, table_geoms),  # Avoid collisions between arms and the table
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.02,
        collision_detection_distance=0.1,
    )

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]

    solver = "quadprog"
    pos_threshold = 1e-2
    ori_threshold = 1e-2
    max_iters = 2

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize the neutral pose
        # mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        # configuration.update(data.qpos)
        data.qpos[actuator_ids] = np.array([0, -0.25844, -0.00013, 1.03062, -0.00006, 1.31739, 0, 0, -0.25844, -0.00013, 1.03062, 0.00006, 1.31739, 0])
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets based on current mocap positions
            l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(vel, rate.dt)

                l_err = l_ee_task.compute_error(configuration)
                r_err = r_ee_task.compute_error(configuration)
                if np.linalg.norm(l_err[:3]) <= pos_threshold and np.linalg.norm(l_err[3:]) <= ori_threshold \
                   and np.linalg.norm(r_err[:3]) <= pos_threshold and np.linalg.norm(r_err[3:]) <= ori_threshold:
                    break

            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()
