import threading
import mink
import numpy as np
import mujoco
from loop_rate_limiters import RateLimiter

class RealIKController:
    def __init__(self, left_arm, right_arm, max_angular_vel, model, data, configuration, actuator_ids, dof_ids,
                    tasks, l_ee_task, r_ee_task,
                    ik_solver, ik_limits, ik_max_iters=2,
                    pos_threshold = 1e-2, ori_threshold = 1e-2, damping=1e-5, frequency=200, human_viewer=None):
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.max_angular_vel = max_angular_vel
        self.model = model
        self.data = data
        self.configuration = configuration
        self.actuator_ids = actuator_ids
        self.dof_ids = dof_ids
        self.tasks = tasks
        self.l_ee_task = l_ee_task
        self.r_ee_task = r_ee_task
        self.ik_solver = ik_solver
        self.ik_limits = ik_limits
        self.ik_max_iters = ik_max_iters
        self.pos_threshold = pos_threshold
        self.ori_threshold = ori_threshold
        self.damping = damping
        self.rate = RateLimiter(frequency)
        self.alpha = 0.999
        self.prev_ema = np.zeros(len(dof_ids))  # Initialize EMA values for each DOF
        self.human_viewer = human_viewer
        self.running = False
        self.lock = threading.Lock()  # Lock for thread-safe operations

    def update_ema(self, current_angles):
        # Update the EMA for each joint angle
        self.prev_ema = self.alpha * current_angles + (1 - self.alpha) * self.prev_ema
        return self.prev_ema

    def set_targets(self, l_pos, l_quat, r_pos, r_quat):
        self.data.mocap_pos[0], self.data.mocap_quat[0] = l_pos, l_quat
        self.data.mocap_pos[1], self.data.mocap_quat[1] = r_pos, r_quat
        self.l_ee_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "left/target"))
        self.r_ee_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "right/target"))

    def run_ik(self):
        self.running = True
        while True:
            while self.running:
                with self.lock:
                    status, left_qpos = self.left_arm.get_servo_angle(is_radian=True)
                    if status == 0:
                        self.data.qpos[self.dof_ids[:7]] = left_qpos
                    else:
                        print(f"Failed to get left arm servo angle: {status}")
                    status, right_qpos = self.right_arm.get_servo_angle(is_radian=True)
                    if status == 0:
                        self.data.qpos[self.dof_ids[-7:]] = right_qpos
                    else:
                        print(f"Failed to get right arm servo angle: {status}")

                    self.configuration.update(self.data.qpos)
                    mujoco.mj_step(self.model, self.data)

                    for i in range(self.ik_max_iters):
                        try:
                            vel = mink.solve_ik(
                                self.configuration,
                                self.tasks,
                                self.rate.dt,
                                self.ik_solver,
                                limits=self.ik_limits,
                                damping=self.damping,
                            )
                            self.configuration.integrate_inplace(vel, self.rate.dt)
                        except Exception as e:
                            print(f"Failed to solve IK: {e}")
                            break

                        l_err = self.l_ee_task.compute_error(self.configuration)
                        r_err = self.r_ee_task.compute_error(self.configuration)
                        if np.linalg.norm(l_err[:3]) <= self.pos_threshold and np.linalg.norm(l_err[3:]) <= self.ori_threshold \
                            and np.linalg.norm(r_err[:3]) <= self.pos_threshold and np.linalg.norm(r_err[3:]) <= self.ori_threshold:
                            break

                    ema_angles = self.update_ema(self.configuration.q[self.dof_ids])
                    self.left_arm.set_servo_angle_j(
                        angles=ema_angles[:7], is_radian=True,
                    )
                    self.right_arm.set_servo_angle_j(
                        angles=ema_angles[-7:], is_radian=True,
                    )

                if self.human_viewer and self.human_viewer.is_running():
                    self.human_viewer.sync()
                self.rate.sleep()

    def start(self):
        with self.lock:
            self.running = True

    def stop(self):
        with self.lock:
            self.running = False
