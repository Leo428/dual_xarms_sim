import threading
import mink
import numpy as np
import mujoco

class IKController:
    def __init__(self, model, data, configuration, actuator_ids, dof_ids,
                    tasks, l_ee_task, r_ee_task,
                    ik_solver, ik_limits, ik_max_iters=2,
                    pos_threshold = 1e-2, ori_threshold = 1e-2, damping=1e-5, rate=None, human_viewer=None):
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
        self.rate = rate
        self.human_viewer = human_viewer
        self.running = False
        self.lock = threading.Lock()  # Lock for thread-safe operations

    def set_targets(self, l_target, r_target):
        self.l_ee_task.set_target(l_target)
        self.r_ee_task.set_target(r_target)

    def run_ik(self):
        self.running = True
        while self.running:
            with self.lock:
                for i in range(self.ik_max_iters):
                    vel = mink.solve_ik(
                        self.configuration,
                        self.tasks,
                        self.rate.dt,
                        self.ik_solver,
                        limits=self.ik_limits,
                        damping=self.damping,
                    )
                    self.configuration.integrate_inplace(vel, self.rate.dt)

                    l_err = self.l_ee_task.compute_error(self.configuration)
                    r_err = self.r_ee_task.compute_error(self.configuration)
                    if np.linalg.norm(l_err[:3]) <= self.pos_threshold and np.linalg.norm(l_err[3:]) <= self.ori_threshold \
                        and np.linalg.norm(r_err[:3]) <= self.pos_threshold and np.linalg.norm(r_err[3:]) <= self.ori_threshold:
                        break
                self.data.ctrl[self.actuator_ids] = self.configuration.q[self.dof_ids]
                mujoco.mj_step(self.model, self.data)

            if self.human_viewer and self.human_viewer.is_running():
                self.human_viewer.sync()
            self.rate.sleep()

    def stop(self):
        self.running = False
