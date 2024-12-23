from xarm.wrapper import XArmAPI
from tqdm import tqdm
import numpy as np
from loop_rate_limiters import RateLimiter

rate = RateLimiter(200)
ip = "192.168.1.199"
arm = XArmAPI(ip, is_radian=True)
arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(0)




for i in tqdm(range(100000)):
    status, angles = arm.get_servo_angle(is_radian=True)
    angles = np.asarray(angles)
    angles[5] -= 0.01
    arm.set_servo_angle_j(angles=angles, speed=0.1, is_radian=True)
    rate.sleep()
