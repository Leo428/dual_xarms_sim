from typing import Dict
from collections import OrderedDict
from loop_rate_limiters import RateLimiter
import cv2
from tqdm import tqdm
import zmq

_REALSENSE_CAMERAS = OrderedDict({
    "left/wrist": "130322273478",
    "right/wrist": "128422272097",
})

import queue
import threading
import time

import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import pyzed.sl as sl

class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 360), fps=60, depth=False):
        self.name = name
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.device = self.profile.get_device()
        self.sensor = self.device.first_depth_sensor()
        # self.sensor.set_option(rs.option.frames_queue_size, 1)
        self.sensor.set_option(rs.option.enable_auto_exposure, True)
        self.sensor.set_option(rs.option.enable_auto_white_balance, True)
        # self.sensor.set_option(rs.option.exposure, 10000)
        # self.sensor.set_option(rs.option.white_balance, 4000)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        try:
            frames = self.pipe.wait_for_frames()
        except Exception as e:
            print(self.name, e)
            return False, None
        # color_frame = frames.get_color_frame()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        # if self.depth:
        #     depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data()).tobytes()
            # if self.depth and depth_frame.is_depth_frame():
            #     depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
            #     return True, image, depth
            # else:
            #     return True, image
            return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()

if __name__ == "__main__":
    zmq_context = zmq.Context()
    image_pub = zmq_context.socket(zmq.PUB)
    image_pub.setsockopt(zmq.SNDHWM, 1)
    image_pub.bind("tcp://127.0.0.1:5004")

    time.sleep(1)
    ctx = rs.context()
    devices = ctx.query_devices()
    [print(device) for device in devices]
    for dev in devices:
        dev.hardware_reset()
        time.sleep(2)

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 60  # Set fps at 30
    init_params.camera_image_flip = sl.FLIP_MODE.ON
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.sdk_verbose = 0

    # Open the camera
    err = zed.open(init_params)
    print(f"error: {err}")
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get camera information (ZED serial number)
    zed_serial = zed.get_camera_information().serial_number
    print("Hello! This is my serial number: {0}".format(zed_serial))

    # Create a dictionary of RealSense cameras
    cameras = OrderedDict()
    for name, serial_number in _REALSENSE_CAMERAS.items():
        print(f"Initializing {name} camera with serial number {serial_number}")
        cameras[name] = RSCapture(name, serial_number, dim=(640, 360), depth=False)

    # # Start a thread to display the frames
    # displayer = ImageDisplayer(frames_queue)
    # displayer.start()

    rate = RateLimiter(frequency=60)
    # bar = tqdm(total=1e8)
    try:
        while True:
            tmp_dict = {}
            for name, camera in cameras.items():
                success, rgb = camera.read()
                if success:
                    tmp_dict[name] = rgb

            left_top_frame = sl.Mat()
            right_top_frame = sl.Mat()
            # Grab an image, a RuntimeParameters object must be given to grab()
            runtime_parameters = sl.RuntimeParameters()
            runtime_parameters.enable_depth = False
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # original numpy array, not a copy (H, W, 4 (bgra))
                zed.retrieve_image(left_top_frame, sl.VIEW.LEFT)
                zed.retrieve_image(right_top_frame, sl.VIEW.RIGHT)
                # (720, 1280, 3) --> (360, 640, 3)
                left_top_frame = cv2.resize(left_top_frame.get_data()[..., :3], (640, 360))
                right_top_frame = cv2.resize(right_top_frame.get_data()[..., :3], (640, 360))
                left_top_frame = left_top_frame.tobytes()
                right_top_frame = right_top_frame.tobytes()
                tmp_dict["left/top"] = left_top_frame
                tmp_dict["right/top"] = right_top_frame

            if len(tmp_dict) == len(cameras) + 2:
                # encode the images in tmp_list to be sent via ZMQ
                meta_data = {
                    "timestamp": time.time(),
                    "cameras": list(tmp_dict.keys())
                }
                image_pub.send_json(meta_data, flags=zmq.SNDMORE)
                for i, img in enumerate(tmp_dict.values()):
                    if i < len(tmp_dict) - 1:
                        image_pub.send(img, zmq.SNDMORE)  # Use SNDMORE for all but the last part
                    else:
                        image_pub.send(img)  # No SNDMORE for the last part

            # bar.update(1)
            rate.sleep()


    except Exception as e:
        print(e)
        # Close all cameras
        for camera in cameras.values():
            camera.close()
        zed.close()
        image_pub.close()
