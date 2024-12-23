# Required Libraries
import zmq
import time
import numpy as np
import json
from multiprocessing import Manager, Process
from loop_rate_limiters import RateLimiter
from tqdm import tqdm

# ZMQ Context
context = zmq.Context()

# Configuration Constants
ROBOT_CONTROL_FPS = 60
BUFFER_SIZE = 30  # Reduced buffer size to store recent data for synchronization

# Shared Buffer for Time Synchronization
with Manager() as manager:
    robot_state_buffer = manager.list()  # Using list to mimic deque
    images_buffer = manager.list()  # Using list to mimic deque

    # Timestamp Function
    def get_timestamp():
        return time.time()

    # ZMQ Server for Robot State Subscriber
    def robot_state_subscriber(robot_state_buffer):
        robot_state_sub = context.socket(zmq.SUB)
        robot_state_sub.setsockopt(zmq.CONFLATE, 1)  # Ensure only the latest message is kept
        robot_state_sub.connect("tcp://127.0.0.1:5003")
        robot_state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        rate = RateLimiter(frequency=1000, warn=False)

        while True:
            try:
                message = robot_state_sub.recv_json(flags=zmq.NOBLOCK)
                if len(robot_state_buffer) >= BUFFER_SIZE:
                    robot_state_buffer.pop()  # Remove the oldest element if the buffer is full
                robot_state_buffer.insert(0, message)  # Insert new data at the front
            except zmq.Again:
                pass
            rate.sleep()

    # ZMQ Server for Images Subscriber
    def images_subscriber(images_buffer):
        images_sub = context.socket(zmq.SUB)
        images_sub.setsockopt(zmq.RCVHWM, 1)
        images_sub.connect("tcp://127.0.0.1:5004")
        images_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        rate = RateLimiter(frequency=1000, warn=False)

        while True:
            try:
                metadata = images_sub.recv_json(flags=zmq.NOBLOCK)
                images_data = {}
                for cam in metadata["cameras"]:
                    binary_data = images_sub.recv()  # Part 2: Image data
                    images_data[cam] = binary_data

                if len(images_buffer) >= BUFFER_SIZE:
                    images_buffer.pop()  # Remove the oldest element if the buffer is full

                images_buffer.insert(0, (metadata, images_data))  # Insert new data at the front
            except zmq.Again:
                pass
            rate.sleep()

    # ZMQ Server for Central Broker (Publishing Messages at 60Hz)
    def zmq_broker(robot_state_buffer):
        rate = RateLimiter(frequency=ROBOT_CONTROL_FPS)
        pub_socket = context.socket(zmq.PUB)
        pub_socket.setsockopt(zmq.SNDHWM, 1)
        pub_socket.bind("tcp://127.0.0.1:5005")

        min_diff_avg = 0.0

        bar = tqdm(total=1e8, desc="freq:")
        while True:
            if len(robot_state_buffer) > 0 and len(images_buffer) > 0:
                # Extract the latest image metadata and robot state
                latest_image_metadata, latest_image_data = images_buffer[0]
                latest_image_timestamp = latest_image_metadata["timestamp"]
                # Find the robot state with the closest timestamp to the image data
                closest_robot_state = None
                min_time_diff = float('inf')

                for robot_state in robot_state_buffer:
                    time_diff = abs(robot_state["timestamp"] - latest_image_timestamp)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_robot_state = robot_state

                if closest_robot_state:
                    min_diff_avg = 0.9 * min_diff_avg + 0.1 * min_time_diff
                    bar.desc = f"avg state image diff: {min_diff_avg * 1000:.2f} ms"
                    msg = {
                        "timestamp": latest_image_timestamp,
                        "robot_state": closest_robot_state,
                        "metadata": latest_image_metadata,
                    }
                    # pub_socket.send_json(msg)
                    pub_socket.send_json(msg, flags=zmq.SNDMORE)
                    for i, name in enumerate(latest_image_metadata["cameras"]):
                        if i < len(latest_image_metadata["cameras"]) - 1:
                            pub_socket.send(latest_image_data[name], zmq.SNDMORE)
                        else:
                            pub_socket.send(latest_image_data[name])
                else:
                    print("No robot state found")
            # elif len(robot_state_buffer) > 0:
            #     pub_socket.send_json(robot_state_buffer[0])  # Publish the latest robot state

            bar.update(1)
            rate.sleep()

    # Run each function in a separate process
    robot_state_process = Process(target=robot_state_subscriber, args=(robot_state_buffer,))
    images_process = Process(target=images_subscriber, args=(images_buffer,))
    broker_process = Process(target=zmq_broker, args=(robot_state_buffer,))

    robot_state_process.start()
    images_process.start()
    broker_process.start()

    robot_state_process.join()
    images_process.join()
    broker_process.join()
