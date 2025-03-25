import cv2
import pyrealsense2 as rs
import numpy as np
import json

jsonObj = json.load(open("/home/huzheyuan/dual_xarms/dual_xarms_sim/test800.json"))
json_string= str(jsonObj).replace("'", '\"').strip()

def get_device_serial_numbers():
    devices = rs.context().devices
    return {d.get_info(rs.camera_info.serial_number): d for d in devices}


devices = get_device_serial_numbers()
serial_number = ["130322273478", "128422272097"]
for serial_number in serial_number:
    assert serial_number in devices
    ser_dev = rs.serializable_device(devices[serial_number])
    ser_dev.load_json(json_string)
    print("loaded json")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert RealSense frame to OpenCV image
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense D405 and OpenCV', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense D405 and OpenCV', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # Stop streaming
    pipeline.stop()