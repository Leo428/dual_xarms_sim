import pyzed.sl as sl

try:
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

    left_top_rgb = sl.Mat()
    right_top_rgb = sl.Mat()
    import ipdb; ipdb.set_trace()
    runtime_parameters = sl.RuntimeParameters()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS.
        # original numpy array, not a copy (H, W, 4 (rgba))
        zed.retrieve_image(left_top_rgb, sl.VIEW.LEFT).get_data()
        zed.retrieve_image(right_top_rgb, sl.VIEW.RIGHT).get_data()
    
except Exception as e:
    print(e)
    zed.close()
    exit(1)