import numpy as np
import h5py
from tqdm import tqdm
import os
import glob
import cv2
import torch
from torchvision.transforms.v2 import CenterCrop, Resize
from einops import rearrange

def process_camera_images(image_dataset, episode_length, resize):
    """
    Reads, decompresses, processes, and recompresses the images.
    Returns an array of padded, recompressed images.
    """
    cropped_resized_frames = []
    compressed_frames = []
    compressed_lengths = []
    compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    # Decompress each frame and convert to tensor
    for t in tqdm(range(episode_length), desc="Decompress OG Frames", leave=False):
        compressed_image = image_dataset[t]
        # Decode the image (returns a numpy array)
        decompressed_image = np.array(cv2.imdecode(compressed_image, 1))
        decompressed_image = torch.from_numpy(decompressed_image)
        cropped_resized_frames.append(decompressed_image)

    # Stack all frames and perform cropping and resizing
    cropped_resized_frames = torch.stack(cropped_resized_frames)                     # Shape: (t, h, w, c)
    cropped_resized_frames = rearrange(cropped_resized_frames, "t h w c -> t c h w")
    cropped_resized_frames = resize(cropped_resized_frames)
    cropped_resized_frames = rearrange(cropped_resized_frames, "t c h w -> t h w c")
    cropped_resized_frames = cropped_resized_frames.numpy()

    # Compress each processed frame and record lengths
    for frame in tqdm(cropped_resized_frames, desc="Compress Cropped Frames", leave=False):
        result, encoded_image = cv2.imencode('.jpg', frame, compression_params)
        compressed_frames.append(encoded_image)
        compressed_lengths.append(len(encoded_image))
    compressed_lengths = np.array(compressed_lengths)
    padded_size = compressed_lengths.max()

    # Pad each compressed image with zeros so that they all have the same length
    padded_compressed_frames = []
    for encoded in compressed_frames:
        padded_frame = np.zeros((padded_size,), dtype=np.uint8)
        padded_frame[:len(encoded)] = encoded
        padded_compressed_frames.append(padded_frame)
    padded_compressed_frames = np.stack(padded_compressed_frames)
    return padded_compressed_frames

if __name__ == "__main__":
    # Directory and file list
    # hdf5_dir = '/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_0226_hdf5'
    # hdf5_dir = '/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_robyn_0228_hdf5'
    # hdf5_dir = '/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_riya_0312_hdf5'
    hdf5_dir = '/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_jasmine_0226_hdf5'
    hdf5_files = glob.glob(os.path.join(hdf5_dir, "*.hdf5"))

    # Define transforms
    resize = Resize((140, 140))

    # List of camera datasets that will be processed
    camera_names = [
        "obses/images/left/top",
        "obses/images/right/top",
        "obses/images/left/wrist",
        "obses/images/right/wrist"
    ]

    for filename in tqdm(hdf5_files, desc="Processing Files"):
        try:
            print(f"Processing file: {filename}")
            # Define new filename, e.g., cropped_originalfilename.hdf5
            new_filename = os.path.join(os.path.dirname(filename), f"cropped_{os.path.basename(filename)}")

            # Open the original file for reading
            with h5py.File(filename, "r") as src:
                # Open a new file for writing (will create a new file)
                with h5py.File(new_filename, "w") as dst:
                    # Copy all top-level groups/datasets from src to dst
                    file_keys = list(src.keys())
                    file_keys.remove("obses")
                    file_keys.append("obses/state")
                    for key in file_keys:
                        src.copy(src[key], dst, key)

                    # Get the episode length from an auxiliary dataset (assumes existence)
                    episode_length = len(src["actions/relative_action"][()])

                    # Process each camera dataset: update the new file with processed images
                    for cam_name in tqdm(camera_names, desc="Processing Cameras", leave=False):
                        print(f"Processing camera: {cam_name}")
                        image_dataset = src[cam_name]
                        # Process the images using the helper function
                        processed_images = process_camera_images(image_dataset, episode_length, resize)
                        # Create a new dataset with the processed image data; here, we use gzip compression as an example
                        dst.create_dataset(cam_name, data=processed_images)
            print(f"Processed and saved new file: {new_filename}")

        except Exception as e:
            print(f"Failed to process file: {filename}")
            raise e
            # Optionally, you can continue to the next file
