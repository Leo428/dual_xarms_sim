import numpy as np
import h5py
import cv2
import imageio
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# 1) Adjust these to your setup
MAX_EPISODES = 50
CAM_KEY = "obses/images/right/top"
DATA_DIR = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_0226_hdf5"
THRESHOLD = 10       # pixel‐difference threshold for “foreground”
SMOOTH_SIGMA = 5        # how much to blur the raw count map
ALPHA_SCALE = 0.5      # max alpha = heat*ALPHA_SCALE
OUT_PATH = "heatmap_overlay.png"

# 2) First, compute a static “background” (median of all frames)
all_medians = []
for ep in tqdm(range(MAX_EPISODES)):
    with h5py.File(f"{DATA_DIR}/episode_{ep}.hdf5", "r") as f:
        # decompress to gray
        buf = f[CAM_KEY][()]
        frames = [cv2.imdecode(b, cv2.IMREAD_GRAYSCALE) for b in buf]
        stack = np.stack(frames, axis=0)            # shape [T, H, W]
        med = np.median(stack, axis=0).astype(np.uint8)
        all_medians.append(med)

# Stack episode-medians and take the median again
background = np.median(np.stack(all_medians, axis=0), axis=0).astype(np.uint8)
H, W = background.shape

# 3) Accumulate “foreground” counts where frame differs from background
heat = np.zeros((H, W), dtype=np.float32)
for ep in tqdm(range(MAX_EPISODES)):
    with h5py.File(f"{DATA_DIR}/episode_{ep}.hdf5", "r") as f:
        buf = f[CAM_KEY][()]
        for b in buf:
            frame = cv2.imdecode(b, cv2.IMREAD_GRAYSCALE)
            mask = np.abs(frame.astype(int) - background.astype(int)) > THRESHOLD
            heat += mask.astype(np.float32)

# 4) Smooth + normalize to [0,1]
heat = gaussian_filter(heat, sigma=SMOOTH_SIGMA)
heat = heat / (heat.max() + 1e-8)

# 5) Colorize + add alpha
#    use OpenCV’s jet colormap for visibility
heat_u8 = (heat * 255).astype(np.uint8)
color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR  uint8
alpha = (heat * ALPHA_SCALE * 255).astype(np.uint8)

# 6) Merge into RGBA and save
rgba = cv2.merge([
    color[:,:,2],  # R
    color[:,:,1],  # G
    color[:,:,0],  # B
    alpha          # A
])
cv2.imwrite(OUT_PATH, rgba)

print(f"Written heatmap overlay → {OUT_PATH}")