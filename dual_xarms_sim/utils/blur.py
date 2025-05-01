# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn.functional as F

def gaussian_2d_kernel(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for convolution.
    
    Args:
        kernel_size: integer, the height/width of the kernel (assumed square).
        sigma: standard deviation for the Gaussian.
        device, dtype: optional, to place the kernel on a specific device / dtype.
    Returns:
        kernel: Tensor of shape (kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords -= (kernel_size - 1) / 2.0  # shift to center
    x, y = torch.meshgrid(coords, coords, indexing='xy')
    kernel_2d = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

def gaussian_2d_smoothing(
    img: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Apply 2D Gaussian smoothing (blur) to a batch of images.
    
    Args:
        img: Tensor of shape (..., C, H, W).
        scale: Controls the standard deviation (sigma) of the Gaussian kernel. Larger scale corresponds to more smoothing.

    Returns:
        blurred: Tensor of the same shape as img.
    """
    if scale <= 0:
        return img

    sigma = scale
    kernel_size = max(3, 2 * math.ceil(3 * sigma) + 1)

    kernel_2d = gaussian_2d_kernel(kernel_size, sigma, device=img.device, dtype=img.dtype)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    C = img.shape[-3]
    kernel_2d = kernel_2d.repeat(C, 1, 1, 1)  # shape: (C, 1, kH, kW)

    padding = kernel_size // 2

    original_shape = img.shape
    batch_shape = original_shape[:-3]
    spatial_shape = original_shape[-2:]
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    img_reshaped = img.view(batch_size, C, *spatial_shape)

    blurred = F.conv2d(img_reshaped, kernel_2d, groups=C, padding=padding)
    return blurred.view(*original_shape)