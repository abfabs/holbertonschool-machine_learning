#!/usr/bin/env python3
"""0-convolve_grayscale_valid.py"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    images: np.ndarray of shape (m, h, w)
    kernel: np.ndarray of shape (kh, kw)

    Returns: np.ndarray of shape (m, h - kh + 1, w - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    # Output container
    output = np.zeros((m, out_h, out_w))

    # Only 2 loops: over output height and width
    for i in range(out_h):
        for j in range(out_w):
            patch = images[:, i:i + kh, j:j + kw]          # (m, kh, kw)
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
