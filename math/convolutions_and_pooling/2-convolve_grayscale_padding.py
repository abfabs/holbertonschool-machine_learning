#!/usr/bin/env python3
"""2-convolve_grayscale_padding.py"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    images: np.ndarray of shape (m, h, w)
    kernel: np.ndarray of shape (kh, kw)
    padding: tuple (ph, pw)

    Returns: np.ndarray of shape (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad with zeros
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode="constant"
    )

    out_h = (h + 2 * ph) - kh + 1
    out_w = (w + 2 * pw) - kw + 1
    output = np.zeros((m, out_h, out_w))

    # Only 2 loops: over output height and width
    for i in range(out_h):
        for j in range(out_w):
            patch = padded[:, i:i + kh, j:j + kw]  # (m, kh, kw)
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
