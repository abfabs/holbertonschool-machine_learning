#!/usr/bin/env python3
"""1-convolve_grayscale_same.py"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    images: np.ndarray of shape (m, h, w)
    kernel: np.ndarray of shape (kh, kw)

    Returns: np.ndarray of shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute padding for "same" convolution
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad images with zeros
    padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    # Output has same height and width as input
    output = np.zeros((m, h, w))

    # Only 2 loops: over output height and width
    for i in range(h):
        for j in range(w):
            patch = padded[:, i:i + kh, j:j + kw]   # (m, kh, kw)
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
