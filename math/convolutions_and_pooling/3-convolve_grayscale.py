#!/usr/bin/env python3
"""3-convolve_grayscale.py"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with padding + stride.

    images: np.ndarray (m, h, w)
    kernel: np.ndarray (kh, kw)
    padding: 'same', 'valid', or (ph, pw)
    stride: (sh, sw)

    Returns: np.ndarray (m, out_h, out_w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding (top/bottom, left/right)
    if padding == 'valid':
        pt = pb = pl = pr = 0

    elif padding == 'same':
        # Target output size for "same" with stride
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))

        # Total padding needed to achieve that output
        pad_h = max((out_h - 1) * sh + kh - h, 0)
        pad_w = max((out_w - 1) * sw + kw - w, 0)

        # Split padding (can be asymmetric)
        pt = pad_h // 2
        pb = pad_h - pt
        pl = pad_w // 2
        pr = pad_w - pl

    else:
        ph, pw = padding
        pt = pb = ph
        pl = pr = pw

    # Pad images with zeros
    padded = np.pad(
        images,
        ((0, 0), (pt, pb), (pl, pr)),
        mode='constant'
    )

    # Output dimensions after padding + stride
    out_h = ((padded.shape[1] - kh) // sh) + 1
    out_w = ((padded.shape[2] - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    # Only 2 loops: over output spatial positions
    for i in range(out_h):
        for j in range(out_w):
            i0 = i * sh
            j0 = j * sw
            patch = padded[:, i0:i0 + kh, j0:j0 + kw]  # (m, kh, kw)
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
