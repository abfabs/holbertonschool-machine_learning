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

    Returns: np.ndarray of convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        # Output should be ceil(h/sh) x ceil(w/sw)
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))

        # Solve for padding needed to hit those output sizes:
        # out_h = floor((h + 2ph - kh)/sh) + 1
        # => 2ph = (out_h - 1)*sh + kh - h
        ph = int(np.ceil(((out_h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((out_w - 1) * sw + kw - w) / 2))
    else:
        ph, pw = padding

    # Pad images with zeros
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Compute output dims
    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    # Only 2 loops: loop over i and j (output spatial positions)
    for i in range(out_h):
        for j in range(out_w):
            i0 = i * sh
            j0 = j * sw
            patch = padded[:, i0:i0 + kh, j0:j0 + kw]  # (m, kh, kw)
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
