#!/usr/bin/env python3
"""6-pool.py"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    images: np.ndarray (m, h, w, c)
    kernel_shape: (kh, kw)
    stride: (sh, sw)
    mode: 'max' or 'avg'

    Returns: np.ndarray (m, out_h, out_w, c)
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w, c))

    # Only 2 loops: over output height and width
    for i in range(out_h):
        for j in range(out_w):
            i0 = i * sh
            j0 = j * sw
            patch = images[:, i0:i0 + kh, j0:j0 + kw, :]  # (m, kh, kw, c)

            if mode == 'max':
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
            else:  # 'avg'
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))

    return output
