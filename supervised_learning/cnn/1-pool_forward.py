#!/usr/bin/env python3
"""1-pool_forward.py
Performs forward propagation over a pooling layer.
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer.

    Args:
        A_prev (np.ndarray): shape (m, h_prev, w_prev, c_prev)
        kernel_shape (tuple): (kh, kw)
        stride (tuple): (sh, sw)
        mode (str): 'max' or 'avg'

    Returns:
        np.ndarray: output of pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = ((h_prev - kh) // sh) + 1
    out_w = ((w_prev - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w, c_prev))

    for i in range(out_h):
        i0 = i * sh
        i1 = i0 + kh
        for j in range(out_w):
            j0 = j * sw
            j1 = j0 + kw

            patch = A_prev[:, i0:i1, j0:j1, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return output
