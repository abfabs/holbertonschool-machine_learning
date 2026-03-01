#!/usr/bin/env python3
"""0-conv_forward.py
Convolutional forward propagation.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer.

    Args:
        A_prev (np.ndarray): shape (m, h_prev, w_prev, c_prev)
        W (np.ndarray): shape (kh, kw, c_prev, c_new)
        b (np.ndarray): shape (1, 1, 1, c_new)
        activation (callable): activation function
        padding (str): 'same' or 'valid'
        stride (tuple): (sh, sw)

    Returns:
        np.ndarray: activated output
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride

    if c_prev_w != c_prev:
        raise ValueError("W channels do not match A_prev channels")
    if padding not in ("same", "valid"):
        raise ValueError("padding must be 'same' or 'valid'")

    if padding == "valid":
        ph = 0
        pw = 0
    else:
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)

    A_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    h_pad = h_prev + 2 * ph
    w_pad = w_prev + 2 * pw

    out_h = ((h_pad - kh) // sh) + 1
    out_w = ((w_pad - kw) // sw) + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        i0 = i * sh
        i1 = i0 + kh
        for j in range(out_w):
            j0 = j * sw
            j1 = j0 + kw

            patch = A_pad[:, i0:i1, j0:j1, :]

            for k in range(c_new):
                w_k = W[:, :, :, k]
                conv = np.sum(patch * w_k, axis=(1, 2, 3))
                Z[:, i, j, k] = conv + b[0, 0, 0, k]

    return activation(Z)
