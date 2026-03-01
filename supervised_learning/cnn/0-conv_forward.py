#!/usr/bin/env python3
"""0-conv_forward.py
Performs forward propagation over a convolutional layer.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer.

    Args:
        A_prev (np.ndarray): Shape (m, h_prev, w_prev, c_prev)
        W (np.ndarray): Shape (kh, kw, c_prev, c_new)
        b (np.ndarray): Shape (1, 1, 1, c_new)
        activation (callable): Activation function
        padding (str): "same" or "valid"
        stride (tuple): (sh, sw)

    Returns:
        np.ndarray: Activated output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride

    if c_prev_w != c_prev:
        raise ValueError("W channels do not match A_prev channels")
    if padding not in ("same", "valid"):
        raise ValueError("padding must be 'same' or 'valid'")

    if padding == "valid":
        pad_top = pad_bottom = 0
        pad_left = pad_right = 0
    else:
        out_h = int(np.ceil(h_prev / sh))
        out_w = int(np.ceil(w_prev / sw))

        pad_h = max((out_h - 1) * sh + kh - h_prev, 0)
        pad_w = max((out_w - 1) * sw + kw - w_prev, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

    A_pad = np.pad(
        A_prev,
        pad_width=(
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0),
        ),
        mode="constant",
        constant_values=0,
    )

    h_pad = h_prev + pad_top + pad_bottom
    w_pad = w_prev + pad_left + pad_right

    out_h = ((h_pad - kh) // sh) + 1
    out_w = ((w_pad - kw) // sw) + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        i0 = i * sh
        i1 = i0 + kh
        for j in range(out_w):
            j0 = j * sw
            j1 = j0 + kw

            patch = A_pad[:, i0:i1, j0:j1, :]  # (m, kh, kw, c_prev)

            for k in range(c_new):
                conv = np.sum(patch * W[:, :, :, k], axis=(1, 2, 3))
                Z[:, i, j, k] = conv + b[0, 0, 0, k]

    return activation(Z)
