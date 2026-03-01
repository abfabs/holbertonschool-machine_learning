#!/usr/bin/env python3
"""2-conv_backward.py
Performs back propagation over a convolutional layer.
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer.

    Args:
        dZ (np.ndarray): shape (m, h_new, w_new, c_new)
        A_prev (np.ndarray): shape (m, h_prev, w_prev, c_prev)
        W (np.ndarray): shape (kh, kw, c_prev, c_new)
        b (np.ndarray): shape (1, 1, 1, c_new)
        padding (str): 'same' or 'valid'
        stride (tuple): (sh, sw)

    Returns:
        tuple: (dA_prev, dW, db)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    m_dz, h_new, w_new, c_new_dz = dZ.shape
    sh, sw = stride

    if m_dz != m:
        raise ValueError("dZ and A_prev must have the same batch size")
    if c_prev_w != c_prev:
        raise ValueError("W channels do not match A_prev channels")
    if c_new_dz != c_new:
        raise ValueError("dZ channels do not match W output channels")
    if padding not in ("same", "valid"):
        raise ValueError("padding must be 'same' or 'valid'")

    if padding == "valid":
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
    else:
        pad_h = max((h_new - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_new - 1) * sw + kw - w_prev, 0)

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

    dA_pad = np.zeros_like(A_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        i0 = i * sh
        i1 = i0 + kh
        for j in range(w_new):
            j0 = j * sw
            j1 = j0 + kw

            a_slice = A_pad[:, i0:i1, j0:j1, :]

            for k in range(c_new):
                dz = dZ[:, i, j, k][:, None, None, None]
                dW[:, :, :, k] += np.sum(a_slice * dz, axis=0)
                dA_pad[:, i0:i1, j0:j1, :] += W[:, :, :, k] * dz

    if padding == "valid":
        dA_prev = dA_pad
    else:
        dA_prev = dA_pad[
            :,
            pad_top:pad_top + h_prev,
            pad_left:pad_left + w_prev,
            :,
        ]

    return dA_prev, dW, db
