#!/usr/bin/env python3
"""0-conv_forward.py
Convolutional forward propagation for a CNN layer.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Parameters
    ----------
    A_prev : np.ndarray (m, h_prev, w_prev, c_prev)
        Output of the previous layer.
    W : np.ndarray (kh, kw, c_prev, c_new)
        Convolution kernels.
    b : np.ndarray (1, 1, 1, c_new)
        Biases.
    activation : callable
        Activation function applied to the convolution output.
    padding : str
        "same" or "valid" indicating padding type.
    stride : tuple
        (sh, sw) stride for height and width.

    Returns
    -------
    np.ndarray
        The output of the convolutional layer after applying activation.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride

    if c_prev_w != c_prev:
        raise ValueError("W has inconsistent number of input channels vs A_prev")

    if padding not in ("same", "valid"):
        raise ValueError("padding must be 'same' or 'valid'")

    # Compute padding
    if padding == "valid":
        ph = 0
        pw = 0
    else:
        # Output dimensions for "same" (ceil division)
        out_h = int(np.ceil(h_prev / sh))
        out_w = int(np.ceil(w_prev / sw))

        # Total padding needed along each dimension
        pad_h = max((out_h - 1) * sh + kh - h_prev, 0)
        pad_w = max((out_w - 1) * sw + kw - w_prev, 0)

        # Split padding top/bottom and left/right
        ph = pad_h // 2
        pw = pad_w // 2

    # Pad the input on height and width (no padding on batch/channel)
    A_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Compute output spatial dimensions
    h_pad = h_prev + 2 * ph
    w_pad = w_prev + 2 * pw
    out_h = ((h_pad - kh) // sh) + 1
    out_w = ((w_pad - kw) // sw) + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    # Convolution
    for i in range(out_h):
        i0 = i * sh
        i1 = i0 + kh
        for j in range(out_w):
            j0 = j * sw
            j1 = j0 + kw

            patch = A_pad[:, i0:i1, j0:j1, :]  # (m, kh, kw, c_prev)

            # For each output channel, convolve patch with its filter
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(patch * W[:, :, :, k], axis=(1, 2, 3)) + b[0, 0, 0, k]

    return activation(Z)
