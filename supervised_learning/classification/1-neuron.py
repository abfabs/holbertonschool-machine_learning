#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""

import numpy as np


class Neuron:
    """Single neuron with private attributes"""

    def __init__(self, nx):
        """
        nx: number of input features
        """

        # Validation (order matters)
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        # Private instance attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    # Getter to W
    @property
    def W(self):
        return self.__W

    # Getter to b
    @property
    def b(self):
        return self.__b

    # Getter to A
    @property
    def A(self):
        return self.__A
