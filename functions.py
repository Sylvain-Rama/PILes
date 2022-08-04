# -*- coding: utf-8 -*-
"""
Functions are used to weight the sizes, alphas, etc... of the plots.
They have to return a single array normalized betwen 0 and 1.
"""

import numpy as np


class Distance:
    """This class will output values between 0 and 1 depending on the distance 
    of the point from the centre of the distribution.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

        if len(x) != len(y):
            raise ValueError(
                f"x and y coordinates must have the same length but got {len(x)} and {len(y)}."
            )
            return

    def _normalize(self, r):
        return (r - r.min()) / (r.max() - r.min())

    def inc_uniform(self):
        # Simple increasing ramp.
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def normal(self, sd=0.4):
        # Centered gaudssian distribution.
        _d = self.inc_uniform()
        gaussian_weight = (
            1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-(_d ** 2) / (2 * sd ** 2))
        )

        return self._normalize(gaussian_weight)

    def wave(self, waves=3):
        # Centered wave pattern
        _d = self.inc_uniform() * waves * 2 * np.pi

        w = np.cos(_d)

        w = self._normalize(w)

        return w

    def laplace(self):
        # Centered Laplace distribution
        _d = self.inc_uniform()
        w = 0.5 * np.exp(-_d)
        w = self._normalize(w)

        return w


class Angle:
    """This class will output weights between 0 and 1, depending on the theta 
    angle of the point.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

        if len(x) != len(y):
            raise ValueError(
                f"x and y coordinates must have the same length but got {len(x)} and {len(y)}."
            )
            return
        
    def _normalize(self, r):
        return (r - r.min()) / (r.max() - r.min())


    def _cart_to_polar(self, x, y):
        # Convert cartesian coordinates to polar.
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.where(y >= 0, np.arccos(x / r), -np.arccos(x / r))
        

        return r, theta

    def wave(self, waves=2):
        # waves depending on the theta angle.
        r, theta = self._cart_to_polar(self.x, self.y)
        theta = theta * waves

        value = (np.sin(theta) + 1) / 2
        value = np.nan_to_num(value, nan=0)

        return value

    def inc_uniform(self):
        _, theta = self._cart_to_polar(self.x, self.y)
        return self._normalize(theta)
        


class Value:
    """This class outputs a normalized version of the input array. Used to index
    an attribute to any array.
    """

    def __init__(self, x):
        self.x = x

    def _normalize(self, r):
        return (r - r.min()) / (r.max() - r.min())

    def as_index(self):
        return self._normalize(self.x)
