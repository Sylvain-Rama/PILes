# -*- coding: utf-8 -*-
"""
Created on Fri May 13 16:19:24 2022

@author: Sylvain

Functions are used to weight the sizes, alphas, etc... of the plots.
They have to return a single array normalized betwen 0 and 1.
"""


import numpy as np

#%% new cell
class Distance:
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
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def normal(self, sd=0.4):

        _d = self.inc_uniform()

        gaussian_weight = (
            1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-(_d ** 2) / (2 * sd ** 2))
        )

        return gaussian_weight

    def wave(self, waves=3):

        _d = self.inc_uniform() * waves * 2 * np.pi

        w = np.cos(_d)

        w = self._normalize(w)

        return w

    def laplace(self):

        _d = self.inc_uniform()
        w = 0.5 * np.exp(-_d)
        w = self._normalize(w)

        return w


class Angle:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        if len(x) != len(y):
            raise ValueError(
                f"x and y coordinates must have the same length but got {len(x)} and {len(y)}."
            )
            return

    def _cart_to_polar(self, x, y):

        # r = np.sqrt(x**2 + y**2)
        # theta = np.arctan(y / x)
        
        r = np.sqrt(x**2 + y**2)

        theta = np.where(y >= 0, np.arccos(x/r), -np.arccos(x/r))
        # theta = np.arctan(y / x)
        

        return r, theta

    def wave(self, waves=2):
        r, theta = self._cart_to_polar(self.x, self.y)
        theta = theta * waves

        value = (np.sin(theta) + 1) / 2
        value = np.nan_to_num(value, nan=0)

        return value
