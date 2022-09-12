# -*- coding: utf-8 -*-
"""
@author: Sylvain
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

from helpers import color_dict


class ColorPile:
    def __init__(self, color_map=None, name=None):

        self.color_map = color_map
        self.name = name

    def __repr__(self):
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        float_cmap = self._RGB_to_float()
        new_cmap = ListedColormap(float_cmap)

        fig, ax = plt.subplots(nrows=1, figsize=(12, 1))

        ax.imshow(gradient, aspect="auto", cmap=new_cmap)
        ax.set_axis_off()
        fig.suptitle(self.name)
        fig.tight_layout(pad=1.2)

        return f"ColorPile {self.name}, length {len(self.color_map)}."

    def __str__(self):
        return self.__repr__()

    def __add__(self, cmap2):
        cmap3 = ColorPile(
            color_map=self.color_map + cmap2.color_map,
            name=self.name + "+" + cmap2.name,
        )
        return cmap3

    def __getitem__(self, i):
        if isinstance(i, int):
            return ColorPile(color_map=self.color_map[i : i + 1], name=self.name)
        elif isinstance(i, slice):
            return ColorPile(color_map=self.color_map[i], name=self.name)

    def __len__(self):
        return len(self.color_map)

    def _RGB_to_float(self):
        return [[x[0] / 255, x[1] / 255, x[2] / 255] for x in self.color_map]

    def _normalize(self, r):
        return (r - r.min()) / (r.max() - r.min())

    def from_matplotlib(self, name="inferno", n=10):

        try:
            plt_map = plt.cm.get_cmap(name, n).colors * 255
        # Many plt Colormaps do not have the .colors attribute. Don't ask why.
        except (ValueError, AttributeError):
            plt_map = plt.cm.get_cmap(name, n)
            plt_map = plt_map(range(n)) * 255
        # Removing alpha channel, converting to int
        plt_map = np.round(plt_map[:, 0:3]).astype(int)

        return ColorPile(color_map=list(map(list, plt_map)), name=name)

    def from_list(self, color_list=["red", "black"]):
        colors = []
        for color in color_list:

            if isinstance(color, str):
                if color.upper() in color_dict:
                    colors.append(color_dict[color.upper()])
            elif isinstance(color, tuple):
                if len(color) == 3:

                    if all(isinstance(v, int) for v in color):
                        colors.append(color)
            else:
                raise ValueError(
                    f"{color} neither in {color_dict.keys()} nor a valid RGB tuple"
                )
        return ColorPile(color_map=colors, name="from_list")

    def to_matplotlib(self):
        return ListedColormap(self._RGB_to_float())

    def to_tuple_list(self):
        return [tuple(x) for x in self.color_map]

    def invert(self):
        return ColorPile(color_map=self.color_map[::-1], name=self.name + "_inverted")

    def loop(self, n=1):
        color_map = self.color_map
        if not isinstance(color_map, list):
            color_map = list(color_map)
        for i in range(n):
            color_map = color_map + self.color_map
        name = self.name + f"-looped_{n}"

        return ColorPile(color_map=color_map, name=name)

    def extend(self, n=10):
        # Detour by Numpy to use linspace
        colors = np.asarray(self.color_map)
        cmap = LinearSegmentedColormap.from_list("", colors / 255, 256)
        cmap = cmap(np.linspace(0, 1, n)) * 255
        cmap = cmap[:, 0:3].astype(int)
        color_map = [tuple(x) for x in cmap]  # back to list of tuples for ease of use.

        name = self.name + f"-extended_{n}"

        return ColorPile(color_map=color_map, name=name)

    def map_to_distance(self, x, y):

        cmap = self.color_map

        d = np.sqrt(x ** 2 + y ** 2)

        bins = np.linspace(0, d.max(), num=len(cmap) + 1)
        bins = bins[:-1]

        indexes = np.digitize(d, bins) - 1

        colors = [cmap[i] for i in indexes]

        return colors

    def map_to_index(self, i):

        i = self._normalize(i)

        bins = np.linspace(0, i.max(), num=len(self.color_map) + 1)
        bins = bins[:-1]

        indexes = np.digitize(i, bins) - 1

        colors = [self.color_map[i] for i in indexes]

        return colors

    def map_to_angle(self, x, y, waves=3):

        cmap = self.color_map

        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.where(y >= 0, np.arccos(x / r), -np.arccos(x / r))
        theta = theta * waves

        bins = np.linspace(0, theta.max(), num=len(cmap) + 1)
        bins = bins[:-1]

        indexes = np.round(((np.sin(theta) + 1) / 2) * (len(cmap) - 1)).astype(int)
        colors = [cmap[i] for i in indexes]

        return colors

    def set_name(self, name=None):
        if isinstance(name, str):
            return ColorPile(color_map=self.color_map, name=name)
        else:
            return TypeError("Name must be a valid string.")


if __name__ == "__main__":
    # Examples for the use of ColorPile class
    cmap1 = ColorPile().from_matplotlib("inferno", n=20)
    cmap2 = ColorPile().from_matplotlib("viridis", n=50).invert()

    cmap3 = (cmap1 + cmap2).loop(n=3).set_name("Special one")
    print(cmap3)

    cmap4 = (
        ColorPile().from_list(color_list=[(255, 0, 0), "yellow", "blue"]).extend(100)
    )
    cmap4 = cmap4 + cmap4.invert()
    print(cmap4)
