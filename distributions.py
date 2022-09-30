# -*- coding: utf-8 -*-
"""
Distributions functions must return  x, y coordinates in cartesian form, 
normalized between -1 and 1.
"""


import numpy as np
import matplotlib.pyplot as plt


class _polar_coords:
    """Base class for polar distributions.
    """

    def __init__(self, r, theta):
        self.r = r
        self.theta = theta

    def _pol_to_cart(self, r, theta):
        # Convert polar coordinates to cartesian.
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return x, y

    def _normalize(self, r):
        # normalize arrays between 0 and 1
        return (r - r.min()) / (r.max() - r.min())


class Spiral(_polar_coords):
    """Output x, y coordinates of multiple types of spirals.
    """

    def __init__(self, n=100, turns=3):
        self.n = n
        self.turns = turns
        self._thetas = np.linspace(0, self.turns * 2 * np.pi, self.n)
        self._step = 1 / (turns * 2 * np.pi)

    # To use np.pi yields strange results...
    def golden(self, a=1, b=3.1415 / 2.0, **kwargs):
        # Golden spiral
        r = a * np.exp(self._thetas * np.cos(b))
        r = self._normalize(r)

        return self._pol_to_cart(r, self._thetas)

    def archimedean(self, arc=1, sep=1, **kwargs):
        # Regular archimedean spiral.
        # actually a slight cheat: drawing circles arcs incrementally.
        _radii = [0]
        _thetas = [0]
        r = arc
        b = sep / (2 * np.pi)
        t = r / b

        for i in range(self.n - 1):
            _radii.append(r)
            _thetas.append(t)
            t += arc / r
            r = b * t
        r = np.array(_radii)
        thetas = np.array(_thetas)
        r = self._normalize(r)

        return self._pol_to_cart(r, thetas)

    def patterned(self, segments=7, inner=0.1, outer=1):
        # Patterned spiral, we can use segments to define the number of sides.
        inner = inner / outer
        outer = 1

        segments = self.turns * segments + 1
        theta = [
            arc / (segments - 1) * np.pi * 2.0 * self.turns
            for arc in range(int(segments))
        ]

        r = np.linspace(inner, outer, segments)

        return self._pol_to_cart(r, theta)

    def quadratic(self, k=13):
        # Quadratic spiral

        n = np.linspace(0, self.n, num=self.n)
        pn = ((k - 2) * n * (n + 1)) / 2 - (k - 3) * n
        theta = 2 * np.pi * np.sqrt(pn)
        r = self._normalize(np.sqrt(pn))

        return self._pol_to_cart(r, theta)


class Circular(_polar_coords):
    def __init__(self, n=8):
        self.n = n

    def uniform(self, turns=1):
        # Uniformly distributed on a circle

        r = np.ones((self.n))
        theta = np.linspace(0, (2 * np.pi * turns) - (2 * np.pi / self.n), self.n)

        return self._pol_to_cart(r, theta)

    def polygon(self, angle=0, **kwargs):
        # regular polygon, similar to uniform but you can choose the angle.

        final_angle = 2 * np.pi + angle
        r = np.ones(self.n)
        theta = np.linspace(angle, final_angle - (2 * np.pi / self.n), self.n)

        return self._pol_to_cart(r, theta)

    def star(self, inner=0.5, outer=1):
        # Star pattern with n branches
        inner = inner / outer
        outer = 1

        r = np.tile([outer, inner], self.n)
        theta = np.linspace(0, (2 * np.pi) - (2 * np.pi / (self.n * 2)), self.n * 2)

        return self._pol_to_cart(r, theta)


class Parametric(_polar_coords):
    """Parametric distributions
    """

    def __init__(self, n=100):
        self.n = n

    def lissajous(self, a=3, b=4, delta=np.pi / 2, **kwargs):
        # Outputs x, y coordinates of a lissajous distribution

        t = np.arange(0, self.n, 1)
        x = np.sin(a * t + delta)
        y = np.sin(b * t)

        return x, y

    def sunflower(self, alpha=1):
        # Outputs x, y coordinates of a sunflower distribution

        indices = np.arange(0, self.n, dtype=float) + 0.5
        r = np.sqrt(indices / self.n)
        theta = np.pi * (1 + 5 ** 0.5) * indices / alpha

        return self._pol_to_cart(r, theta)

    def gear(self, gears=7, inner=0.8, outer=1, smooth=2):
        # Outputs x, y coordinates of a sunflower distribution

        inner = inner / outer
        outer = 1
        ppg = int(np.ceil(self.n / gears) / 2)
        one_gear = np.append(np.repeat(inner, ppg), np.repeat(outer, ppg))

        # 2 more gears to remove smoothness artefacts
        r = np.tile(one_gear, gears + 2)
        # We smooth here
        r2 = np.convolve(r, np.ones(smooth) / smooth, mode="same")
        # And we remove the border gears.
        cleaned = r2[len(one_gear) : -len(one_gear)]

        theta = np.linspace(0, 2 * np.pi, len(cleaned))

        x, y = self._pol_to_cart(cleaned, theta)

        return x, y


class RandomCoords(_polar_coords):
    """Outputs coordinates in a random shape
    """

    def __init__(self, n=100, ratio=1):
        self.n = n
        self.ratio = ratio

    def disk(self, distance="squared"):
        # Disk shape
        r = np.random.uniform(low=0, high=1, size=self.n)
        theta = np.random.uniform(low=0, high=2 * np.pi, size=self.n)

        # Squared distance correction or the density is not homogenous.
        if distance == "squared":
            r = np.sqrt(r)
        return self._pol_to_cart(r, theta)

    def normal(self, sd=0.3):
        # Gaussian shape
        r = np.sqrt(np.abs(np.random.normal(loc=0, scale=sd, size=self.n)))
        theta = np.random.uniform(low=0, high=2 * np.pi, size=self.n)

        return self._pol_to_cart(r, theta)

    def circular(self):
        # Random points on a circle
        r = np.ones((self.n))
        theta = np.random.uniform(low=0, high=2 * np.pi, size=self.n)

        return self._pol_to_cart(r, theta)

    def rectangular(self):
        # Random points in a square / rectangle pattern.
        x = np.random.uniform(low=-1, high=1, size=self.n)
        y = np.random.uniform(low=-1, high=1, size=self.n)

        return x, y / self.ratio
    
    def linear(self, angle=0):
        # Random points on a straight line
        r = np.random.uniform(low=0, high=1, size=self.n)
        theta = np.full(self.n, fill_value=(angle/360)*2*np.pi)
        
        return self._pol_to_cart(r, theta)
        


class Uniform:
    """Outputs coordinates in a regular shape
    """

    def __init__(self, n=100):
        self.n = n

    def square(self):
        # Square lattice.

        n = int(np.sqrt(self.n))
        dim = np.linspace(-1, 1, n)

        x, y = np.meshgrid(dim, dim)

        return np.ravel(x), np.ravel(y)

    def hexagon(self):
        """Hexagonal lattice. Taken from:
            https://laurentperrinet.github.io/sciblog/posts/2020-04-16-creating-an-hexagonal-grid.html
        """

        ratio = np.sqrt(3) / 2  # cos(60°)
        nx = int(np.sqrt(self.n / ratio))
        ny = self.n / nx

        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x = x * ratio
        x[::2, :] += (ratio) / 2

        x = x - np.max(x) / 2
        y = y - np.max(y) / 2

        x /= np.max(x)
        y /= np.max(y)

        return np.ravel(x), np.ravel(y)


class Noise:
    def __init__(self, steps=100, amount=0.3):

        self.steps = steps
        self.amount = amount

        lin = np.linspace(-1, 1, steps)
        self.xx, self.yy = np.meshgrid(lin, lin)

    def _return_points(self, data):
        p = self.amount * data
        r = np.random.random(p.shape)

        return (
            self.xx[np.where(r < p)],
            self.yy[np.where(r < p)],
        )

    def gaussian(self, sd=0.3):
        distrib = np.exp(
            -(self.xx ** 2 / (2 * sd ** 2) + (self.yy ** 2 / (2 * sd ** 2)))
        )

        return self._return_points(distrib)

    def perlin(self, grid=5, seed=48):
        # taken from https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy

        def lerp(a, b, x):
            "linear interpolation"
            return a + x * (b - a)

        def fade(t):
            "6t^5 - 15t^4 + 10t^3"
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        def gradient(h, x, y):
            "grad converts h to the right gradient vector and return the dot product with (x,y)"
            vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
            g = vectors[h % 4]
            return g[:, :, 0] * x + g[:, :, 1] * y

        lin = np.linspace(0, grid, self.steps)
        x, y = np.meshgrid(lin, lin)

        # permutation table
        np.random.seed(seed)
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.stack([p, p]).flatten()

        # coordinates of the top-left
        xi, yi = x.astype(int), y.astype(int)

        # internal coordinates
        xf, yf = x - xi, y - yi

        # fade factors
        u, v = fade(xf), fade(yf)

        # noise components
        n00 = gradient(p[p[xi] + yi], xf, yf)
        n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)

        # combine noises
        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)

        data = lerp(x1, x2, v)

        return self._return_points(data)


if __name__ == "__main__":
    # If run alone, simply plot the distributions as examples.

    distribs1 = [
        Spiral().golden(),
        Spiral().archimedean(),
        Spiral().quadratic(),
        Spiral().patterned(),
        Parametric().sunflower(),
        Parametric().lissajous(),
        Parametric().gear(),
        Circular().uniform(),
        Circular().polygon(),
        Circular().star(),
        RandomCoords().disk(),
        RandomCoords().normal(),
        RandomCoords().circular(),
        RandomCoords().rectangular(),
        Uniform().square(),
        Uniform().hexagon(),
    ]

    distribs2 = [
        Noise().gaussian(),
        Noise(amount=2).perlin(),
    ]

    names1 = [
        "Golden Spiral",
        "Archimedean Spiral",
        "Quadratic Spiral",
        "Patterned Spiral",
        "Sunflower",
        "Lissajous",
        "Gears",
        "Circular Uniform",
        "Polygon",
        "Star",
        "Random Disk",
        "Random Normal",
        "Random Circular",
        "Random Rectangular",
        "Uniform Square",
        "Uniform Hexagonal",
    ]

    names2 = [
        "Gaussian Noise",
        "Perlin Noise",
    ]

    fig, axes = plt.subplots(
        figsize=(12, 12), nrows=4, ncols=4, sharex=True, sharey=True
    )
    ax = axes.ravel()

    for i, (dist, name) in enumerate(zip(distribs1, names1)):

        ax[i].scatter(*dist, s=10)
        ax[i].set_aspect("equal")
        ax[i].set_title(name)
    fig.suptitle("Simple Distributions", fontsize=18)
    fig.tight_layout(pad=1.2)

    fig2, axes = plt.subplots(
        figsize=(12, 4), nrows=1, ncols=2, sharex=True, sharey=True
    )
    ax = axes.ravel()

    for i, (dist, name) in enumerate(zip(distribs2, names2)):

        ax[i].scatter(*dist, s=5)
        ax[i].set_aspect("equal")
        ax[i].set_title(name)
    fig2.suptitle("2D noise Distributions", fontsize=18)
    fig2.tight_layout(pad=1.2)
