# -*- coding: utf-8 -*-
"""
Functions are used to weight the sizes, alphas, etc... of the plots.
They have to return a single array normalized betwen 0 and 1.
"""

import numpy as np


class _BaseFunction:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        if len(x) != len(y):
            raise ValueError(
                f"x and y coordinates must have the same length but got {len(x)} and {len(y)}."
            )
            

    def _normalize(self, r):
        return (r - r.min()) / (r.max() - r.min())

    def _cart_to_pol(self, x, y):
        # convert cartesian coordinates to polar.
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.where(y >= 0, np.arccos(x / r), -np.arccos(x / r))

        return r, theta

    def _pol_to_cart(self, r, theta):
        # Convert polar coordinates to cartesian.
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return x, y


class Distance(_BaseFunction):
    """This class will output values between 0 and 1 depending on the distance 
    of the point from the centre of the distribution.
    """

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


class Angle(_BaseFunction):
    """This class will output weights between 0 and 1, depending on the theta 
    angle of the point.
    """

    def wave(self, waves=2):
        # waves depending on the theta angle.
        r, theta = self._cart_to_pol(self.x, self.y)
        theta = theta * waves

        value = (np.sin(theta) + 1) / 2
        value = np.nan_to_num(value, nan=0)

        return value

    def inc_uniform(self):
        # Increasing with the angle of the point in polar coordinates.
        _, theta = self._cart_to_pol(self.x, self.y)
        return self._normalize(theta)


class Neighbors(_BaseFunction):
    '''This class will output weights depending on the number of neighbors around each point.
    Neighbors can be taken from another distribution.
    '''
    
    
    def n_neighbors(self, x=None, y=None, dist=0.2):
        from scipy.spatial import cKDTree
        
        self.points = np.stack((self.x, self.y), axis=1)
        
        if (x is None) | (y is None):
            pop = np.stack((self.x, self.y), axis=1)
        
        else:
            pop = np.stack((x, y), axis=1)
            
        tree = cKDTree(pop)
        s = []
        for point in self.points:
            n = tree.query_ball_point(point, dist)
            s = np.append(s, len(n))
            
        return self._normalize(s)
        
    

class Modify(_BaseFunction):
    ''' Class used to modify already generated distributions.
    '''
    
    def normalize(self):
        return self._normalize(self.x), self._normalize(self.y)
    
    
    def rotate(self, angle=10, centre=(0, 0)):

        self.x += centre[0]
        self.y += centre[1]

        r, theta = self._cart_to_pol(self.x, self.y)
        theta = theta + angle / 360 * 2 * np.pi
        
        x, y = self._pol_to_cart(r, theta)
        
        x -= centre[0]
        y -= centre[1]

        return x, y
    
    
    def jitter(self, x=0.1, y=0.1):
        
        j_x = np.random.uniform(-x, x, size = len(self.x))
        j_y = np.random.uniform(-y, y, size = len(self.y))
        
        return self.x + j_x, self.y + j_y
    
    def interleave(self, x, y):
        if len(x) != len(y):
            raise ValueError(
                f"x and y coordinates must have the same length but got {len(x)} and {len(y)}."
            )
            
        
        if (len(x) != len(self.x)) | (len(y) != len(self.y)):
            raise ValueError(
                f"Arrays to interleave must have the same length but got ({len(self.x)}, {len(self.y)} and ({len(x)}, {len(y)}."
            )
            
        
        int_x = np.empty((self.x.size + x.size,), dtype=float)
        int_x[0::2] = self.x
        int_x[1::2] = x
        
        int_y = np.empty((self.y.size + y.size,), dtype=float)
        int_y[0::2] = self.y
        int_y[1::2] = y
        
        return int_x, int_y
        
        
if __name__ == "__main__":
    # If called by itself, demonstrate the use of the functions.
    from distributions import Parametric
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(figsize=(16, 4), ncols=4, nrows=1)
    ax = axes.ravel()
    [(x.axis("off"), x.set_aspect("equal"),) for x in ax]

    x, y = Parametric(n=500).sunflower()  # 500 points with sunflower distribution

    # Examples of use
    sizes = Distance(x, y).wave(waves=2)  # 2 waves depending on distance from centre
    sizes2 = Angle(x, y).wave(
        waves=3
    )  # 3 waves depending on the theta angle of the point

    ax[0].scatter(x, y, s=20, c="red")
    ax[0].set_title("Sunflower Distribution")

    ax[1].scatter(x, y, s=sizes * 20, c="red")
    ax[1].set_title("Sizes depending on distance")

    ax[2].scatter(x, y, s=sizes2 * 20, c="red")
    ax[2].set_title("Sizes depending on angle")

    ax[3].scatter(
        x, y, s=sizes * sizes2 * 20, c="red"
    )  # Simply multiply the sizes to combine them
    ax[3].set_title("Sizes depending on distance & angle")

    fig.suptitle(
        "Attributes can be modified by distance from centre and theta angle", fontsize=18
    )
    fig.tight_layout(pad=1.2)

    fig.show()

    # Demonstrating rotation of points.
    fig2, axes = plt.subplots(figsize=(16, 4), ncols=3, nrows=1)
    ax = axes.ravel()
    [(x.axis("off"), x.set_aspect("equal"),) for x in ax]

    x, y = Parametric(n=100).sunflower()  # 100 points with sunflower distribution

    angle = 10
    x2, y2 = Modify(x, y).rotate(angle=angle)

    ax[0].scatter(x, y, s=20, c="red")
    ax[0].set_title("Sunflower Distribution")

    ax[1].scatter(x, y, s=20, c="red")
    ax[1].scatter(x2, y2, s=20, c="black")
    ax[1].set_title(f"Points rotated by {angle} degrees")

    x3, y3 = Modify(x, y).jitter()

    ax[2].scatter(x, y, s=20, c="red")
    ax[2].scatter(x3, y3, s=20, c="black")
    ax[2].set_title("Points with added jitter")
    
    fig2.suptitle("Distributions can be further modified", fontsize=18)
    fig2.tight_layout(pad=1.2)

    fig2.show()
