# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:25:14 2022

@author: Sylvain


Distributions functions must return  x, y coordinates in cartesian form, 
normalized between -1 and 1.
"""


import numpy as np
import matplotlib.pyplot as plt



class _polar_coords:
    def __init__(self, r, theta):
        self.r = r
        self.theta = theta

    def _pol_to_cart(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return x, y

    def _normalize(self, r):
        return (r - r.min()) / (r.max() - r.min())


class Spiral(_polar_coords):
    def __init__(self, n=100, turns=3):
        self.n = n
        self.turns = turns
        self._thetas = np.linspace(0, self.turns * np.pi, self.n)
        self._step = 1 / (turns * 2 * np.pi)
        
   
    # To use np.pi yields strange results...
    def golden(self, a=1, b=3.1415 / 2.0, **kwargs):

        r = a * np.exp(self._thetas * np.cos(b))
        r = self._normalize(r)

        return self._pol_to_cart(r, self._thetas)

    def archimedean(self, arc=1, sep=1, **kwargs):

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
    
    def patterned(self, loops=3, segments=7, inner=0.1, outer=1):
        
        inner = inner / outer
        outer = 1
        
        segments = loops*segments+1
        theta = [arc/(segments-1) * np.pi * 2.0 * loops for arc in range(int(segments))]
        
        r = np.linspace(inner, outer, segments)
        
        return self._pol_to_cart(r, theta)

    def quadratic(self, k=13):

        n = np.linspace(0, self.n, num=self.n)

        pn = (((k-2)*n*(n+1))/2 - (k-3)*n)

        theta = 2*np.pi*np.sqrt(pn)
        r = self._normalize(np.sqrt(pn))

        return self._pol_to_cart(r, theta)

class Circular(_polar_coords):
    def __init__(self, n=8):
        self.n = n

    def uniform(self, turns=1):

        r = np.ones((self.n))
        theta = np.linspace(0, (2 * np.pi * turns) -
                            (2 * np.pi / self.n), self.n)

        return self._pol_to_cart(r, theta)

    def polygon(self, angle=0, **kwargs):

        final_angle = 2 * np.pi + angle
        r = np.ones(self.n)
        theta = np.linspace(angle, final_angle - (2 * np.pi / self.n), self.n)

        return self._pol_to_cart(r, theta)
    
    def star(self, inner=0.5, outer=1):
        inner = inner / outer
        outer = 1
        
        r = np.tile([outer, inner], self.n)
        theta = np.linspace(0, (2*np.pi) - (2*np.pi / (self.n*2)), self.n*2)
        
        return self._pol_to_cart(r, theta)
        
class Parametric(_polar_coords):
    def __init__(self, n=100):
        self.n = n

    def lissajous(self, a=3, b=4, delta=np.pi/2, **kwargs):

        t = np.arange(0, self.n, 1)
        x = np.sin(a * t + delta)
        y = np.sin(b * t)

        return x, y
    
    def sunflower(self, alpha=1):
        indices = np.arange(0, self.n, dtype=float) + 0.5
        r = np.sqrt(indices / self.n)
        theta = np.pi * (1 + 5 ** 0.5) * indices / alpha

        return self._pol_to_cart(r, theta)

class RandomCoords(_polar_coords):
    def __init__(self, n=100, ratio=1):
        self.n = n
        self.ratio = ratio
        
    def disk(self, distance="squared"):

        r = np.random.uniform(low=0, high=1, size=self.n)
        theta = np.random.uniform(low=0, high=2 * np.pi, size=self.n)

        if distance == "squared":
            r = np.sqrt(r)
        
        return self._pol_to_cart(r, theta)
    
    def normal(self, sd=0.3, **kwargs):
        # sd=0.3 to constrict the gaussian between -1 and 1
        r = np.sqrt(np.abs(np.random.normal(loc=0, scale=sd, size=self.n)))
        theta = np.random.uniform(low=0, high=2 * np.pi, size=self.n)

        return self._pol_to_cart(r, theta)
    
    def circular(self):
        r = np.ones((self.n))
        theta = np.random.uniform(low=0, high=2 * np.pi, size=self.n)

        return self._pol_to_cart(r, theta)
    
    def rectangular(self):
        x = np.random.uniform(low=-1, high=1, size=self.n)
        y = np.random.uniform(low=-1, high=1, size=self.n)

        return x, y / self.ratio
        
        
class Uniform: # Not finished
    def __init__(self, n=100):
        self.n = n
        

    def rectangular(self):
        
        n = int(np.sqrt(self.n))
        dim = np.linspace(-1, 1, n)
                
        grid = np.meshgrid(dim, dim)
        
        return grid[0], grid[1]
    
    
    def disk(self):
        x, y = self.rectangular()
        mask = np.sqrt(x**2 + y**2) < 1
        
        return x[mask], y[mask]
        

if __name__ == '__main__':
    
    distribs = [Spiral().golden(), Spiral().archimedean(), Spiral().quadratic(), Spiral().patterned(),
                Parametric().sunflower(), Parametric().lissajous(), 
                Circular().uniform(), Circular().polygon(), Circular().star(),
                RandomCoords().disk(), RandomCoords().normal(), RandomCoords().circular(), RandomCoords().rectangular(),
                Uniform().rectangular(), Uniform(300).disk()]
    
    names = ['Golden Spiral', 'Archimedean Spiral', 'Quadratic Spiral', 'Patterned Spiral',
             'Sunflower', 'Lissajous',
             'Circular Uniform', 'Polygon', 'Star',
             'Random Disk', 'Random Normal', 'Random Circular', 'Random Rectangular',
             'Uniform Rectangular', 'Uniform Disk']
    fig, axes = plt.subplots(figsize=(16, 16), nrows=3, ncols=5, sharex=True, sharey=True)
    ax = axes.ravel()
    
    for i, (dist, name) in enumerate(zip(distribs, names)):
        
        ax[i].scatter(*dist)
        ax[i].set_aspect('equal')
        ax[i].set_title(name)
        
    
