# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:31:10 2022

@author: Sylvain Rama
"""


import numpy as np
from dataclasses import dataclass
import re

from helpers import color_dict, polygon_dict

from PIL import Image
from PIL.ImageDraw import ImageDraw, _compute_regular_polygon_vertices

@dataclass
class PILe:
    
    ''' Basic data structure for PILes distributions.
    Parameters available:
        * coords: 2-tuple of np.ndarray(), x and y coordinates of the distribution.
        The distribution centre (0, 0) is at the centre of the image.
        Default: 
            (np.asarray([0]), np.asarray([0]))
        * shapes: str or list of str, name of the shapes to draw at the coordinates x, y.
        Default:
            'circle'
        * sizes: int or list of int, sizes of the shapes to draw, in number of pixels.
        Default: 
            20
        * colors: 3-tuple of int8 (R, G, B), or list of tuples [(R, G, B), ...]
        or string defining a color ('red') or list of strings defining a color ['red', 'blue', ...]
        or list of tuples and colors mixed [(255, 0, 0), 'blue', ...] defining
        the colors of the drawn shapes.
        Default: 
            (255, 0, 0)
        * alphas: int8 or list of int8 defining the alpha value of the drawn shapes.
        Default: 255
        * widths: int or list of int defining the width of the outline of the shape
        Default: 
            3
        * outlines: 3-tuple of int8 (R, G, B), or list of tuples [(R, G, B), ...]
        or string defining a color ('red') or list of strings defining a color ['red', 'blue', ...]
        or list of tuples and colors mixed [(255, 0, 0), 'blue', ...] defining 
        the color of the outlines of the drawn shapes.
        Default: 
            (0, 0, 0)
        * angles: int or list of int defining the rotation angle of the shape. 
        From 0 to 360 degrees.
        Default: 0
        * ratios: float or list of float, ratio width/height for defining an ellipse or a rectangle.
        Default: 
            1
        height & width, the dimensions of the final drawing, centered in the image.
        As the final coordinates will be centered by
                x = x * distribution.width / 2 + img.width / 2
                y = y * distribution.height / 2 + img.height / 2
        then using height, width = 2, 2 will ignore the scaling and centering of
        the drawing. Coordinates will still use (0, 0) as the centre of the image.
        
        Default values:
            height = 2
            width = 2
    '''
    
    coords = (np.asarray([0]), np.asarray([0]))
    height = 2
    width = 2
    shapes = 'circle'
    
    sizes = 20
    alphas = 255
    colors = (255, 0, 0)
    outlines = (0, 0, 0)
    widths = 3
    angles = 0
    ratios = 1 

# Regex for parsing the names of n-gons.
# Will match '3-gon', '4-gon', etc...     
n_gons_patterns = re.compile("^\d+-gon$")


class ImageDraws(ImageDraw):
    ''' Main class for drawing multiple shapes with a single call.
    Built on top of ImageDraw from PIL.    
    
    '''
    def __init__(self, img):
        ImageDraw.__init__(self, img, mode="RGBA")
        self.img = img

    def _return_proper_values(self, values, n=100):
        '''
        If a single element, converts it to a list of these elements, of length n.
        If a numpy array, coerce it to length n.
        Parameters
        ----------
        values : int, float, [int], [float] or np.asarray()
            
        n : list or np.asarray()
            number of elements to create in the list. The default is 100.

        Returns
        -------
        values : list or np.asarray() of length n

        '''

        if isinstance(values, np.ndarray):
            if values.size < n:
                return np.resize(values, n)

        if not isinstance(values, (list, np.ndarray)):
            values = [values]

        if len(values) < n:
            values = values * (n // len(values))

        return values

    def _return_proper_color(self, color, alpha):
        '''
        Check if the input color matches (R, G, B) format or is a known color string.
        And combines the value with alpha.
        
        Parameters
        ----------
        color : str or (R, G, B) tuple of int8
            The color as string or RGB tuple.
        alpha : int8
            Alpha value, from 0 to 255.

        Returns
        -------
        fill_color : tuple (R, G, B, A) of int8
            Final color value.

        '''

        if isinstance(color, str):
            color = color_dict[color.upper()]
        if color == None:
            fill_color = None
        else:
            if len(color) < 4:
                fill_color = (*color, alpha)
            else:
                fill_color = color
        return fill_color

    def _generic_drawer(self, shape, x, y, size, fill_color, outline, width, angle, ratio):
        # Will be used for circles, rectangles, ellipses, squares.
        # PIL does not manage alpha channel correctly when drawing a shape.
        # We have to draw the shape on another image and use alpha_composite to paste this temp image.
        
        # We have to manage the width in the image size, as it is an external width:
        # a circle of radius 10 with a line width of 1 has a full radius of 11.
        tmp = Image.new("RGBA", (int(size*2*ratio + width*2+1), int(size*2 + width*2+1)),
                        (255, 255, 255, 0))
        tmp_draw = ImageDraw(tmp)
        # Simply get the method from the class and use it.
        getattr(tmp_draw, shape)(
            (0, 0, int(size*2*ratio + width), int(size*2 + width)), fill=fill_color, outline=outline, width=width)

        # The generic drawer does not manage rotations, adding it here.
        if angle != 0:
            tmp = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Correcting for the centre.
        tmp_x, tmp_y = tmp.size
        tmp_x /= 2
        tmp_y /= 2

        # And pasting with alpha.
        self.img.alpha_composite(tmp, (int(x-tmp_x + width/2), int(y-tmp_y + width/2)))
        
    def regular_polygon(
        self, bounding_circle, n_sides, rotation=0, fill=None, outline=None, width=1
    ):
        """I overclassed the original PIL function, as it does not allow different line widths"""
        xy = _compute_regular_polygon_vertices(bounding_circle, n_sides, rotation)
        self.polygon(xy, fill, outline, width)
        
        
    def _draw_polygon(self, x, y, size, n_sides, fill_color, outline, width, angle, ratio):
        
        # Funny one: rectangle & ellipse drawers are defined by their bounding boxes
        # but polygon drawer is defined by centre and radius of the inscribed circle.
        # So a square is drawn 1.424 bigger than a 4-gon. Correcting this here.
        
        size *= 1.424
        
        tmp = Image.new("RGBA", (int(size*2*ratio + width), int(size*2 + width)),
                        (255, 255, 255, 0))
        tmp_draw = ImageDraws(tmp)
        
        tmp_x, tmp_y = tmp.size
        tmp_x /= 2
        tmp_y /= 2
        # The drawer manages rotations by itself, no need to add it.
        tmp_draw.regular_polygon((tmp_x, tmp_y, size), n_sides, rotation=angle, fill=fill_color, outline=outline, width=width)
       
        # And pasting with alpha.
        self.img.alpha_composite(tmp, (int(x-tmp_x + width/2), int(y-tmp_y + width/2)))
        

    def DrawShapes(self, params=PILe):
        # Main method to draw multiple shapes in one call.

    
        # Checking if x and y are numpy arrays.
        xs, ys = params.coords
        if not isinstance(xs, np.ndarray):
            xs = np.asarray(xs)
        if not isinstance(ys, np.ndarray):
            ys = np.asarray(ys)
            
        # Building all the arrays of parmeters with proper length.
        shapes = self._return_proper_values(params.shapes, len(xs))
        sizes = self._return_proper_values(params.sizes, len(xs))
        alphas = self._return_proper_values(params.alphas, len(xs))
        colors = self._return_proper_values(params.colors, len(xs))
        outlines = self._return_proper_values(params.outlines, len(xs))
        widths = self._return_proper_values(params.widths, len(xs))
        widths = [int(x) for x in widths]
        angles = self._return_proper_values(params.angles, len(xs))
        ratios = self._return_proper_values(params.ratios, len(xs))
        
        # Scaling x & y and centering them in the image.
        xs = xs * (params.width) / 2 + self.img.width / 2
        ys = ys * (params.height) / 2 + self.img.height / 2
        
        all_params = zip(shapes, xs, ys, sizes, alphas, colors, outlines, widths, angles, ratios)
        

        for shape, x, y, size, alpha, fill_color, outline, width, angle, ratio in all_params:

            fill_color = self._return_proper_color(fill_color, alpha)

            outline = self._return_proper_color(outline, alpha)
            
            # Basic drawers
            if shape in ['rectangle', 'ellipse']:
                self._generic_drawer(
                    shape, x, y, size, fill_color, outline, width, angle, ratio)
                
            if shape == 'circle':
                shape = 'ellipse'
                ratio = 1
                angle = 0
                self._generic_drawer(
                    shape, x, y, size, fill_color, outline, width, angle, ratio)
 
            # Shortcuts if you don't want to call a triangle '3-gon'.
            if shape in polygon_dict:
                n_sides = polygon_dict[shape]
                ratio = 1
                
                self._draw_polygon(x, y, size, n_sides, fill_color, outline, width, angle, ratio)
                
            if n_gons_patterns.match(shape):
                n_sides = int(shape.split('-')[0])
                ratio = 1

                self._draw_polygon(x, y, size, n_sides, fill_color, outline, width, angle, ratio)
                

    def DrawLines(self, params=PILe, closed=False):
        # Same as DrawShapes, but for lines.       

        xs, ys = params.coords
        
        if not isinstance(xs, np.ndarray):
            xs = np.asarray(xs)
        if not isinstance(ys, np.ndarray):
            ys = np.asarray(ys)
            
        outlines = self._return_proper_values(params.outlines, len(xs))
        widths = self._return_proper_values(params.widths, len(xs))
        widths = [int(x) for x in widths]
        
        alphas = self._return_proper_values(params.alphas, len(xs))
        
        xs = xs * (params.width) / 2 + self.img.width / 2
        ys = ys * (params.height) / 2 + self.img.height / 2
        
        # We may want to close the figure, thus drawing a line between 
        # the last and the first point.
        if closed:
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])
   
        for i in range(len(xs)-1):
            
            x1, x2 = xs[i], xs[i+1]
            y1, y2 = ys[i], ys[i+1]
            
            width = widths[i]
            outline = outlines[i]
            alpha = alphas[i]
            outline = self._return_proper_color(outline, alpha)
            
            tmp_width = abs(x1 - x2) + 2*width
            tmp_height = abs(y1 - y2) + 2*width

            tmp = Image.new("RGBA", (int(tmp_width), int(tmp_height)),
                            (255, 255, 255, 0))
            tmp_draw = ImageDraw(tmp)

            tmp_draw.line((width, width, tmp_width-width, tmp_height-width),
                          fill=outline, width=width)
            
            # Annoying thing to draw lines in any direction and rotate them in any angle.
            # Instead of computing the angle, I flip the image if needed.
            corner_x, corner_y = x1 - width, y1 - width

            if x1 > x2:
                tmp = tmp.transpose(Image.FLIP_LEFT_RIGHT)
                corner_x = x2 - width

            if y1 > y2:
                tmp = tmp.transpose(Image.FLIP_TOP_BOTTOM)
                corner_y = y2 - width

            self.img.alpha_composite(tmp, dest=(int(corner_x), int(corner_y)))

