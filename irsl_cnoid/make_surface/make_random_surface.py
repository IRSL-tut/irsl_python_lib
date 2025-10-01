import cv2
import random
import time
import numpy as np
from math import pi as PI
from cnoid.IRSLCoords import coordinates
import irsl_choreonoid.make_shapes as mkshapes

"""
Examples:
    >>>> from irsl_cnoid.make_surface.make_random_surface import MakeRandomTerrain
    >>>> mr = MakeRandomTerrain(8.0, 4.0) ## size (x, y) : 8.0[m] x 4.0[m]
    >>>> res = mr.makeRandomSurface(20, blur=0.1, sizeRange=(0.4, 3.0), heightRange=(-0.05, 0.05)) ## 20 objects on surface
    >>>> di=DrawInterface()
    >>>> di.clear()
    >>>> di.addObject(res)
    >>>> mkshapes.exportMesh("/tmp/surface.stlb", res) ## exporting stl-binary

"""

class MakeRandomTerrain(object):
    def __init__(self, sizeX, sizeY=None, resolution=0.005, resolution_Y=None):
        """
        Initializes the object with the specified dimensions, resolution, and random seed.

        Args:
            sizeX (float): The size of the surface along the X-axis.
            sizeY (float, optional): The size of the surface along the Y-axis. Defaults to None.
            resolution (float, optional): The resolution of the surface along the X-axis. Defaults to 0.005.
            resolution_Y (float, optional): The resolution of the surface along the Y-axis. If not provided, 
                it defaults to the value of `resolution`.

        Attributes:
            resX (float): The resolution of the surface along the X-axis.
            resY (float): The resolution of the surface along the Y-axis.
            random_seed (int): The random seed generated using the monotonic clock.

        Examples:
            >>>> from irsl_cnoid.make_surface.make_random_surface import MakeRandomTerrain
            >>>> mr=MakeRandomTerrain(8.0, 4.0)
            >>>> for i in range(20):
            >>>>     mr.addRandomTerrain(sizeRange=(0.4, 3.0), heightRange=(-0.05, 0.05))
            >>>> mr.gaussianBlur(0.1)
            >>>> res=mr.makeGridSurface()
            >>>> ## res=mr.makeRandomSurface(20, blur=0.1, sizeRange=(0.4, 3.0), heightRange=(-0.05, 0.05))
            >>>> di=DrawInterface()
            >>>> di.clear()
            >>>> di.addObject(res)

        """
        self.resX = resolution
        self.resY = resolution_Y if resolution_Y is not None else resolution
        self.random_seed = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        self.makeArray(sizeX, sizeY)
        random.seed( self.random_seed )
    def makeArray(self, sizeX, sizeY=None):
        """
        Creates a 2D array with specified dimensions and initializes it with zeros.

        Args:
            sizeX (float): The size of the array along the X-axis.
            sizeY (float, optional): The size of the array along the Y-axis. 
                If not provided, it defaults to the value of `sizeX`.

        Attributes:
            sizex (float): The size of the array along the X-axis.
            sizey (float): The size of the array along the Y-axis.
            isizex (int): The number of elements along the X-axis, calculated as `sizeX / resX` and rounded.
            isizey (int): The number of elements along the Y-axis, calculated as `sizeY / resY` and rounded.
            rawarray (numpy.ndarray): A 2D array of shape `(isizey, isizex)` initialized with zeros.
        """
        sizeY = sizeY if sizeY is not None else sizeX
        self.sizex=sizeX
        self.sizey=sizeY
        self.isizex = round(sizeX/self.resX)
        self.isizey = round(sizeY/self.resY)
        self.rawarray = np.zeros((self.isizey, self.isizex), dtype='float32')
    def addRandomTerrain(self, positionRange=None, sizeRange=(0.01, 2.0), heightRange=(-0.1, 0.1)):
        """
        Adds a random terrain feature to the surface.

        This method generates a random terrain feature with random position, size, 
        and height within the specified ranges and adds it to the surface.

        Args:
            positionRange (tuple, optional): Reserved for specifying the range of 
                positions (not currently used). Defaults to None.
            sizeRange (tuple, optional): A tuple specifying the range of sizes 
                (sx, sy) for the terrain feature. Defaults to (0.01, 2.0).
            heightRange (tuple, optional): A tuple specifying the range of heights 
                (hh) for the terrain feature. Defaults to (-0.1, 0.1).

        """
        #
        xx = self.isizex * self.resX * random.random()
        yy = self.isizey * self.resY * random.random()
        #
        sx = sizeRange[0] + (sizeRange[1] - sizeRange[0])*random.random()
        sy = sizeRange[0] + (sizeRange[1] - sizeRange[0])*random.random()
        #
        hh = heightRange[0] + (heightRange[1] - heightRange[0])*random.random()
        #
        self.addTerrain(xx, yy, sx, sy, hh)
    def addTerrain(self, x, y, sizex, sizey, height, gaussianBlur=None):
        """
        Adds a terrain feature to the surface by modifying the internal raw array.

        Args:
            x (float): The x-coordinate of the terrain's center in world units.
            y (float): The y-coordinate of the terrain's center in world units.
            sizex (float): The size of the terrain along the x-axis in world units.
            sizey (float): The size of the terrain along the y-axis in world units.
            height (float): The height of the terrain feature.
            gaussianBlur (None or tuple, optional): If specified, applies a Gaussian blur 
                with the given kernel size (width, height). Defaults to None.

        Notes:
            - The terrain is added to the `rawarray` attribute of the object.
            - The coordinates and sizes are converted to indices based on the resolution 
              of the surface (`resX` and `resY`).
            - The terrain is blurred using OpenCV's `cv2.blur` function.
        """
        ix = round(x/self.resX)
        iy = round(y/self.resY)
        sx = round(sizex/self.resX)
        sy = round(sizey/self.resY)
        if ix < 0:
            ix = 0
        if ix >= self.isizex:
            ix = self.isizex-1
        if iy < 0:
            iy = 0
        if iy >= self.isizey:
            iy = self.isizey-1
        #
        temp=np.zeros((self.isizey, self.isizex), dtype='float32')
        temp[iy, ix] = sx*sy*height
        temp=cv2.blur(temp, (sx, sy))
        #
        self.rawarray += temp
    def addBoxTerrain(self, x, y, sizex, sizey, height, gaussianBlur=None):
        ix = round(x/self.resX)
        iy = round(y/self.resY)
        sx = round(sizex/self.resX)
        sy = round(sizey/self.resY)
        if ix < 0:
            ix = 0
        if ix >= self.isizex:
            ix = self.isizex-1
        if iy < 0:
            iy = 0
        if iy >= self.isizey:
            iy = self.isizey-1
        temp=np.zeros((self.isizey, self.isizex), dtype='float32')
        cv2.rectangle(temp, (ix-sx, iy-sy), (ix+sx, iy+sy), (height), thickness=-1)
        self.rawarray += temp
    def addEllipseTerrain(self, x, y, sizex, sizey, angle, height, gaussianBlur=None):
        ix = round(x/self.resX)
        iy = round(y/self.resY)
        sx = round(sizex/self.resX)
        sy = round(sizey/self.resY)
        if ix < 0:
            ix = 0
        if ix >= self.isizex:
            ix = self.isizex-1
        if iy < 0:
            iy = 0
        if iy >= self.isizey:
            iy = self.isizey-1
        temp=np.zeros((self.isizey, self.isizex), dtype='float32')
        cv2.ellipse(temp, ((ix, iy), (sx, sy), angle), (height), thickness=-1)
        self.rawarray += temp
    def addCircleTerrain(self, x, y, radius, height, gaussianBlur=None):
        ix = round(x/self.resX)
        iy = round(y/self.resY)
        if ix < 0:
            ix = 0
        if ix >= self.isizex:
            ix = self.isizex-1
        if iy < 0:
            iy = 0
        if iy >= self.isizey:
            iy = self.isizey-1
        temp=np.zeros((self.isizey, self.isizex), dtype='float32')
        cv2.circle(temp, (ix, iy), round(radius / self.resX), (height), thickness=-1)
        self.rawarray += temp
    def gaussianBlur(self, sizex=0.02, sizey=None):
        """
        Applies a Gaussian blur

        This method blurs the `rawarray` attribute using a Gaussian kernel. The size
        of the kernel is determined by the `sizex` and `sizey` parameters, which are
        converted to kernel dimensions based on the resolution (`resX` and `resY`).
        The kernel dimensions are adjusted to ensure they are odd numbers, as required
        by the GaussianBlur function.

        Args:
            sizex (float, optional): The size of the Gaussian kernel in the x-direction.
                Defaults to 0.02.
            sizey (float, optional): The size of the Gaussian kernel in the y-direction.
                If not provided, it defaults to the value of `sizex`.

        Returns:
            None: The method modifies the `rawarray` attribute in place.
        """
        sizey = sizex if sizey is None else sizey
        gbx = round(sizex/self.resX)
        gby = round(sizey/self.resY)
        if gbx % 2 == 0:
            gbx += 1
        if gby % 2 == 0:
            gby += 1
        self.rawarray = cv2.GaussianBlur(self.rawarray, (gbx, gby), 0)
    def makeGridSurface(self, transformOrigin=True, **kwargs):
        """
        Generates a grid surface based on elevation data and applies transformations.

        This method creates an elevation grid using the specified dimensions and resolution,
        reshapes the raw elevation data, and applies a rotation and translation to the grid.

        Args:
            transformOrigin (boolean, default=True) : If True, transform the origin of the mesh. Otherwise, transform coordinates of viewing object

        Returns:
            object: The transformed elevation grid object.
        """
        res = mkshapes.makeElevationGrid(self.isizex, self.isizey, self.resX, self.resY,
                                         self.rawarray.reshape(self.isizex*self.isizey).tolist(), **kwargs)
        cds = coordinates()
        cds.rotate(PI/2, coordinates.X)
        cds.translate(np.array([-0.5*self.sizex, 0.5*self.sizey, 0]), wrt=coordinates.world)
        if transformOrigin:
            shape = mkshapes.extractShapes(res.target)[0][0]
            shape.mesh.transform(cds.cnoidPosition)
        else:
            res.newcoords(cds)
        return res
    def makeRandomSurface(self, numberOfBumps=10, blur=None, positionRange=None, sizeRange=(0.01, 2.0), heightRange=(-0.1, 0.1), transformOrigin=True, **kwargs):
        """
        Generates a random surface by adding multiple random terrain bumps and optionally applying a Gaussian blur.

        Args:
            numberOfBumps (int, optional): The number of random bumps to add to the surface. Defaults to 10.
            blur (float or None, optional): The standard deviation for Gaussian blur. If None, no blur is applied. Defaults to None.
            positionRange (tuple or None, optional): The range of positions for the bumps as (min, max). If None, the default range is used. Defaults to None.
            sizeRange (tuple, optional): The range of sizes for the bumps as (min, max). Defaults to (0.01, 2.0).
            heightRange (tuple, optional): The range of heights for the bumps as (min, max). Defaults to (-0.1, 0.1).
            transformOrigin (boolean, default=True) : If True, transform the origin of the mesh. Otherwise, transform coordinates of viewing object

        Returns:
            object: A grid surface object representing the generated random surface.
        """
        for i in range(numberOfBumps):
            self.addRandomTerrain(positionRange=positionRange, sizeRange=sizeRange, heightRange=heightRange)
        if blur is not None:
            self.gaussianBlur(blur)
        return self.makeGridSurface(transformOrigin=transformOrigin, **kwargs)
