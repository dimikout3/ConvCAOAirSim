import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import json
from scipy.spatial import Delaunay

class geoFence:

    def __init__(self, boundaries):

        self.minX = boundaries["Fence"]["minX"]
        self.maxX = boundaries["Fence"]["maxX"]

        self.minY = boundaries["Fence"]["minY"]
        self.maxY = boundaries["Fence"]["maxY"]

        self.minZ = boundaries["Fence"]["minZ"]
        self.maxZ = boundaries["Fence"]["maxZ"]

        # TODO: change this to itertools combinations
        p1 = [self.minX, self.minY, self.minZ]
        p2 = [self.minX, self.minY, self.maxZ]
        p3 = [self.minX, self.maxY, self.minZ]
        p4 = [self.minX, self.maxY, self.maxZ]
        p5 = [self.maxX, self.minY, self.minZ]
        p6 = [self.maxX, self.minY, self.maxZ]
        p7 = [self.maxX, self.maxY, self.minZ]
        p8 = [self.maxX, self.maxY, self.maxZ]
        pointsHull = [p1, p2, p3, p4, p5, p6, p7, p8]

        self.hull = Delaunay(pointsHull)


    def isInside(self, point):

        return self.hull.find_simplex(point) >=0


    def clearPointCloud(self, pointCloud):

        validPoints = self.isInside(pointCloud)

        return pointCloud[validPoints]
