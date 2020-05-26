import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import json
import open3d as o3d
from scipy.spatial import distance

class drone:

    def __init__(self, name="UAV_NO_NAME", maxView=50., pose=[0,0,0],
                 fence = None, map = None):

        self.name = name

        self.maxView = maxView

        self.pose = pose

        self.fence = fence

        self.map = map


    def isSafe(self, canditates=[0,0,0], safeDist=2.):

        dist = distance.cdist(self.map, canditates, 'euclidean')

        minDistCanditate = np.min(dist, axis=0)

        return minDistCanditate > safeDist


    def move(self, randomPointsSize=100, maxDist = 20):

        deltas = (np.random.random((randomPointsSize,3)) - 0.5)*maxDist

        canditates = self.pose + deltas

        inGeoFence = self.fence.isInside(canditates)
        canditates = canditates[inGeoFence]

        isSafe = self.isSafe(canditates)
        canditates = canditates[isSafe]

        import pdb; pdb.set_trace()

        # jEstimated = []
