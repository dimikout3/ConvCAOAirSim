import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import json
import open3d as o3d
from scipy.spatial import distance

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


ESTIMATORWINDOW = 30


class drone:

    def __init__(self, name="UAV_NO_NAME", maxView=50., pose=[],
                 orientation=[], fence = None, map = None):

        self.name = name

        self.maxView = maxView

        self.pose = np.array(pose)
        self.orientation = np.array(orientation)

        self.poseList = []
        self.Ji = []
        self.timeStep = 0

        self.fence = fence

        self.map = map

        DoF = self.pose.size + self.orientation.size
        # print(f"{self.name} has DoF={DoF}")
        self.model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                               ('linear', LinearRegression())])
        self.estimator = self.model.fit([np.random.uniform(0,1,DoF)],[np.random.uniform(0,1)])


    def isSafe(self, canditates=[0,0,0], safeDist=2.):

        dist = distance.cdist(self.map, canditates, 'euclidean')

        minDistCanditate = np.min(dist, axis=0)

        return minDistCanditate > safeDist


    def updateState(self, timeStep):

        self.timeStep =timeStep

        self.poseList.append(self.pose)


    def updateJi(self, Ji_in):

        self.Ji.append(Ji_in)


    def estimate(self,pose):

        return self.estimator.predict(pose)


    def updateEstimator(self):

        data = self.poseList
        weights = np.linspace(1,1,len(data[-ESTIMATORWINDOW:]))

        # print(f"{self.getName()} {data}")

        self.estimator = self.model.fit(data[-ESTIMATORWINDOW:],self.Ji[-ESTIMATORWINDOW:])


    def move(self, randomPointsSize=100, maxDist = 5):

        deltas = (np.random.random((randomPointsSize,3)) - 0.5)*maxDist

        canditates = self.pose + deltas

        inGeoFence = self.fence.isInside(canditates)
        canditates = canditates[inGeoFence]

        isSafe = self.isSafe(canditates)
        canditates = canditates[isSafe]

        jEstimated = self.estimate(canditates)

        argminJ = np.argmin(jEstimated)

        self.pose = canditates[argminJ]

        # jEstimated = []
