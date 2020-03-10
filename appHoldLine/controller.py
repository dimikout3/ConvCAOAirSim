# import setup_path
import os
import cv2
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

ESTIMATORWINDOW = 40

class controller:

    def __init__(self, droneName, ip="1"):

        self.name = droneName
        self.ip = ip

        self.model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                               ('linear', LinearRegression())])
        self.estimator = self.model.fit([np.random.uniform(0,1,2)],[np.random.uniform(0,1)])

        self.Ji = []
        self.timeStep = 0

        np.random.seed()
        self.currentX = np.random.uniform(0.2,0.8)
        self.currentY = np.random.uniform(0.2,0.8)

        self.xList = []
        self.yList = []

    def getName(self):

        return self.name


    def updateJi(self, Ji_in):

        self.Ji.append(Ji_in)


    def getJi(self, index=-1):

        return self.Ji[index]


    def setGeoFence(self, x=1, y=1):
        """Applying geo fence as a square (0,0,x,y) """

        self.fenceX = x
        self.fenceY = y


    def insideGeoFence(self, x, y):

        if x>self.fenceX or x<0:
            return False
        elif y>self.fenceY or y<0:
            return False

        return True


    def move(self, randomPointsSize=100, maxDelta = 0.01, plot=True):

        deltaX = np.random.uniform(-maxDelta,maxDelta,randomPointsSize)
        deltaY = np.random.uniform(-maxDelta,maxDelta,randomPointsSize)

        xCanditateList = []
        yCanditateList = []
        jEstimated = []

        for i in range(randomPointsSize):

            xCanditate = self.currentX + deltaX[i]
            yCanditate = self.currentY + deltaY[i]

            inGeoFence = self.insideGeoFence(xCanditate, yCanditate)

            if inGeoFence:
                xCanditateList.append(xCanditate)
                yCanditateList.append(yCanditate)
                jEstimated.append(self.estimate(xCanditate,yCanditate))

        tartgetPointIndex = np.argmin(jEstimated)

        self.currentX = xCanditateList[tartgetPointIndex]
        self.currentY = yCanditateList[tartgetPointIndex]

        if plot:
            self.plotCanditates(xCanditateList, yCanditateList, jEstimated)


    def plotCanditates(self, xCanditateList, yCanditateList, jEstimated):

        canditate_folder = os.path.join(os.getcwd(), f"results_{self.ip}",
                                        f"canditates", f"{self.getName()}")
        try:
            os.makedirs(canditate_folder)
        except OSError:
            if not os.path.isdir(canditate_folder):
                raise

        plt.scatter(xCanditateList, yCanditateList, c=jEstimated)

        # plt.xlim(0,1)
        # plt.ylim(0,1)

        plt.colorbar()

        plt.tight_layout()
        outFile = os.path.join(canditate_folder, f"canditates_{self.timeStep}.png")
        plt.savefig(outFile)
        plt.close()


    def estimate(self,x,y):

        return float(self.estimator.predict([[x,y]]))


    def updateEstimator(self):

        data = np.stack((self.xList,self.yList),axis=1)
        weights = np.linspace(1,1,len(data[-ESTIMATORWINDOW:]))

        # print(f"{self.getName()} {data}")

        self.estimator = self.model.fit(data[-ESTIMATORWINDOW:],self.Ji[-ESTIMATORWINDOW:])


    def updateState(self, timeStep):

        self.timeStep =timeStep

        self.xList.append(self.currentX)
        self.yList.append(self.currentY)


    def getPositions(self, index=-1):

        if abs(index)>len(self.xList):
            return self.xList[-1], self.yList[-1]
        else:
            return self.xList[index], self.yList[index]
