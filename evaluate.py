import numpy as np
import time
import os
import airsim
from controller import controller


class evaluate:

    def __init__(self, KW={}):

        if KW == {}:
            KW['noRandomPoint'] = 100000
            KW['noDetectionsCost'] = 1
            self.KW = KW


    def setGeoFence(self, x=0, y=0, z=0, r=0):
        """ Evaluator know the limits of the enviroment, as shere"""

        self.fenceX = x
        self.fenceY = y
        self.fenceZ = z
        self.fenceR = r


    def randomPoints(self, pointsSize=100, setZ=False):
        """ Generate random points in the Geo Fence to use for exploration,
            similart to hold the line """

        if setZ:

            # TODO: adapt to sphere coordinates
            print(f" currently Drones move only on X-Y plane")

        else:

            r = self.fenceR * np.sqrt(np.random.rand(pointsSize))
            theta = np.random.rand(pointsSize) * 2 * np.pi

            x = self.fenceX + r * np.cos(theta)
            y = self.fenceY + r * np.sin(theta)

            self.targetPoints = np.stack((x,y), axis = 1)


    def randomPointsCost(self, ego=None):

        distR2P = np.zeros((len(self.targetPoints),len(self.controllers)))
        cellsAssigned = np.zeros(len(self.controllers))

        j = 0.

        for i,(xTarget, yTarget) in enumerate(self.targetPoints):

            for r,drone in enumerate(self.controllers):

                if ego is None:
                    positions = drone.getPositions()
                elif ego.getName() == drone.getName():
                    positions = drone.getPositions(index = -2)
                else:
                    positions = drone.getPositions()

                xDrone = positions.x_val
                yDrone = positions.y_val

                distR2P[i,r] = np.sqrt((xDrone-xTarget)**2 + (yDrone-yTarget)**2)

            minDist = np.min(distR2P[i,:])
            imin = np.argmin(distR2P[i,:])

            cellsAssigned[imin] = cellsAssigned[imin] + 1

            j += minDist


        for i in range(len(self.controllers)):
            if cellsAssigned[i] == 0:
                j += self.KW['noRandomPoint']*np.min(distR2P[:,i])

        return j


    def updateControllers(self, controllers):

        self.controllers = controllers


    def updateExcludedDict(self, excludedDict):

        self.excludedDict = excludedDict


    def updateDecectionsDict(self, detectionsDict):

        self.detectionsDict = detectionsDict


    def update(self, controllers=None, excludedDict=None, detectionsDict=None):

        if controllers != None:
            self.updateControllers(controllers)

        if excludedDict != None:
            self.updateExcludedDict(excludedDict)

        if detectionsDict != None:
            self.updateDecectionsDict(detectionsDict)


    def detectionsScore(self, ego="None", minusDuplicates=False):

        informationScore = 0.

        for ctrl in self.controllers:

            if isinstance(ego,str):
                index = -1
            else:
                if ctrl.getName() == ego.getName():
                    index = -2
                else:
                    index = -1

            score = ctrl.scoreExcludingDetections(index=index,
                                                  excludedList=self.excludedDict[ctrl.getName()],
                                                  minusDuplicates=minusDuplicates)
            informationScore += score

        return informationScore


    def noDetectionsCost(self, ego="None"):

        score = 0.

        for ctrl in self.controllers:

            if isinstance(ego,str):
                detectionsCoordinates = ctrl.getDetectionsCoordinates()
            else:
                if ctrl.getName() == ego.getName():
                    detectionsCoordinates = ctrl.getDetectionsCoordinates(index=-2)
                else:
                    detectionsCoordinates = ctrl.getDetectionsCoordinates()

            if detectionsCoordinates == []:
                # calculate the closest distance to a currently detcted object
                score += ctrl.getDistanceClosestDetection(self.detectionsDict)

        return -self.KW['noDetectionsCost']*score
