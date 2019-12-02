import numpy as np
import time
import os
import airsim
from controller import controller


class evaluate:

    def __init__(self, KW=1):

        self.KW = KW


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

        return -self.KW*score
