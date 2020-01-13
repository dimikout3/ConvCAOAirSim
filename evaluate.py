import numpy as np
import time
import os
import airsim
from controller import controller
from scipy.spatial import ConvexHull
import utilities.AlphaShape as alphashape
# import alphashape
from scipy.spatial import distance

DEBUG_AlPHA = False
DEBUG_CLOUD = True
ALPHA = 2.
DETECTIONS_SIZE = 100

class evaluate:

    def __init__(self, KW={}):

        if KW == {}:
            KW['noRandomPoint'] = 1000
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

            self.KW['noRandomPoint'] = len(self.targetPoints)/25

            # cubicDistances = distance.cdist(self.targetPoints, self.targetPoints)
            # self.worstDistance = np.max(cubicDistances)
            # self.avgDistance = np.average(cubicDistances)

            worstX = self.fenceX + self.fenceR
            worstY = self.fenceY
            worstPoints = [worstX, worstY]

            self.worstDist =  np.sum( np.sqrt( np.sum( (self.targetPoints - worstPoints)**2,axis=1) ))



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


    def randomPointCloudCost(self, ego=None):

        distR2P = np.zeros((len(self.targetPoints),len(self.controllers)))
        cellsAssigned = np.zeros(len(self.controllers))

        j = 0.

        for i,(xTarget, yTarget) in enumerate(self.targetPoints):

            for r,drone in enumerate(self.controllers):

                if ego is None:
                    detections = drone.getPointCloudList()
                elif ego.getName() == drone.getName():
                    detections = drone.getPointCloudList(index = -2)
                else:
                    detections = drone.getPointCloudList()

                x,y,z,col = detections
                # using only x-y because we are on 2D space
                # TODO: adapt fro 3D enviroment movements
                detectionsPoints = np.stack((x,y), axis=1)

                target = np.array([xTarget, yTarget])
                # Distance between target points and each detection point (cloud points)
                # is computed, and only the closest is kept
                distR2P[i,r] = np.min( np.sqrt( np.sum( (detectionsPoints - target)**2,axis=1) ) )

            minDist = np.min(distR2P[i,:])
            imin = np.argmin(distR2P[i,:])

            cellsAssigned[imin] = cellsAssigned[imin] + 1

            # j += minDist
        # jList -> for each target point choose the min dist from each drone's detected point,
        # somehow similar to np.min(distR2P[i,:])
        jList = np.min(distR2P, axis=1)
        jSorted = np.argsort(jList)

        # how many drones have no detections assigned
        # TODO: check what happens if a drone has no detections assigned (rare), add a cost value
        zeroDetections = len(cellsAssigned) - np.count_nonzero(cellsAssigned)
        # jList[jSorted[-int(zeroDetections*( len(self.targetPoints)/len(self.controllers) ) ):-1]] = self.avgDistance

        return 1/(np.sum(jList) / self.worstDist)


    def pointInHull(self, hull, point):

        # import pdb; pdb.set_trace()
        # new_points = np.append(hull.points, point, axis=0)
        new_points = np.concatenate((hull.points, [point]), axis=0)
        new_hull = ConvexHull(new_points)

        if list(hull.vertices) == list(new_hull.vertices):
            return True
        else:
            return False


    def pointToHullDist(self, hull, point):

        hullEdgePoints = hull.points[hull.vertices]
        distances = np.sqrt(np.sum((hullEdgePoints - point)**2, axis=1))
        dist = np.min(distances)

        return dist


    def hullDistanceCost(self, ego=None):

        droneHull = []

        for i,drone in enumerate(self.controllers):

            if ego is None:
                detections = drone.getPointCloudList()
            elif ego.getName() == drone.getName():
                detections = drone.getPointCloudList(index = -2)
            else:
                detections = drone.getPointCloudList()

            x,y,z,col = detections
            # using only x-y because we are on 2D space
            # TODO: adapt fro 3D enviroment movements
            detectionsPoints = np.stack((x,y), axis=1)

            hull = ConvexHull(detectionsPoints)
            droneHull.append(hull)

        distR2P = np.zeros((len(self.targetPoints),len(self.controllers)))
        cellsAssigned = np.zeros(len(self.controllers))

        j = 0.

        for i,point in enumerate(self.targetPoints):

            for r,hull in enumerate(droneHull):

                if self.pointInHull(hull,point):
                    distR2P[i,r] = 0.
                else:
                    distR2P[i,r] = self.pointToHullDist(hull,point)

            minDist = np.min(distR2P[i,:])
            imin = np.argmin(distR2P[i,:])

            cellsAssigned[imin] = cellsAssigned[imin] + 1

            j += minDist


        for i in range(len(self.controllers)):
            if cellsAssigned[i] == 0:
                j += self.KW['noRandomPoint']*np.min(distR2P[:,i])

        return j


    def pointToPointCloudDist(self, pointCloud, point):

        distances = np.sqrt(np.sum((pointCloud - point)**2, axis=1))
        dist = np.min(distances)

        return dist


    def pointInAlphaShape(self, alpha, detectionPoints, point):

        new_points = np.concatenate((detectionPoints, [point]), axis=0)
        new_hull = alphashape.alphashape(new_points, ALPHA)

        if alpha == new_hull:
            return True
        else:
            return False

    def alphaShapeDistanceCost(self, ego=None):

        droneHullDetections = []

        for i,drone in enumerate(self.controllers):

            if DEBUG_AlPHA:
                print(f"Creating Polygon for {drone.getName()}")

            if ego is None:
                detections = drone.getPointCloudList()
            elif ego.getName() == drone.getName():
                detections = drone.getPointCloudList(index = -2)
            else:
                detections = drone.getPointCloudList()

            x,y,z,col = detections
            # using only x-y because we are on 2D space
            # TODO: adapt fro 3D enviroment movements
            detectionsPoints = np.stack((x,y), axis=1)

            detectionsPoints = detectionsPoints[np.random.randint(0,len(detectionsPoints),DETECTIONS_SIZE)]
            # hull = ConvexHull(detectionsPoints)
            hull = alphashape.alphashape(detectionsPoints, ALPHA)
            droneHullDetections.append([hull, detectionsPoints])

        distR2P = np.zeros((len(self.targetPoints),len(self.controllers)))
        cellsAssigned = np.zeros(len(self.controllers))

        j = 0.

        for i,point in enumerate(self.targetPoints):

            if DEBUG_AlPHA:
                print(f"Testing target point{i}")

            for r, (hull, detectionPoints) in enumerate(droneHullDetections):

                if alphashape.pointInAlphaShape(hull, detectionPoints, point, ALPHA):
                    distR2P[i,r] = 0.
                else:
                    distR2P[i,r] = self.pointToPointCloudDist(detectionPoints,point)

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
