# import setup_path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from controllerBase import controller

import airsim
import numpy as np
from scipy.spatial import distance

class controllerApp(controller):

    def __init__(self, clientIn, droneName, offSets, ip="1",
                 wayPointsSize=200, estimatorWindow=55,
                 maxDistView = 10.0):

        super().__init__(clientIn, droneName, offSets, ip=ip,
                         wayPointsSize = wayPointsSize,
                         estimatorWindow = estimatorWindow,
                         maxDistView = maxDistView)

        self.descretePointCloud = []

        self.attributed = []


    def updateDescretizator(self, discretizator_in):

        self.descretizator = discretizator_in


    def pointCloud2Descrete(self):

        x, y, z, colors = self.pointCloud[-1]

        data = np.stack((x,y,z),axis=1)

        self.descretePointCloud.append( self.descretizator.descretize(data) )


    def connectIntermidiate(self, line_points=30):

        """Determine where the points between last descrete and UAV belong
           (Obstacles, Explored), using points as lines! """

        x, y, z, colors = self.pointCloud[-1]

        points = np.stack((x,y,z),axis=1)

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val
        uav = np.array([xCurrent, yCurrent, zCurrent])

        lines = np.linspace(points, uav, line_points)
        line_points  = np.reshape(lines, (lines.shape[0]*lines.shape[1],3))

        descreteLinePoints = self.descretizator.descretize(line_points).T

        self.xVoxels = descreteLinePoints[0]
        self.yVoxels = descreteLinePoints[1]
        self.zVoxels = descreteLinePoints[2]


    def connectIntermidiateDist(self, map=None, minDist=1., as_true=True):

        """Determine where the points between last descrete and UAV belong
           (Obstacles, Explored)"""

        x, y, z, colors = self.pointCloud[-1]

        points = np.stack((x,y,z),axis=1)

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val
        uav = np.array([xCurrent, yCurrent, zCurrent])

        i = np.meshgrid(np.arange(self.descretizator.discreteX),
                        np.arange(self.descretizator.discreteY),
                        np.arange(self.descretizator.discreteZ))
        xVoxels = np.reshape(i[0],i[0].size)
        yVoxels = np.reshape(i[1],i[1].size)
        zVoxels = np.reshape(i[2],i[2].size)
        voxels = np.stack((xVoxels, yVoxels, zVoxels),axis=1)

        # pointRepeated -> [p1,p2,p3, p1,p2,p3, p1,p2,p3 ...]
        pointsRepeated = np.tile(points,(voxels.shape[0],1))
        # voxelsRepeated -> [v1,v1,v1, v2,v2,v2, v3,v3,v3 ...]
        voxelsRepeated = np.repeat(voxels,points.shape[0],axis=0)

        # BUG: This is wrong ... instead of line we need LINE SEGMENT ...
        dist = np.linalg.norm( np.cross(pointsRepeated-uav, voxelsRepeated-uav),axis=1) / np.linalg.norm(uav - pointsRepeated, axis=1)
        dist = np.reshape(dist, (voxels.shape[0], points.shape[0]))
        dist = np.min(dist, axis=1)

        # inside -> voxels which are inside the line from UAV to lidar/depth point/pixel
        inside = np.where(dist < minDist)

        self.xVoxels = xVoxels[inside]
        self.yVoxels = yVoxels[inside]
        self.zVoxels = zVoxels[inside]


    def getIntermediate(self):

        return self.xVoxels, self.yVoxels, self.zVoxels


    def insideGeoFence(self, c=0, d=0):
        """ Not needed since Frontier Cells will be inside the GeoFence anyway"""
        pass


    def isSafeDist(self, canditates=[], lidarPoints=[], minDist=1.,
                         pathSize=10, multiplier=1.5):


        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        canditatesTrue = []

        # TODO: Avoid this loop if possible

        for ind, (x, y, z) in enumerate(canditates):

            xInter = np.linspace(xCurrent, x, pathSize)
            yInter = np.linspace(yCurrent, y, pathSize)
            zInter = np.linspace(zCurrent, z,  pathSize)

            pathPoints = np.stack((xInter, yInter, zInter),axis=1)

            dist = distance.cdist(lidarPoints, pathPoints)

            min = np.min(dist)

            if min>=minDist:

                canditatesTrue.append(ind)

        return canditatesTrue


    def getCanditates(self, pertubations=300, saveLidar=False, minDist = 1.,
                            maxTravelTime=3., controllers=[]):

        lidarPoints = self.getLidarData(save_lidar=saveLidar)
        lidarPoints = self.clearLidarPoints(lidarPoints=lidarPoints,
                                            maxTravelTime=maxTravelTime,
                                            controllers=controllers)
        lidarPoints = self.addOffsetLidar(lidarPoints=lidarPoints)

        self.updateMultirotorState()
        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        for helperIncreased in np.linspace(1,5,40):

            xCanditate = xCurrent + (np.random.random(pertubations) - .5)*2*maxTravelTime*helperIncreased
            yCanditate = yCurrent + (np.random.random(pertubations) - .5)*2*maxTravelTime*helperIncreased
            # zCanditate = zCurrent + (np.random.random(pertubations) - .5)*maxTravelTime*helperIncreased
            zCanditate = np.repeat(zCurrent,pertubations)
            canditates = np.stack((xCanditate,yCanditate,zCanditate),axis=1)

            # inGeoFence = self.insideGeoFence(c = canditates, d = minDist)
            isSafeDist = self.isSafeDist(canditates = canditates,
                                         lidarPoints = lidarPoints,
                                         minDist = minDist)

            # geoFenceSafe = np.where(inGeoFence==True)[0]
            safeDistTrue = np.where(np.array(isSafeDist)==True)[0]

            if safeDistTrue.size == 0:
                print(f"[CANDITATES] {self.getName()} found canditates with helperIncreased={helperIncreased}")
            else:
                break
            # validCandidatesIndex = np.intersect1d(geoFenceSafe, safeDistTrue)

        if safeDistTrue.size == 0:
            import pdb; pdb.set_trace()

        xCanditate = xCanditate[safeDistTrue]
        yCanditate = yCanditate[safeDistTrue]
        zCanditate = zCanditate[safeDistTrue]

        canditatesPoints = np.stack((xCanditate,yCanditate,zCanditate), axis=1)

        return canditatesPoints


    def canditatesInsidePseudoLidar(self, canditates, hullList):

        validIndexes = np.array([])

        for hull in hullList:

            currentValid = np.where( hull.find_simplex(canditates)>=0 )

            validIndexes = np.append(validIndexes, currentValid)

        # import pdb; pdb.set_trace()
        return canditates[np.unique(validIndexes).astype(int)]


    def getCanditatesPseudoLidar(self, pertubations=600, saveLidar=False, minDist = 2.,
                            maxTravelTime=3., controllers=[]):

        lidarPoints, hullList = self.getPseudoLidar(size=3000)
        lidarPoints = self.clearLidarPoints(lidarPoints=lidarPoints,
                                            maxTravelTime=maxTravelTime,
                                            controllers=controllers)

        self.updateMultirotorState()
        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        xCanditate = xCurrent + (np.random.random(pertubations) - .5)*2*maxTravelTime
        yCanditate = yCurrent + (np.random.random(pertubations) - .5)*2*maxTravelTime
        # zCanditate = zCurrent + (np.random.random(pertubations) - .5)*maxTravelTime
        zCanditate = np.repeat(zCurrent,pertubations)
        canditates = np.stack((xCanditate,yCanditate,zCanditate),axis=1)

        canditates = self.canditatesInsidePseudoLidar(canditates, hullList)

        isSafeDist = self.isSafeDist(canditates = canditates,
                                     lidarPoints = lidarPoints,
                                     minDist = minDist)

        # geoFenceSafe = np.where(inGeoFence==True)[0]
        safeDistTrue = np.where(np.array(isSafeDist)==True)[0]

        # validCandidatesIndex = np.intersect1d(geoFenceSafe, safeDistTrue)

        # import pdb; pdb.set_trace()
        if safeDistTrue.size == 0:
            import pdb; pdb.set_trace()

        xCanditate = xCanditate[safeDistTrue]
        yCanditate = yCanditate[safeDistTrue]
        zCanditate = zCanditate[safeDistTrue]

        canditatesPoints = np.stack((xCanditate,yCanditate,zCanditate), axis=1)

        return canditatesPoints


    def move(self,controllers=[], frontierCellsAttributed = [], moveToClosestFrontier=True):

        frontierCellsAttributed = frontierCellsAttributed[0]

        self.attributed.append(frontierCellsAttributed)

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val
        currentPos = np.array([xCurrent, yCurrent, zCurrent])

        if moveToClosestFrontier:

            # import pdb; pdb.set_trace()
            distances = distance.cdist(frontierCellsAttributed, [currentPos])
            argmin = np.argmax(distances)
            meanFrontierCell = frontierCellsAttributed[argmin]

        else:

            meanFrontierCell = np.mean(frontierCellsAttributed, axis=0)

        print(f"[MOVE] {self.getName()} meanFrontierCell={meanFrontierCell}")

        # canditatesPoints = self.getCanditates(controllers = controllers)
        # canditatesPoints = self.getCanditatesPseudoLidar(controllers = controllers)
        canditatesPoints = self.getCanditatesSegmented(maxDistTravel=2.,
                                                       size=500,
                                                       safeDist=2.)

        # import pdb; pdb.set_trace()

        distances = distance.cdist(canditatesPoints, [meanFrontierCell])

        argmin = np.argmin(distances)

        x,y,z = canditatesPoints[argmin]
        print(f"[MOVE] {self.getName()} toward target points (x:{x}, y:{y}, z:{z})")
        task = self.moveToPositionYawModeAsync(x, y, z, 2.0)

        return task


    def quit(self):

        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)

        state_file = os.path.join(self.parentRaw,
                                  self.getName(), f"state_{self.name}.pickle")
        pickle.dump(self.stateList,open(state_file,"wb"))
