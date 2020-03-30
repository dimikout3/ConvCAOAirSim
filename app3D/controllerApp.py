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


    def getCanditates(self, pertubations=70, saveLidar=False, minDist = 2.,
                            maxTravelTime=5., controllers=[]):

        lidarPoints = self.getLidarData(save_lidar=saveLidar)
        lidarPoints = self.clearLidarPoints(lidarPoints=lidarPoints,
                                            maxTravelTime=maxTravelTime,
                                            controllers=controllers)
        lidarPoints = self.addOffsetLidar(lidarPoints=lidarPoints)

        self.updateMultirotorState()
        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        xCanditate = xCurrent + (np.random.random(pertubations) - 1.)*maxTravelTime
        yCanditate = yCurrent + (np.random.random(pertubations) - 1.)*maxTravelTime
        zCanditate = zCurrent + (np.random.random(pertubations) - 1.)*maxTravelTime
        canditates = np.stack((xCanditate,yCanditate,zCanditate),axis=1)

        # inGeoFence = self.insideGeoFence(c = canditates, d = minDist)
        isSafeDist = self.isSafeDist(canditate = canditates,
                                     lidarPoints = lidarPoints,
                                     minDist = minDist)

        # geoFenceSafe = np.where(inGeoFence==True)[0]
        safeDistTrue = np.where(np.array(isSafeDist)==True)[0]

        # validCandidatesIndex = np.intersect1d(geoFenceSafe, safeDistTrue)

        xCanditate = xCanditate[safeDistTrue]
        yCanditate = yCanditate[safeDistTrue]
        zCanditate = zCanditate[safeDistTrue]

        canditatesPoints = np.stack((xCanditate,yCanditate,zCanditate), axis=1)

        return canditatesPoints


    def move(self,controllers=[], frontierCellsAttributed = []):

        meanFrontierCell = np.mean(frontierCellsAttributed, axis=0)
        canditatesPoints = self.getCanditates(controllers = controllers)

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val
        currentPos = np.array([xCurrent, yCurrent, zCurrent])

        distances = distance.cdist(canditatesPoints, [meanFrontierCell])

        argmin = np.argmin(distances, axis=1)

        x,y,z = canditatesPoints[0]
        task = self.moveToPosition(x, y, z, 2.0)

        return task


    def quit(self):

        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)

        state_file = os.path.join(self.parentRaw,
                                  self.getName(), f"state_{self.name}.pickle")
        pickle.dump(self.stateList,open(state_file,"wb"))
