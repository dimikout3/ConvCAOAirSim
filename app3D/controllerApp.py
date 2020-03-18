# import setup_path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from controllerBase import controller

import airsim
import numpy as np
from scipy.spatial.distance import cdist

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


    def quit(self):
        pass
