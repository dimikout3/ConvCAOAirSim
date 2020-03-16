# import setup_path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from controllerBase import controller

import airsim
import numpy as np

class controllerApp(controller):

    def __init__(self, clientIn, droneName, offSets, ip="1",
                 wayPointsSize=200, estimatorWindow=55):

        super().__init__(clientIn, droneName, offSets, ip=ip,
                         wayPointsSize = wayPointsSize,
                         estimatorWindow = estimatorWindow)

        self.descretePointCloud = []


    def updateDescretizator(self, discretizator_in):

        self.descretizator = discretizator_in


    def pointCloud2Descrete(self):

        x, y, z, colors = self.pointCloud[-1]

        data = np.stack((x,y,z),axis=1)
        print(f"[pointCloud2Descrete]")
        self.descretePointCloud.append( self.descretizator.descretize(data) )


    def connectIntermidiate(self):
        """Determine where the points between last descrete and UAV belong
           (Obstacles, Explored)"""
        pass


    def quit(self):
        pass
