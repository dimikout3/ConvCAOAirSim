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
