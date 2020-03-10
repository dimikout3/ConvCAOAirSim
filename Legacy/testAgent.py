import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import setup_path
import airsim
import numpy as np
import time
from controller import controller
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# https://stackoverflow.com/questions/44758588/running-python-script-in-interactive-python-prompt-and-keep-the-variables

optionsIP = 1

OFFSETS = {"UAV1":[0,0,0]}

ip_id = f"127.0.0.{optionsIP}"
client = airsim.MultirotorClient(ip = ip_id)
client.confirmConnection()

drone = "UAV1"

ctrl = controller(client, drone, OFFSETS[drone], ip=optionsIP, wayPointsSize=100, estimatorWindow=100)
ctrl.takeOff()
# points[points[:,2]<1]

import yoloDetector
yolo = yoloDetector.yoloDetector()

import cv2
