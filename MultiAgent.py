import setup_path
import airsim

import numpy as np
import os
import cv2
import time
from tqdm import tqdm
from controller import controller
import yoloDetector
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

CAM_YAW = -0.5
CAM_PITCH = 0.
CAM_ROOL = 0.

def monitor(droneList, posInd, timeInterval = 1, totalTime = 1):

    print(f"[MONITORING] position {posInd}")

    detector = yoloDetector.yoloDetector()

    for timeStep in tqdm(range(0,totalTime,timeInterval)):

        absoluteCoordinates = []

        for ctrl in controllers:

            ctrl.updateState(posInd, timeStep)
            responses = ctrl.getImages(save_raw=True)
            detections = ctrl.detectObjects(detector, save_detected=True)

        time.sleep(timeInterval)

# path expressed as x, y, z and speed
# PATH = {"Drone1":[(10,0,-10,5), (30,0,-10,5),(50,0,-10,5)],
#         "Drone2":[(0,10,-10,5), (0,30,-10,5),(0,50,-10,5)],
#         }
PATH = {"Drone1":[(x,-7.5,-12,5) for x in range(50,-50,-2)],
        "Drone2":[(x,-7.5,-8,5) for x in range(-50,50,2)],
        }

dronesID = list(PATH.keys())
wayPointsSize = len(PATH[dronesID[0]])
print(f"Detected {dronesID} with {wayPointsSize} positions")

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

controllers = []
for drone in dronesID:
    controllers.append(controller(client, drone))

# Setting Camera Orientation
for ctrl in controllers: ctrl.setCameraOrientation(CAM_YAW, CAM_PITCH, CAM_ROOL)

print("Taking off all drones")
tasks = []
for ctrl in controllers:
    t = ctrl.takeOff()
    tasks.append(t)
for t in tasks: t.join()

# print("Lifting all drones to specified Z altitude")
# tasks = []
# for ctrl in controllers:
#     t = ctrl.moveToZ(-10,2)
#     tasks.append(t)
# for t in tasks: t.join()

# Setting same yaw
tasks = []
for ctrl in controllers:
    t = ctrl.rotateToYaw(90)
    tasks.append(t)
print("Rotated succesfully")
# It does not work with .join() ... 
# for t in tasks: t.join()
time.sleep(10)
# wayPointsSize = 100
for positionIdx in range(0,wayPointsSize):
    tasks = []
    for ctrl in controllers:
        # t = ctrl.randomMoveZ()
        x,y,z,speed = PATH[ctrl.getName()][positionIdx]
        t = ctrl.moveToPostion(x,y,z,speed)
        tasks.append(t)

    for t in tasks: t.join()

    # Stabilizing the drones for better images
    tasks = []
    for ctrl in controllers:
        t = ctrl.stabilize()
        tasks.append(t)
    for t in tasks: t.join()

    for ctrl in controllers:
        state = ctrl.getState()

        x = state.kinematics_estimated.position.x_val
        y = state.kinematics_estimated.position.y_val
        z = state.kinematics_estimated.position.z_val
        _,_,yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

        print(f"{2*' '}[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{yaw:.2f})")

    monitor(dronesID, positionIdx)

print("\n[RESETING] to original state ....")
for ctrl in controllers: ctrl.quit()
client.reset()
