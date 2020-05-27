import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import json
import open3d as o3d
from drone import drone
from geoFence import geoFence
from scipy.spatial import distance


# Loading App settings from json file
appSettings = json.load(open('appSettings.json','r'))

SWARM_SIZE = appSettings["UAVS"]["SwarmSize"]
MAX_VIEW = appSettings["UAVS"]["MaxView"]
POSE_X = appSettings["UAVS"]["Positions"]["X"]
POSE_Y = appSettings["UAVS"]["Positions"]["Y"]
POSE_Z =appSettings["UAVS"]["Positions"]["Z"]

wayPointsSize = 200


def loadMap():

    return np.load(appSettings["MapPath"])


def calculateCostJ():

    global costJ, controllers, map

    # ___________ Calculating Cost J ____________
    dronesPositions = []
    for ctrl in controllers:
        dronesPositions.append(ctrl.pose)

    distP2R = distance.cdist(map, dronesPositions, 'euclidean')
    minDistPFromR = np.argmin(distP2R, axis=1)

    J = 0.0
    for ind, ctrl in enumerate(controllers):
        Ji = np.sum(distP2R[np.where(minDistPFromR==ind)])
        J += Ji

    costJ.append(J)
    # _____________ End of Cost J ____________________


    # ____________ Update Ji ___________________
    for ctrl in controllers:
        


    # ____________ End of Ji Update  ___________________


def show3DMap():

    global map, controllers

    pcdList = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(map)
    pcdList.append(pcd)

    for ctrl in controllers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2, resolution= 20)
        sphere.translate(ctrl.pose)
        pcdList.append(sphere)


    o3d.visualization.draw_geometries(pcdList) # Visualize the point cloud


if __name__ == "__main__":

    global map, controllers, costJ

    map = loadMap()

    fence = geoFence(appSettings)
    map = fence.clearPointCloud(map)

    controllers = []
    for i in range(SWARM_SIZE):

        ctrl = drone(name = f"UAV{i}",
                     maxView = MAX_VIEW,
                     pose = [POSE_X, POSE_Y + 5*i, POSE_Z],
                     fence = fence,
                     map = map)

        controllers.append(ctrl)

    costJ = []

    for step in range(wayPointsSize):

        print(f"---- Time Step: {step} -----")
        for ctrl in controllers:
            ctrl.updateState(step)
            ctrl.move()

        calculateCostJ()

        for ctrl in controllers:
            # ctrl.updateEstimator1DoF()
            ctrl.updateEstimator()

        print(f"    J = {costJ[-1]}")

    print(f"\n----- Improvement in J={costJ[0] - costJ[-1]}")
    show3DMap()
