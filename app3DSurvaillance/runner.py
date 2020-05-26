import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import json
import open3d as o3d
from drone import drone
from geoFence import geoFence


# Loading App settings from json file
appSettings = json.load(open('appSettings.json','r'))

SWARM_SIZE = appSettings["UAVS"]["SwarmSize"]
MAX_VIEW = appSettings["UAVS"]["MaxView"]
POSE_X = appSettings["UAVS"]["Positions"]["X"]
POSE_Y = appSettings["UAVS"]["Positions"]["Y"]
POSE_Z =appSettings["UAVS"]["Positions"]["Z"]

def loadMap():

    return np.load(appSettings["MapPath"])


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

    global map, controllers

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

    for step in range(10):

        print(f"---- Time Step: {step} -----")
        for ctrl in controllers:
            ctrl.move()

    show3DMap()
