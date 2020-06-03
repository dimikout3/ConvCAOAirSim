import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utilities.utils as utils

import numpy as np
import pickle
import open3d as o3d
import airsim
import cv2
from itertools import product
from tqdm import tqdm

"""Parses all the position and reconstructs 3D model of the full(!) depth map"""

def savePointCloud(data, fileName):

   f = open(fileName, "w")

   for index in range(len(data[0])):
        # color = (255,0,0)
        x,y,z = data[0][index], data[1][index], data[2][index]

        color = data[3][index]
        color = (color[2]/255, color[1]/255, color[0]/255)
        rgb = "%f %f %f" % color
        # print(f"rgb={rgb} color={color}")

        f.write("%f %f %f %s\n" % (x, y, z, rgb))

   f.close()


def plot3D(image_file):
    pcd = o3d.io.read_point_cloud(image_file, format='xyzrgb')
    o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud


def getPixelsLegacy(img_dir):

    image = os.path.join(img_dir, f"scene_time_0.png")
    scene = cv2.imread(image)

    width, height, _ = scene.shape

    halfWidth = width/2
    halfHeight= height/2

    r = np.random.uniform(0,min(halfHeight,halfWidth),randomPointsSize)
    thetas = np.random.uniform(0,2*np.pi,randomPointsSize)

    pointsH = r*np.sin(thetas)
    pointsW = r*np.cos(thetas)

    centerH = int(halfHeight)
    centerW = int(halfWidth)

    pointsH = centerH + pointsH.astype(int)
    pointsW = centerW + pointsW.astype(int)

    colors = self.imageScene[pointsH, pointsW]

    return x,y,colors


def getPixels(img_dir):

    def points_in_circle(radius):
        for x, y in product(range(int(radius) + 1), repeat=2):
            if x**2 + y**2 <= radius**2:
                 yield from set(((x, y), (x, -y), (-x, y), (-x, -y),))

    image = os.path.join(img_dir, f"scene_time_0.png")
    scene = cv2.imread(image)

    width, height, _ = scene.shape

    halfWidth = width/2
    halfHeight= height/2

    radius = min(halfHeight,halfWidth)
    points = np.array(list(points_in_circle(radius-3)))
    points = points + np.array([int(halfWidth), int(halfHeight)])

    pointsW = points[:,0]
    pointsH = points[:,1]

    colors = scene[pointsH, pointsW]

    return pointsW, pointsH, colors


if __name__ == "__main__":

    simulation_dir = os.path.join(os.getcwd(),"results_1_legacy")

    parent_dir = os.path.join(simulation_dir, "swarm_raw_output")
    detected_dir = os.path.join(simulation_dir, "swarm_detected")

    dronesID = os.listdir(parent_dir)
    dronesID = [drone for drone in dronesID if drone!="GlobalHawk"]
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    for drone in dronesID:

        print(f"=== Woriking on {drone} ===")

        camera_dir = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}",f"state_{drone}.pickle")
        state = pickle.load(open(camera_dir,"rb"))

        for posIndex, position in enumerate(tqdm(wayPointsID)):

            data_dir = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}")
            img_dir = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}", f"{position}")

            pointsW, pointsH, colors = getPixels(img_dir)

            depth_dir = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}", f"{position}", f"depth_time_0.pfm")
            xRelative, yRelative, zRelative, colors = utils.to3D(pointsW, pointsH,
                                              state[posIndex][1], depth_dir,
                                              color = colors)
            x, y, z = utils.to_absolute_coordinates(xRelative, yRelative, zRelative,
                                                    state[posIndex][1])

            data = [x,y,z,colors]

            outputFile = os.path.join(img_dir, f"pointCloud.asc")
            savePointCloud(data, outputFile)

    # plot3D(outputFile)
