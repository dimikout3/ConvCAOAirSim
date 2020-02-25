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

TIME_STEP = 0
DRONE = "Drone1"

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

    image = os.path.join(img_dir, f"scene_time_{TIME_STEP}.png")
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

    image = os.path.join(img_dir, f"scene_time_{TIME_STEP}.png")
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

    path = os.path.join(os.getcwd(),"results_1")

    data_dir = os.path.join(path, "swarm_raw_output",f"{DRONE}")
    img_dir = os.path.join(path, "swarm_raw_output",f"{DRONE}", f"position_{TIME_STEP}")

    camera_dir = os.path.join(path, "swarm_raw_output",f"{DRONE}",f"state_{DRONE}.pickle")
    state = pickle.load(open(camera_dir,"rb"))

    pointsW, pointsH, colors = getPixels(img_dir)

    depth_dir = os.path.join(path, "swarm_raw_output",f"{DRONE}", f"position_{TIME_STEP}", f"depth_time_{TIME_STEP}.pfm")
    xRelative, yRelative, zRelative, colors = utils.to3D(pointsW, pointsH,
                                      state[TIME_STEP][1], depth_dir,
                                      color = colors)
    x, y, z = utils.to_absolute_coordinates(xRelative, yRelative, zRelative,
                                            state[TIME_STEP][1])

    data = [x,y,z,colors]
    outputFile = f"pointCloud_XXX.asc"
    savePointCloud(data, outputFile)

    plot3D(outputFile)
