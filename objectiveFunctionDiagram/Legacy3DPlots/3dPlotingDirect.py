import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
from functools import reduce
import open3d as o3d
import airsim
import cv2
import math

TIME_STEP = 0
DRONE = "Drone1"

def savePointCloud(data, fileName):

   f = open(fileName, "w")

   c = np.stack((data[0],data[1],data[2]),axis=1)

   for index,(x,y,z) in enumerate(c):
        # color = (255,0,0)
        color = data[3][index]
        color = (color[2]/255, color[1]/255, color[0]/255)
        rgb = "%f %f %f" % color
        # print(f"rgb={rgb} color={color}")

        f.write("%f %f %f %s\n" % (x, y, z, rgb))

   f.close()


def plot3D(image_file):
    pcd = o3d.io.read_point_cloud(image_file, format='xyzrgb')
    o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud


if __name__ == "__main__":

    path = os.path.join(os.getcwd(),"results_1")

    data_dir = os.path.join(path, "swarm_raw_output",f"{DRONE}")
    img_dir = os.path.join(path, "swarm_raw_output",f"{DRONE}", f"position_{TIME_STEP}")

    cloudFile = os.path.join(path,"pointCloud_Drone1.pickle")
    data = pickle.load(open(cloudFile,"rb"))

    outputFile = f"pointCloud_XXX.asc"
    savePointCloud(data[TIME_STEP], outputFile)

    plot3D(outputFile)
