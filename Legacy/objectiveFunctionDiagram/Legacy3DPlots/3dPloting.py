import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
from functools import reduce
import open3d as o3d
import airsim
import cv2
import math

LIDAR = False

# color = (0,255,0)
# rgb = "%d %d %d" % color

Width=1200
Height=1200
focal_length=Width/2
B=20
projectionMatrix =  np.array([
        [1, 0, 0, -Width/2],
        [0, 1, 0, -Height/2],
        [0, 0, 0, focal_length],
        [0, 0, -1/B, 0]
    ])

# projectionMatrix = np.array([[-0.501202762, 0.000000000, 0.000000000, 0.000000000],
#                               [0.000000000, -0.501202762, 0.000000000, 0.000000000],
#                               [0.000000000, 0.000000000, 10.00000000, 100.00000000],
#                               [0.000000000, 0.000000000, -10.0000000, 0.000000000]])


TIME_STEP = 0

DRONE_ID_LIST = ["Drone1", "Drone2"]
# DRONE_ID_LIST = ["Drone2"]

def restriction(image, drone_id):
    """Restricting the original image"""

    x_size, y_size = image.shape[0], image.shape[1]

    if drone_id == "Drone1":
        X_CUT = 0.
        Y_Cut = 0.
        xPixels = np.arange(int(X_CUT*x_size),int((1-X_CUT)*x_size))
        yPixels = np.arange(int(Y_Cut*x_size),int((1-Y_Cut)*x_size))
    elif drone_id == "Drone2":
        X_CUT = 0.
        Y_Cut = 0.
        xPixels = np.arange(int(X_CUT*x_size),int((1-X_CUT)*x_size))
        yPixels = np.arange(int(Y_Cut*x_size),int((1-Y_Cut)*x_size))

    return xPixels, yPixels


def savePointCloud(points, ImageRGB, fileName, drone_id):
   f = open(fileName, "w")
   # xPixels, yPixels = restriction(ImageRGB, drone_id)
   for x in range(points.shape[0]):
     for y in range(points.shape[1]):
        pt = points[x,y]
        if (math.isinf(pt[0]) or math.isnan(pt[0])):
          # skip it
          None
        else:

          if drone_id == "Drone1":
              color = (255, 0,0)
          elif drone_id == "Drone2":
              color = (0, 0, 255)

          rgb = "%d %d %d" % color
          # print(f"rgb={rgb} color={color}")

          f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], rgb))
   f.close()


def saveLidar(points, fileName, drone_id):
    f = open(fileName, "w")
    for pt in points:
       if drone_id == "Drone1":
           color = (255, 0,0)
       elif drone_id == "Drone2":
           color = (0, 0, 255)

       rgb = "%d %d %d" % color
       # print(f"rgb={rgb} color={color}")

       f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], rgb))
    f.close()

def plot3D(image_file):
    # https://stackoverflow.com/questions/50965673/python-display-3d-point-cloud
    # https://github.com/intel-isl/Open3D/issues/921
    # Download cloudcompare
    # pcd = o3d.io.read_point_cloud(image_file) # Read the point cloud
    pcd = o3d.io.read_point_cloud(image_file, format='xyzrgb')
    o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud

if __name__ == "__main__":

    for DRONE_ID in DRONE_ID_LIST:
        path = os.path.join(os.getcwd(), "results_1", "swarm_raw_output", DRONE_ID, f"position_{TIME_STEP}")

        if LIDAR:

            cloudFile = os.path.join(path,"lidar_time_0.npy")
            points = np.load(cloudFile)
            outputFile = f"lidar_{DRONE_ID}.asc"
            saveLidar(points, outputFile, DRONE_ID)

        else:

            cloudFile = os.path.join(path,"depth_time_0.pfm")
            rawImage,scale = airsim.read_pfm(cloudFile)
            Image3D = cv2.reprojectImageTo3D(rawImage, projectionMatrix)

            rgbFile = os.path.join(path,"scene_time_0.png")
            ImageRGB = cv2.imread(rgbFile)
            ImageRGB = cv2.cvtColor(ImageRGB, cv2.COLOR_BGR2RGB)

            outputFile = f"cloud_{DRONE_ID}.asc"
            savePointCloud(Image3D, ImageRGB, outputFile, DRONE_ID)

        plot3D(outputFile)
