import airsim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
import open3d as o3d

# Multi point clouds to one:
# http://www.open3d.org/docs/release/tutorial/Advanced/multiway_registration.html

rawImage,scale = airsim.read_pfm("depth_time_0.pfm")

outputFile = "cloud.asc"
color = (0,255,0)
rgb = "%d %d %d" % color

# https://github.com/microsoft/AirSim/issues/778
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

def savePointCloud(image, fileName):
   f = open(fileName, "w")
   for x in range(image.shape[0]):
     for y in range(image.shape[1]):
        pt = image[x,y]
        if (math.isinf(pt[0]) or math.isnan(pt[0])):
          # skip it
          None
        else:
          f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, rgb))
   f.close()

def plot3D(image_file):
    # https://stackoverflow.com/questions/50965673/python-display-3d-point-cloud
    # https://github.com/intel-isl/Open3D/issues/921
    # Download cloudcompare
    # pcd = o3d.io.read_point_cloud(image_file) # Read the point cloud
    pcd = o3d.io.read_point_cloud(image_file, format='xyzrgb')
    o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud


# png = cv2.imdecode(np.frombuffer(rawImage, np.uint8) , cv2.IMREAD_UNCHANGED)
# gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
Image3D = cv2.reprojectImageTo3D(rawImage, projectionMatrix)
# savePointCloud(Image3D, outputFile)
plot3D(outputFile)
print("saved " + outputFile)
# print("view in https://sketchfab.com/3d-models/cloud-4895412f29264724bc44a0027541ee6d")
