import airsim
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kickstart():

    d,s = airsim.read_pfm("type_0_time_0.pfm")

    width, height = d.shape
    halfWidth = width/2
    halfHeight= height/2

    camPitch = -0.5
    camYaw = 0.0

    FoV = (np.pi/2)

    randomPointsSize = 300*300
    points = np.random.randint(width,size=(2,randomPointsSize))

    pixelPitch = ((points[0,:]-halfHeight)/halfHeight) * (FoV/2)
    pixelYaw = ((points[1,:]-halfWidth)/halfWidth) * (FoV/2)

    theta = +(np.pi/2) - pixelPitch + camPitch
    # turn
    phi = pixelYaw + camYaw

    r = d[ points[0,:] , points[1,:] ]
    r[r>100] = 100

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    img = getColorPerPixel()
    colors = img[points[0,:] , points[1,:]]

    return x,y,z,colors

def getColorPerPixel():
    img = cv2.imread("type_1_time_0.png")
    imgRes = cv2.resize(img,(300,300))
    return imgRes


def plot3d(x,y,z,size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=size)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.invert_zaxis()
    ax.invert_xaxis()

    plt.show()
    plt.close()

def plot3dColor(x,y,z,size,colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z,c=colors/255.0, s=size)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.invert_zaxis()

    plt.show()
    plt.close()
