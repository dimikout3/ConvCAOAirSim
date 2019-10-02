import airsim
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')

def kickstart(random_points=[300,300]):

    d,s = airsim.read_pfm("depth_time_4.pfm")

    height, width = d.shape
    print(f"Image size: width:{width} -- height:{height}")
    halfWidth = width/2
    halfHeight= height/2

    camPitch = -0.5
    camYaw = 0.0

    FoV = (np.pi/2)

    randomPointsSize = random_points[0]*random_points[1]
    # points = np.random.randint(width,size=(2,randomPointsSize))
    pointsH = np.random.randint(height,size=(randomPointsSize))
    pointsW = np.random.randint(width,size=(randomPointsSize))

    pixelPitch = ((pointsH-halfHeight)/halfHeight) * (FoV/2)
    pixelYaw = ((pointsW-halfWidth)/halfWidth) * (FoV/2)

    theta = (np.pi/2) - pixelPitch + camPitch
    # turn
    phi = pixelYaw + camYaw

    r = d[ pointsH , pointsW ]
    idx = np.where(r<100)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    img = getColorPerPixel()
    colors = img[ pointsH , pointsW ]

    return x[idx],y[idx],z[idx],colors[idx]

def getColorPerPixel():
    img = cv2.imread("scene_time_4.png")
    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return imgRGB


def plot3d(x,y,z,size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=size)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.invert_zaxis()
    # ax.invert_xaxis()

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
    ax.invert_yaxis()

    # ax.view_init(elev=0,azim=180)

    # ax.set_zlim()
    ax.set_xlim(100,0)
    # ax.set_ylim(-55,55)
    plt.show()
    # plt.savefig("test.png")
    plt.close()
