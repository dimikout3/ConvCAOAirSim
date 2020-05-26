import airsim

import numpy as np
import cv2
import time
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from matplotlib import pyplot as plt
import pickle
from threading import Thread
import optparse
import json
import subprocess as sp

import open3d as o3d

if os.name == 'nt':
    settingsDir = r"C:/Users/" + os.getlogin() + "/Documents/AirSim"
    envDir = r"C:/Users/" + os.getlogin() + "/Documents/AirSim/CityEnviron"
    call = f"{envDir}\\CityEnviron -windowed -ResX=640 -ResY=480"
else:
    settingsDir = r"/home/" + os.getlogin() + "/Documents/AirSim"
    envDir = r"/home/" + os.getlogin() + "/Downloads/Neighborhood/AirSimNH.sh -ResX=640 -ResY=480 -windowed"
    call = f"{envDir}"

# Loading App settings from json file
appSettings = json.load(open('appSettings.json','r'))


def fillTemplate():

    # settingsTemplate = os.path.join(settingsDir,"settingsTemplate.json")
    json_data = json.load(open('airsimSettings.json','r'))

    settingsOutput = os.path.join(settingsDir,"settings.json")
    json.dump(json_data,open(settingsOutput,"w"),indent=2)


def launchAirSim():

    sp.Popen(call, shell=True)
    time.sleep(10)


def killAirSim():
    """ Killing all the exe that have 'CityEnviron' string """

    if os.name == 'nt':
        print(f"\n[KILLING|AIRSIM] closing CityEnviron.exe")
        os.system('TASKKILL /F /IM CityEnviron*')
    else:
        print(f"\n[KILLING|AIRSIM] closing AirSimNH")
        output = os.system("pkill AirSim")


def getLidar(client):

    for test in range(10):

        lidarData = client.getLidarData()

        points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
        if points.size != 0:
            break
        time.sleep(1)

    points = np.reshape(points, (int(points.shape[0]/3), 3))

    return points


def show3DMap(lidarPoints):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidarPoints)
    o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud


if __name__ == "__main__":

    fillTemplate()

    launchAirSim()

    client = airsim.CarClient()
    client.confirmConnection()

    lidraPoints = np.array([[0,0,0]])

    for step in range(appSettings["TotalSteps"]):

        newPoints = getLidar(client)
        lidraPoints = np.concatenate((lidraPoints, newPoints))

        print(f"{step} [TimeStep] | new points :{newPoints.shape}| total points {lidraPoints.shape} ")

        time.sleep(appSettings["LidarPerSec"])

    show3DMap(lidraPoints)

    np.save("lidarPoints.npy",lidraPoints)

    killAirSim()
