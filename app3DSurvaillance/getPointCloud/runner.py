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


if __name__ == "__main__":

    launchAirSim()

    for i in range(10):

    killAirSim()
