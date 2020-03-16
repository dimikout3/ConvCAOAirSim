import setup_path
import airsim

import numpy as np
import cv2
import time
from tqdm import tqdm
from controllerApp import controllerApp
import Discretizator

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from matplotlib import pyplot as plt
import pickle
from utilities.similarity import similarityOut
from threading import Thread
import optparse
import json
import subprocess as sp
from multiprocessing import Process

import GeoFence

import copy

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
baseSet = appSettings['baseApp']

CAM_YAW = baseSet['CamYaw']
CAM_PITCH = baseSet['CamPitch']
CAM_ROOL = baseSet['CamRoll']

geoFenceSet = baseSet['GeoFence']
fenceCenterX = geoFenceSet['centerX']
fenceCenterY = geoFenceSet['centerY']
fenceWidth = geoFenceSet['width']
fenceLength = geoFenceSet['length']
fenceHeight = geoFenceSet['height']

globalHawkSet = baseSet['GlobalHawk']
GLOBAL_HAWK_ACTIVE = globalHawkSet['active']
Xglobal = globalHawkSet['x']
Yglobal = globalHawkSet['y']
Zglobal = globalHawkSet['z']

SAVE_RAW_IMAGES = baseSet['SaveRawImages']
MAX_EXPLORATION_STEPS = 50


def setGlobalHawk(client):
    """Setting the position and heading of the drone that will observer the Enviroment"""
    global options, globalHawk

    OFFSET_GLOBALHAWK = [10,10,0]
    globalHawk = controller(client, "GlobalHawk", OFFSET_GLOBALHAWK, ip=options.ip)
    # The camera orientation of the global view | yaw,pitch,roll | radians
    globalHawk.setCameraOrientation(-np.pi/2, 0., 0.)
    globalHawk.takeOff()
    # globalHawk.setPose(Xglobal, Yglobal, Zglobal, -np.pi/2, 0., 0.)
    #first climb to target altitude | avoid collision
    globalHawk.moveToZ(Zglobal, 3).join()
    globalHawk.moveToPositionYawMode(Xglobal, Yglobal, Zglobal, 3)
    globalHawk.hover()


def plotData(data=None, folder=None, file=None):

    plt.plot(data)

    plt.xlabel("Time")
    plt.ylabel(file)

    plt.tight_layout()

    save_plot = os.path.join(os.getcwd(),f"results_{options.ip}", folder,
                            f"{file}_{positionIdx}.png")
    plt.savefig(save_plot)
    plt.close()


def globalViewScene():

    global controllers, globalHawk

    scene = globalHawk.imageScene
    cameraInfo = globalHawk.cameraInfo
    height, width, colors = scene.shape
    print(f"height={height} width={width} ")

    altitude = abs(cameraInfo.pose.position.z_val)
    hfov = cameraInfo.fov
    vFoV = (height/width)*hfov
    print(f"altitude={altitude} hFoV={hfov} vFoV={vFoV}")

    # what is the farest point global hawk can monitor
    Horizontalhypotenuse = altitude/np.cos( np.radians(hfov/2) )
    maxViewHorizontal = Horizontalhypotenuse*np.sin( np.radians(hfov/2) )
    print(f"Horizontalhypotenuse={Horizontalhypotenuse} maxViewHorizontal={maxViewHorizontal}")

    verticalhypotenuse = altitude/np.cos( np.radians(vFoV/2) )
    maxViewVertical = verticalhypotenuse*np.sin( np.radians(vFoV/2) )
    maxViewVertical = maxViewHorizontal*(height/width)
    print(f"verticalhypotenuse={verticalhypotenuse} maxViewVertical={maxViewVertical}")

    left, right = -maxViewHorizontal + fenceY, maxViewHorizontal + fenceY
    bottom, top = -maxViewVertical + fenceX, maxViewVertical + fenceX
    # https://techtutorialsx.com/2019/04/21/python-opencv-flipping-an-image/
    # flipedScene = cv2.flip(scene, 1)
    # flipedScene = cv2.flip(scene, 0)
    flipedScene = scene
    print(f"left={left} right={right} bot={bottom} top={top}")

    flipedScene = cv2.cvtColor(flipedScene, cv2.COLOR_BGR2RGB)
    plt.imshow(flipedScene,extent=[left, right, bottom, top])

    colors = ['r','b','m', 'c']
    for ind, ctrl in enumerate(controllers):
        # x,y,z,col = ctrl.getPointCloud(x=100,y=100)
        x,y,z,col = ctrl.getPointCloudList()
        plt.scatter(y, x, s=0.2, alpha=0.4, label=ctrl.getName(), c = colors[ind])

    plt.xlim(left, right)
    plt.ylim(bottom, top)

    # plt.gca().invert_yaxis()

    plt.grid(False)
    plt.axis('off')
    # ax1.set_xlabel("Y-Axis (NetWork)")
    # ax1.set_ylabel("X-Axis (NetWork)")
    # plt.margins(0,0)

    # Add line with operations area
    theta = np.linspace(0,2*np.pi,500)
    # r = np.sqrt(fenceR)
    r = fenceR
    y = fenceY + r*np.cos(theta)
    x = fenceX + r*np.sin(theta)
    plt.plot(y,x,'k--')

    plt.legend(markerscale=20)
    # plt.tight_layout()

    globalView_file = os.path.join(os.getcwd(),f"results_{options.ip}", "globalView",
                            f"globalViewScene_{positionIdx}.png")

    plt.savefig(globalView_file, bbox_inches = 'tight', dpi=1500)
    plt.close()


def generatingResultsFolders():

    global options

    result_folder = os.path.join(os.getcwd(), f"results_{options.ip}")
    try:
        os.makedirs(result_folder)
    except OSError:
        if not os.path.isdir(result_folder):
            raise

    detected_objects_folder = os.path.join(result_folder, "detected_objects")
    try:
        os.makedirs(detected_objects_folder)
    except OSError:
        if not os.path.isdir(detected_objects_folder):
            raise

    similarity_objects_folder = os.path.join(result_folder, "similarity_objects")
    try:
        os.makedirs(similarity_objects_folder)
    except OSError:
        if not os.path.isdir(similarity_objects_folder):
            raise

    information_folder = os.path.join(result_folder, "information")
    try:
        os.makedirs(information_folder)
    except OSError:
        if not os.path.isdir(information_folder):
            raise

    survaillance_folder = os.path.join(result_folder, "survaillance")
    try:
        os.makedirs(survaillance_folder)
    except OSError:
        if not os.path.isdir(survaillance_folder):
            raise

    costJ_folder = os.path.join(result_folder, "costJ")
    try:
        os.makedirs(costJ_folder)
    except OSError:
        if not os.path.isdir(costJ_folder):
            raise

    report_folder = os.path.join(result_folder, "report")
    try:
        os.makedirs(report_folder)
    except OSError:
        if not os.path.isdir(report_folder):
            raise

    globalView_folder = os.path.join(result_folder, "globalView")
    try:
        os.makedirs(globalView_folder)
    except OSError:
        if not os.path.isdir(globalView_folder):
            raise

    globalViewDetection_folder = os.path.join(result_folder, "globalViewDetections")
    try:
        os.makedirs(globalViewDetection_folder)
    except OSError:
        if not os.path.isdir(globalViewDetection_folder):
            raise


def getImages():

    global controllers

    threadList = []
    for ctrl in controllers:
        argsDict = dict(save_raw = True)
        thread = Thread(target = ctrl.getImages, kwargs=argsDict)
        thread.start()
        threadList.append(thread)
    for thread in threadList:
        thread.join()


def getPointClouds():

    global controllers

    threadList = []
    for ctrl in controllers:
        argsDict = dict(x = 100, y = 100)
        thread = Thread(target = ctrl.getPointCloud, kwargs=argsDict)
        thread.start()
        threadList.append(thread)
    for thread in threadList:
        thread.join()


def descretization(discretizator):

    global controllers

    threadList = []
    for ctrl in controllers:
        # argsDict = dict(discretizator = copy.deepcopy(discretizator))
        thread = Thread(target = ctrl.pointCloud2Descrete)
        thread.start()
        threadList.append(thread)
    for thread in threadList:
        thread.join()


def fillTemplate():

    global options

    # settingsTemplate = os.path.join(settingsDir,"settingsTemplate.json")
    json_data = json.load(open('airsimSettings.json','r'))

    json_data["LocalHostIp"] = f"127.0.0.{options.ip}"

    settingsOutput = os.path.join(settingsDir,"settings.json")
    json.dump(json_data,open(settingsOutput,"w"),indent=2)


def get_options():

    optParser = optparse.OptionParser()
    optParser.add_option("--ip", dest="ip", default=1,type="int", help="the ip of the simulations launched")
    optParser.add_option("--waypoints", default=500, dest="waypoints",type="int", help="the number of waypoints provided")
    optParser.add_option("--maxYaw", default=10, dest="maxYaw",type="float", help="max Yaw change")
    optParser.add_option("--maxTravelTime", default=2.5, dest="maxTravelTime",type="float", help="max distance to be travelled in one step")
    optParser.add_option("--estimatorWindow", default=30, dest="estimatorWindow",type="int", help="CAO estimator window")
    options, args = optParser.parse_args()

    return options


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

    global options, controllers, globalHawk

    options = get_options()

    generatingResultsFolders()

    fillTemplate()

    launchAirSim()

    wayPointsSize = options.waypoints

    OFFSETS = {"UAV1":[0,0,0],
               "UAV2":[0,-5,0]
               # "UAV3":[5,0,0],
               # "UAV4":[5,5,0]
              }

    dronesID = list(OFFSETS.keys())

    ip_id = f"127.0.0.{options.ip}"

    if GLOBAL_HAWK_ACTIVE:
        client = airsim.MultirotorClient(ip = ip_id)
        client.confirmConnection()
        setGlobalHawk(client)

    controllers = []
    for drone in dronesID:
        client = airsim.MultirotorClient(ip = ip_id)
        client.confirmConnection()
        controllers.append(controllerApp(client, drone, OFFSETS[drone],
                                      ip=options.ip,
                                      wayPointsSize=wayPointsSize,
                                      estimatorWindow=options.estimatorWindow))

    # Setting Camera Orientation
    for ctrl in controllers: ctrl.setCameraOrientation(CAM_YAW, CAM_PITCH, CAM_ROOL)

    print("Taking off all drones")
    tasks = []
    for ctrl in controllers:
        t = ctrl.takeOff()
        tasks.append(t)
    for t in tasks: t.join()

    print("\nSetting Geo Fence for all drones")
    fence = GeoFence.GeoFence(centerX = fenceCenterX, centerY = fenceCenterY,
                              width = fenceWidth, length = fenceLength,
                              height = fenceHeight)
    for ctrl in controllers:
        # no need for task list (just setting values here)
        ctrl.setGeoFence(geofence = fence)

    discretizator = Discretizator.Discretizator(discrete=baseSet['Discrete'], geofence=fence)
    discretizator.report()

    for ctrl in controllers:
        ctrl.updateDescretizator(discretizator)

    discr = baseSet['Discrete']
    Explored = np.zeros((discr['x'], discr['y'], discr['z']), dtype=bool)
    Obstacles = np.ones((discr['x'], discr['y'], discr['z']), dtype=bool)

    startTime = time.time()

    global costJ
    costJ = []

    for positionIdx in range(0,options.estimatorWindow):

        ptime = time.time()

        for ctrl in controllers:
            ctrl.updateState(positionIdx,0)

        getImages()
        getPointClouds()
        descretization(discretizator)
        data = controllers[0].descretePointCloud[-1]
        discretizator.show(data)

        # tasks = []
        # for ctrl in controllers:
        #     t = ctrl.moveToPositionYawModeAsync( float(d[ctrl.getName()]["x"][positionIdx]),
        #                                          float(d[ctrl.getName()]["y"][positionIdx]),
        #                                          float(d[ctrl.getName()]["z"][positionIdx]),
        #                                          speed=3, yawmode = d[ctrl.getName()]["yaw"][positionIdx])
        #     tasks.append(t)
        #
        # for task in tasks[::-1]:
        #     task.join()

        for ctrl in controllers:

            positions = ctrl.getPositions()
            orientation = ctrl.getOrientation()

            x = positions.x_val
            y = positions.y_val
            z = positions.z_val
            _,_,yaw = airsim.to_eularian_angles(orientation)

            # print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{np.degrees(yaw):.2f}) with Ji:{ctrl.getJi():.2f}")
            print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{np.degrees(yaw):.2f})")

        # threadList = []
        # for ctrl in controllers:
        #     thread = Thread(target = ctrl.updateEstimator)
        #     thread.start()
        #     threadList.append(thread)
        # for thread in threadList:
        #     thread.join()

        # for ctrl in controllers: ctrl.plotEstimator1DoF()

        print(f"----- elapsed time: {time.time() - ptime:.3f} ------")
        print("---------------------------------\n")

        # plotData(data=informationJ, folder="information", file="information")
        # plotData(data=survaillanceJ, folder="survaillance", file="survaillance")

        if GLOBAL_HAWK_ACTIVE:
            globalViewScene()

    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "costJ",
                            f"costJ.pickle")
    pickle.dump(costJ,open(file_out,"wb"))

    print("\n[RESETING] to original state ....")
    for ctrl in controllers: ctrl.quit()
    client.reset()

    killAirSim()

    print(f"\n --- elapsed time:{startTime - time.time():.2f} [sec] ---")
