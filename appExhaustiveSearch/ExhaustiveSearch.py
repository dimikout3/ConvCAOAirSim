import setup_path
import airsim

import numpy as np
import os
import cv2
import time
from tqdm import tqdm
from controller import controller
import yoloDetector
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from utilities.similarity import similarityOut
from threading import Thread
import optparse
import json
import subprocess as sp

# TODO: rename to ruuner.py and kill airsim env when finished

settingsDir = r"C:/Users/" + os.getlogin() + "/Documents/AirSim"
envDir = r"C:/Users/" + os.getlogin() + "/Documents/AirSim/CityEnviron"

CAM_YAW = -0.5
CAM_PITCH = 0.
CAM_ROOL = 0.

NORM = {'information':10.0, 'similarity':10.0}
WEIGHT = {'information':1.0, 'similarity':-1.0}
KW = 1

def plotDetections(detectionsDict, excludedDict, posInd):

    global options

    dronesID = list(detectionsDict.keys())

    fig, ax = plt.subplots(2,2,figsize=(10,10))

    xUnraveled, yUnraveled = np.unravel_index(np.arange(4),(2,2))

    for j,drone in enumerate(dronesID):

        xPositive, yPositive = [], []
        xNegative, yNegative = [], []

        for i,detection in enumerate(detectionsDict[drone][0]):

            x,y = detection[0], detection[1]

            if i in excludedDict[drone]:
                xNegative.append(x)
                yNegative.append(y)
            else:
                xPositive.append(x)
                yPositive.append(y)

        ax[xUnraveled[j], yUnraveled[j]].scatter(yPositive, xPositive, marker='o', label=f"{drone} (+)")
        ax[xUnraveled[j], yUnraveled[j]].scatter(yNegative, xNegative, marker='v', label=f"{drone} (-)")

        ax[xUnraveled[j], yUnraveled[j]].legend()
        ax[xUnraveled[j], yUnraveled[j]].set_xlim(-130,130)
        ax[xUnraveled[j], yUnraveled[j]].set_ylim(-130,130)

        ax[xUnraveled[j], yUnraveled[j]].set_xlabel("Y-Axis (NetWork)")
        ax[xUnraveled[j], yUnraveled[j]].set_ylabel("X-Axis (NetWork)")

    detected_objects_folder = os.path.join(os.getcwd(),f"results_{options.ip}",
                                           "detected_objects", f"detections_{posInd}.png")

    plt.tight_layout()
    plt.savefig(detected_objects_folder)
    plt.close()


def detectionsScore(ego="None", excludedDict={}):

    global controllers

    informationScore = 0

    for ctrl in controllers:

        if isinstance(ego,str):
            index = -1
        else:
            if ctrl.getName() == ego.getName():
                index = -2
            else:
                index = -1

        score = ctrl.scoreExcludingDetections(index=index, excludedList=excludedDict[ctrl.getName()], minusDuplicates=False)
        informationScore += score

    return informationScore


def monitor(posInd, timeInterval = 1, totalTime = 1):

    global options, controllers
    global similarityList, informationScoreList, costJ

    for timeStep in range(0,totalTime,timeInterval):

        detectionsDict = {}

        for i,ctrl in enumerate(controllers):
            ctrl.updateState(posInd, timeStep)
            ctrl.getImages(save_raw=False)

        threadList = []
        for i,ctrl in enumerate(controllers):
            thread = Thread(target = ctrl.detectObjects)
            thread.start()
            threadList.append(thread)
        for thread in threadList:
            thread.join()

        for i,ctrl in enumerate(controllers):
            detectionsCoordinates, detectionsInfo = ctrl.getDetections()
            detectionsData = [detectionsCoordinates, detectionsInfo]

            detectionsDict[ctrl.getName()] = detectionsData

        excludedDict = similarityOut(detectionsDict, similarityKPI="DistExhaustive", ip=options.ip)

        plotDetections(detectionsDict, excludedDict, posInd)

        informationScore = detectionsScore(excludedDict = excludedDict)

        J = informationScore

        time.sleep(timeInterval)

    return J

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


def fillTemplate():

    global options

    settingsTemplate = os.path.join(settingsDir,"settingsTemplate.json")
    json_data = json.load(open(settingsTemplate,'r'))

    json_data["LocalHostIp"] = f"127.0.0.{options.ip}"

    settingsOutput = os.path.join(settingsDir,"settings.json")
    json.dump(json_data,open(settingsOutput,"w"),indent=2)


def get_options():

    optParser = optparse.OptionParser()
    optParser.add_option("--ip", dest="ip", default=1,type="int", help="the ip of the simulations launched")
    optParser.add_option("--waypoints", default=300, dest="waypoints",type="int", help="the number of waypoints provided")
    optParser.add_option("--canditatesS", default=60, dest="canditatesSize",type="int", help="number of canditates to visit before moving")
    options, args = optParser.parse_args()

    return options


def launchAirSim():

    call = f"{envDir}\\CityEnviron -windowed -ResX=640 -ResY=480"
    sp.Popen(call, shell=True)
    time.sleep(10)


def killAirSim():
    """ Killing all the exe that have 'CityEnviron' string """
    os.system('TASKKILL /F /IM CityEnviron*')

if __name__ == "__main__":

    global options, controllers

    options = get_options()

    generatingResultsFolders()

    fillTemplate()

    launchAirSim()

    wayPointsSize = options.waypoints

    OFFSETS = {"Drone1":[0,0,0],
               "Drone2":[0,-5,0],
               "Drone3":[5,0,0],
               "Drone4":[5,5,0]
              }


    dronesID = list(OFFSETS.keys())

    ip_id = f"127.0.0.{options.ip}"
    client = airsim.MultirotorClient(ip = ip_id)
    client.confirmConnection()

    controllers = []
    for drone in dronesID:
        controllers.append(controller(client, drone, OFFSETS[drone],
                                      ip=options.ip, timeWindow=wayPointsSize))

    # Setting Camera Orientation
    for ctrl in controllers: ctrl.setCameraOrientation(CAM_YAW, CAM_PITCH, CAM_ROOL)

    print("Taking off all drones")
    tasks = []
    for ctrl in controllers:
        t = ctrl.takeOff()
        tasks.append(t)
    for t in tasks: t.join()

    print("\nLifting all drones to specified Z altitude")
    tasks = []
    intialAlt = -14
    stepAlt = -0.5
    for i,ctrl in enumerate(controllers):
        t = ctrl.moveToZ(intialAlt + stepAlt*i,2)
        tasks.append(t)
    for t in tasks: t.join()

    print("\nSetting Geo Fence for all drones")
    for ctrl in controllers:
        # no need for task list (just setting values here)
        ctrl.setGeoFence(x = 25, y = -25, z = -14, r=45)

    print("\nTaking initial photos")
    for ctrl in controllers:
        # no need for task list (just setting values here)
        ctrl.getImages()


    startTime = time.time()

    global costJ
    costJ = []

    for positionIdx in range(0,wayPointsSize):

        print(f"[INFO] curently on position:{positionIdx}")

        ptime = time.time()

        canditates = []
        for i in range(options.canditatesSize):
            """Create a set of random canditate positions to visit and choose one as ideal next"""
            # canditates = [set1, set2, set3 ... setCANDITATESIZE]
            # set1 = {drone1:[x,y,z,yaw], drone2:[x,y,z,yaw] ... droneN:[x,y,z,yaw]}
            candiatateDict = {}
            for ctrl in controllers:
                # TODO: imporove the canditates (use exactly 1) - using more otherwise possible crush
                canditatesData = ctrl.getCanditates(randomPointsSize=10,
                                                    maxTravelTime=2.5,
                                                    minDist=7.5,
                                                    maxYaw=10)

                _, xCanditateList, yCanditateList, zCanditateList, yawCanditateList = canditatesData
                pick = np.random.randint(0,len(xCanditateList))
                candiatateDict[ctrl] = [xCanditateList[pick], yCanditateList[pick],
                                        zCanditateList[pick], yawCanditateList[pick]]

            canditates.append(candiatateDict)

        JCanditate = []
        for c,cand in enumerate(canditates):

            print(f"[INFO] moving to canditates set:{c}")

            tasks = []
            for drone,canditatePositions in cand.items():

                xCanditate, yCanditate, zCanditate, yawCanditate = canditatePositions
                t = drone.moveToPositionYawModeAsync(xCanditate, yCanditate,
                                                     zCanditate,
                                                     yawmode = yawCanditate)
                tasks.append(t)
            for task in tasks[::-1]:
                # wait all drones to reach their canditates positions
                task.join()

            JCanditate.append(monitor(positionIdx))

        best = np.argmax(JCanditate)
        print(f"[INFO] ideal canditate is set:{best}")
        costJ.append(JCanditate[best])
        tasks = []
        for drone,canditatePositions in canditates[best].items():
            xCanditate, yCanditate, zCanditate, yawCanditate = canditatePositions
            t = drone.moveToPositionYawModeAsync(xCanditate, yCanditate,
                                                 zCanditate,
                                                 yawmode = yawCanditate)
            tasks.append(t)
        for task in tasks[::-1]:
            # wait all drones to reach their canditates positions
            task.join()

        for ctrl in controllers:

            positions = ctrl.getPositions()
            orientation = ctrl.getOrientation()

            x = positions.x_val
            y = positions.y_val
            z = positions.z_val
            _,_,yaw = airsim.to_eularian_angles(orientation)

            print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{np.degrees(yaw):.2f})")


        # for ctrl in controllers: ctrl.plotEstimator1DoF()

        print(f"----- elapsed time: {time.time() - ptime:.3f} ------")
        print("---------------------------------\n")

        plotTime = time.time()

        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(costJ, label="Cost J")

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")

        ax1.legend()

        for ctrl in controllers:
            ax2.plot(ctrl.getJiList(), label=ctrl.getName())

        # ax2.set_ylim(-180,180)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Ji")
        ax2.legend()

        plt.tight_layout()

        report_plot = os.path.join(os.getcwd(),f"results_{options.ip}", "report",
                                f"report_{positionIdx}.png")
        plt.savefig(report_plot)
        # plt.show(block=False)
        # plt.pause(5)
        plt.close()


        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,10))

        for ctrl in controllers:
            x,y,z,col = ctrl.getPointCloud(x=100,y=100)
            ax2.scatter(y, x,c=col/255.0, s=0.05)
            ax1.scatter(y, x, s=0.05, label=ctrl.getName())

        xlim = [-130,130]
        ylim = [-130,130]
        ax1.set_xlim(xlim[0],xlim[1])
        ax1.set_ylim(ylim[0],ylim[1])
        ax2.set_xlim(xlim[0],xlim[1])
        ax2.set_ylim(ylim[0],ylim[1])

        ax1.legend(markerscale=20)

        ax1.set_xlabel("Y-Axis (NetWork)")
        ax1.set_ylabel("X-Axis (NetWork)")
        ax2.set_xlabel("Y-Axis (NetWork)")
        ax2.set_ylabel("X-Axis (NetWork)")

        plt.tight_layout()

        globalView_file = os.path.join(os.getcwd(),f"results_{options.ip}", "globalView",
                                f"globalView_{positionIdx}.png")
        plt.savefig(globalView_file)
        plt.close()

        # print(f"----- ploting global view & costJ adds time: {time.time() - plotTime:.3f} ------")
        # print("---------------------------------\n")

    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "costJ",
                            f"costJ.pickle")
    pickle.dump(costJ,open(file_out,"wb"))

    print("\n[RESETING] to original state ....")
    for ctrl in controllers: ctrl.quit()
    client.reset()

    print(f"\n[KILLING|AIRSIM] closing CityEnviron.exe")
    killAirSim()

    print(f"\n --- elapsed time:{startTime - time.time():.2f} [sec] ---")
