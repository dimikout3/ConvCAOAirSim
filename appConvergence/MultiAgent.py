import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import setup_path
import airsim

import numpy as np
import cv2
import time
from tqdm import tqdm
from controller import controller
from evaluate import evaluate
import yoloDetector
from matplotlib import pyplot as plt
import pickle
from utilities.similarity import similarityOut
from threading import Thread
import optparse
import json
import subprocess as sp
from multiprocessing import Process

if os.name == 'nt':
    settingsDir = r"C:/Users/" + os.getlogin() + "/Documents/AirSim"
    envDir = r"C:/Users/" + os.getlogin() + "/Documents/AirSim/CityEnviron"
    call = f"{envDir}\\CityEnviron -windowed -ResX=640 -ResY=480"
else:
    settingsDir = r"/home/" + os.getlogin() + "/Documents/AirSim"
    envDir = r"/home/" + os.getlogin() + "/Downloads/Neighborhood/AirSimNH.sh"
    call = f"{envDir}"

# Loading App settings from json file
appSettings = json.load(open('appSettings.json','r'))
baseSet = appSettings['baseApp']

CAM_YAW = baseSet['CamYaw']
CAM_PITCH = baseSet['CamPitch']
CAM_ROOL = baseSet['CamRoll']

geoFenceSet = baseSet['GeoFence']
fenceR = geoFenceSet['R']
fenceX = geoFenceSet['x']
fenceY = geoFenceSet['y']
fenceZ = geoFenceSet['z']

#positions of GlobalHawk
# (25, 33, -20)
globalHawkSet = baseSet['GlobalHawk']
GLOBAL_HAWK_ACTIVE = globalHawkSet['active']
Xglobal = globalHawkSet['x']
Yglobal = globalHawkSet['y']
Zglobal = globalHawkSet['z']

SAVE_RAW_IMAGES = baseSet['SaveRawImages']
MAX_EXPLORATION_STEPS = 50


def collisionCorrection(timeOutCollision=300):

    global controllers

    wayPointDict = {}
    for ctrl in controllers:
        wayPointDict[ctrl.getName()] = 0
    time.sleep(timeOutCollision)

    running = True

    while running:

        time.sleep(timeOutCollision)

        for ctrl in controllers:

            if ctrl.getWayPoint() == wayPointDict[ctrl.getName()]:

                print(f"{ctrl.getName()} is at the same spot after {timeOutCollision}[sec]. Check collision")

                try:
                    state = ctrl.getState()
                    print(f"{ctrl.getName()} got state")
                except:
                    print(f"{ctrl.getName()} could not get state and failed")

                try:
                    if state.collision.has_collided:

                        print(f"{ctrl.getName()} has collided ...")

                        stateList = ctrl.getStateList()
                        x = stateList[-2].kinecmatics_estimated.position.x_val
                        y = stateList[-2].kinecmatics_estimated.position.y_val
                        z = stateList[-2].kinecmatics_estimated.position.z_val
                        _,_,yaw = airsim.to_eularian_angles(stateList[-3].kinematics_estimated.orientation)

                        task = ctrl.moveToPositionYawModeAsync(x,y,z,1,yaw)
                        task.join()

                        print(f"{ctrl.getName()}moved successfully to previous position")

                except:
                    print(f"{ctrl.getName()} failed at moving drone. Debug ...")
                    # import pdb; pdb.set_trace()

            else:
                print(f"{ctrl.getName()} has changed spot. Update wayPointDict")
                wayPointDict[ctrl.getName()] = ctrl.getWayPoint()

            if ctrl.getWayPoint() >= (ctrl.wayPointSize-2):
                running = False


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


def reportPlot():

    global controllers, informationJ

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(informationJ, label="Cost J")

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


def globalViewSceneOriginal():

    global controllers, globalHawk

    def scale(array, OldMax, OldMin, NewMax=127, NewMin=0):
        # https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

        return [(((OldValue - OldMin) * NewRange) / OldRange) + NewMin for OldValue in array]

    scene = globalHawk.imageScene
    cameraInfo = globalHawk.cameraInfo
    width, height, colors = scene.shape

    altitude = abs(cameraInfo.pose.position.z_val)
    fov = cameraInfo.fov

    # what is the farest point global hawk can monitor
    hypotenuse = altitude/np.cos( np.radians(fov/2) )
    maxView = hypotenuse*np.sin( np.radians(fov/2) )

    OldMaxX = fenceX + maxView
    OldMinX = fenceX - maxView

    OldMaxY = fenceY + maxView
    OldMinY = fenceY - maxView

    # https://techtutorialsx.com/2019/04/21/python-opencv-flipping-an-image/
    # flipedScene = cv2.flip(scene, 1)
    flipedScene = cv2.flip(scene, 0)

    flipedScene = cv2.cvtColor(flipedScene, cv2.COLOR_BGR2RGB)
    plt.imshow(flipedScene)

    for ctrl in controllers:
        # x,y,z,col = ctrl.getPointCloud(x=100,y=100)
        x,y,z,col = ctrl.getPointCloudList()

        x = scale(x, OldMax=OldMaxX, OldMin=OldMinX, NewMax=(width-1))
        y = scale(y, OldMax=OldMaxY, OldMin=OldMinY, NewMax=(width-1))

        plt.scatter(y, x, s=0.2, alpha=0.4, label=ctrl.getName())

    plt.xlim(0,width-1)
    plt.ylim(0,width-1)

    # plt.gca().invert_yaxis()

    # plt.grid(False)
    # plt.axis('off')
    # ax1.set_xlabel("Y-Axis (NetWork)")
    # ax1.set_ylabel("X-Axis (NetWork)")

    plt.legend(markerscale=20)
    plt.tight_layout()

    globalView_file = os.path.join(os.getcwd(),f"results_{options.ip}", "globalView",
                            f"globalViewScene_{positionIdx}.png")
    plt.savefig(globalView_file)
    plt.close()


def globalViewDetections(excludedDict=[]):

    global controllers, globalHawk

    def scale(array, OldMax, OldMin, NewMax=127, NewMin=0):
        # https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

        return [(((OldValue - OldMin) * NewRange) / OldRange) + NewMin for OldValue in array]

    scene = globalHawk.imageScene
    cameraInfo = globalHawk.cameraInfo
    width, height, colors = scene.shape

    altitude = abs(cameraInfo.pose.position.z_val)
    fov = cameraInfo.fov

    # what is the farest point global hawk can monitor
    hypotenuse = altitude/np.cos( np.radians(fov/2) )
    maxView = hypotenuse*np.sin( np.radians(fov/2) )

    OldMaxX = fenceX + maxView
    OldMinX = fenceX - maxView

    OldMaxY = fenceY + maxView
    OldMinY = fenceY - maxView

    # https://techtutorialsx.com/2019/04/21/python-opencv-flipping-an-image/
    # flipedScene = cv2.flip(scene, 1)
    flipedScene = cv2.flip(scene, 0)

    flipedScene = cv2.cvtColor(flipedScene, cv2.COLOR_BGR2RGB)
    plt.imshow(flipedScene)

    for ctrl in controllers:
        # x,y,z,col = ctrl.getPointCloudList()
        x = []
        y = []

        detectionsCoordinates = ctrl.getDetectionsCoordinates()
        for i,detection in enumerate(detectionsCoordinates):

            if i not in excludedDict[ctrl.getName()]:
                x.append(detection[0])
                y.append(detection[1])

        x = scale(x, OldMax=OldMaxX, OldMin=OldMinX, NewMax=(width-1))
        y = scale(y, OldMax=OldMaxY, OldMin=OldMinY, NewMax=(width-1))

        plt.scatter(y, x, s=22.5, alpha=0.4, label=ctrl.getName())

    plt.xlim(0,width-1)
    plt.ylim(0,width-1)

    plt.grid(False)
    plt.axis('off')

    # plt.legend(markerscale=20)
    plt.legend()
    plt.tight_layout()

    globalView_file = os.path.join(os.getcwd(),f"results_{options.ip}", "globalViewDetections",
                            f"globalViewDetection_{positionIdx}.png")
    plt.savefig(globalView_file)
    plt.close()


def globalView():

    global controllers

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,10))

    for ctrl in controllers:
        # x,y,z,col = ctrl.getPointCloud(x=100,y=100)
        x,y,z,col = ctrl.getPointCloudList()
        ax2.scatter(y, x,c=col/255.0, s=0.05)
        ax1.scatter(y, x, s=0.05, label=ctrl.getName())

    xlim = [fenceY-(fenceR+70),fenceY+(fenceR+70)]
    ylim = [fenceX-(fenceR+70),fenceX+(fenceR+70)]
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
                            f"globalViewOriginal_{positionIdx}.png")
    plt.savefig(globalView_file)
    plt.close()


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


def updateInformationDeltaJi(ego=None):

    global controllers, informationJ

    if ego.posIdx == 0:
        update = informationJ[-1]
    elif ego.posIdx > 0:

        # J_isolation = evaluator.randomPointCloudCost(ego=ego)
        J_isolation = 0.
        for ctrl in controllers:
            if ctrl.getName() == ego.getName():
                J_isolation += ctrl.getInformationJ(index=-2)
            else:
                J_isolation += ctrl.getInformationJ(index=-1)

        delta = informationJ[-1] - J_isolation
        update = ego.getinformationJi() + delta

    ego.appendInforamtionJi(update)


def updateDelta(ego="None", detectionsDict={}, excludedDict={}, updateRule="_"):

    global controllers

    if updateRule == "delta":
        """ Updating Using the delta differences, proposed in distributed CAO """

        if ego.posIdx == 0:
            update = costJ[-1]
        elif ego.posIdx > 0:
            # J_isolation = evaluator.hullDistanceCost(ego=ego)
            J_isolation = evaluator.randomPointCloudCost(ego=ego)
            delta = costJ[-1] - J_isolation
            update = ego.getJi() + delta

        ego.appendJi(update)

    elif updateRule == "gradient":
        """Update using gradient descent like operations (similar to BCD)"""

        infromationJ = ego.getInformationJ(index=-1)

        if ego.posIdx == 0:
            update = infromationJ

        elif ego.posIdx > 0:
            infromationJprev = ego.getInformationJ(index=-2)
            update = infromationJ - infromationJprev

        ego.appendJi(update)

    elif updateRule == "directInformationJ":
        """ Update using direct values from informationJ """

        ego.resetJi(resetStyle="directInformationJ")

    elif updateRule == "deltaInformationJi":
        """ Update using BCD jI values from informationJi """

        ego.resetJi(resetStyle="deltaInformationJi")


def checkPhase(posInd):
    """Checking if we are in exploration or exploitation phase"""

    if posInd>=MAX_EXPLORATION_STEPS:
        return False, True

    # TODO: check if the N last steps of survaillanceJ are within a range (deviation)
    # their gradient is stable (h klish tous)
    # https://en.wikipedia.org/wiki/Root-mean-square_deviation

    return True, False


def monitor(droneList, posInd, timeInterval = 1, totalTime = 1):

    global options, controllers, evaluator, globalHawk

    print(f"[MONITORING] position {posInd}")

    # detector = yoloDetector.yoloDetector()

    global similarityList, informationJ,survaillanceJ, costJ

    for timeStep in range(0,totalTime,timeInterval):

        detectionsDict = {}

        if GLOBAL_HAWK_ACTIVE:
            globalHawk.updateState(posInd, timeStep)
            globalHawk.getImages(save_raw=True)

        for i,ctrl in enumerate(controllers):
            ctrl.updateState(posInd, timeStep)
            ctrl.getImages(save_raw=SAVE_RAW_IMAGES)
            ctrl.getPointCloud(x=50,y=50)

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
        if GLOBAL_HAWK_ACTIVE:
            globalViewDetections(excludedDict = excludedDict)

        evaluator.update(controllers = controllers,
                         excludedDict = excludedDict,
                         detectionsDict = detectionsDict)

        informationScore = evaluator.detectionsScore()
        # costNoDetection = evaluator.noDetectionsCost()
        randomCloudDistCost = evaluator.randomPointCloudCost()

        informationJ.append(informationScore)
        survaillanceJ.append(randomCloudDistCost)

        explorationActive, exploitationActive = checkPhase(posInd)

        for ctrl in controllers:
            ctrl.setExcludedList(excludedDict[ctrl.getName()])
            ctrl.storeInformationJ(detectionsDict=detectionsDict)
            updateInformationDeltaJi(ego=ctrl)

        if explorationActive:
            J = randomCloudDistCost
            deltaUpdate = "delta"
        elif exploitationActive:
            J = informationScore
            deltaUpdate = "directInformationJ"
            # deltaUpdate = "deltaInformationJi"

        costJ.append(J)
        print(f"[INFO] Cost J:{J:.8f}")

        threadList = []
        for i,drone in enumerate(controllers):
            argsDict = dict(ego = drone,
                            detectionsDict = detectionsDict.copy(),
                            excludedDict = excludedDict.copy(),
                            updateRule = deltaUpdate)
            thread = Thread(target = updateDelta, kwargs=argsDict)
            thread.start()
            threadList.append(thread)
        for thread in threadList:
            thread.join()

        time.sleep(timeInterval)

    print(f"[MONITORING] finished position {posInd}" )


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
    os.system('TASKKILL /F /IM CityEnviron*')

if __name__ == "__main__":

    global options, controllers, globalHawk

    options = get_options()

    generatingResultsFolders()

    fillTemplate()

    launchAirSim()

    wayPointsSize = options.waypoints

    OFFSETS = {"UAV1":[0,0,0],
               "UAV2":[0,-5,0],
              }

    dronesID = list(OFFSETS.keys())

    ip_id = f"127.0.0.{options.ip}"
    client = airsim.MultirotorClient(ip = ip_id)
    client.confirmConnection()

    if GLOBAL_HAWK_ACTIVE:
        setGlobalHawk(client)

    controllers = []
    for drone in dronesID:
        controllers.append(controller(client, drone, OFFSETS[drone],
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
        ctrl.setGeoFence(x = fenceX, y = fenceY, z = fenceZ, r=fenceR)

    print("\nTaking initial photos")
    for ctrl in controllers:
        # no need for task list (just setting values here)
        ctrl.getImages()

    # collisionCorrectionThread = Thread(target = collisionCorrection)
    # collisionCorrectionThread.start()

    startTime = time.time()

    global similarityList, informationJ, survaillanceJ, costJ, evaluator
    similarityList = []
    informationJ = []
    survaillanceJ = []
    costJ = []

    evaluator = evaluate()
    evaluator.setGeoFence(x=fenceX, y=fenceY,z=fenceZ,r=fenceR)
    evaluator.randomPoints(pointsSize = 500)

    for positionIdx in range(0,wayPointsSize):

        ptime = time.time()

        tasks = []
        for ctrl in controllers:
            t = ctrl.moveOmniDirectional(maxTravelTime=options.maxTravelTime,
                                         maxYaw=options.maxYaw,
                                         plotEstimator=False,
                                         # minDist=options.maxTravelTime*2.5,
                                         minDist=options.maxTravelTime*1.2,
                                         lidar=True,
                                         controllers=controllers)
            tasks.append(t)

        # TODO: Chech if we have collision, if yes, then move drone to previous position
        for task in tasks[::-1]:
            task.join()

        # so drones start fromw worst positioning
        if positionIdx == 0:
            for i,ctrl in enumerate(controllers):
                np.random.seed()
                print("\nSetting random Yaw all drones")
                yawRandom = 90
                ctrl.rotateToYaw(yawRandom)
                # ctrl.rotateToYaw(-5 + i*90)

        monitor(dronesID, positionIdx)

        for ctrl in controllers:

            positions = ctrl.getPositions()
            orientation = ctrl.getOrientation()

            x = positions.x_val
            y = positions.y_val
            z = positions.z_val
            _,_,yaw = airsim.to_eularian_angles(orientation)

            print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{np.degrees(yaw):.2f}) with Ji:{ctrl.getJi():.2f}")

        threadList = []
        for ctrl in controllers:
            thread = Thread(target = ctrl.updateEstimator)
            thread.start()
            threadList.append(thread)
        for thread in threadList:
            thread.join()

        # for ctrl in controllers: ctrl.plotEstimator1DoF()

        print(f"----- elapsed time: {time.time() - ptime:.3f} ------")
        print("---------------------------------\n")

        plotData(data=informationJ, folder="information", file="information")
        plotData(data=survaillanceJ, folder="survaillance", file="survaillance")

        # globalView()
        if GLOBAL_HAWK_ACTIVE:
            globalViewScene()
        # globalViewDetections()

    # collisionCorrectionThread.join()

    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "similarity_objects",
                            f"similarityList.pickle")
    pickle.dump(similarityList,open(file_out,"wb"))

    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "information",
                            f"informationAggregated.pickle")
    pickle.dump(informationJ,open(file_out,"wb"))

    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "costJ",
                            f"costJ.pickle")
    pickle.dump(costJ,open(file_out,"wb"))

    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "survaillance",
                            f"survaillanceJ.pickle")
    pickle.dump(survaillanceJ,open(file_out,"wb"))

    print("\n[RESETING] to original state ....")
    for ctrl in controllers: ctrl.quit()
    client.reset()

    print(f"\n[KILLING|AIRSIM] closing CityEnviron.exe")
    killAirSim()

    print(f"\n --- elapsed time:{startTime - time.time():.2f} [sec] ---")
