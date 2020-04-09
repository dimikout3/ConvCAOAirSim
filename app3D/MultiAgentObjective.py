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
import open3d as o3d
from functools import reduce
from scipy.spatial import distance

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

MAX_DIST_VIEW = baseSet['maxDistView']

CAM_YAW = baseSet['CamYaw']
CAM_PITCH = baseSet['CamPitch']
CAM_ROOL = baseSet['CamRoll']

geoFenceSet = baseSet['GeoFence']
fenceCenterX = geoFenceSet['centerX']
fenceCenterY = geoFenceSet['centerY']
fenceWidth = geoFenceSet['width']
fenceLength = geoFenceSet['length']
fenceHeight = geoFenceSet['height']

# ,
# "GlobalHawk": {
#   "VehicleType": "SimpleFlight",
#     "AutoCreate": true,
#   "X": 5, "Y": 0, "Z": 0
# }
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

    OFFSET_GLOBALHAWK = [5,0,0]
    globalHawk = controllerApp(client, "GlobalHawk", OFFSET_GLOBALHAWK, ip=options.ip)
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

    debugGlobalScene = False

    global controllers, globalHawk

    globalHawk.getImages()

    scene = globalHawk.imageScene
    cameraInfo = globalHawk.cameraInfo
    height, width, colors = scene.shape
    if debugGlobalScene:
        print(f"height={height} width={width} ")

    altitude = abs(cameraInfo.pose.position.z_val)
    hfov = cameraInfo.fov
    vFoV = (height/width)*hfov
    if debugGlobalScene:
        print(f"altitude={altitude} hFoV={hfov} vFoV={vFoV}")

    # what is the farest point global hawk can monitor
    Horizontalhypotenuse = altitude/np.cos( np.radians(hfov/2) )
    maxViewHorizontal = Horizontalhypotenuse*np.sin( np.radians(hfov/2) )
    if debugGlobalScene:
        print(f"Horizontalhypotenuse={Horizontalhypotenuse} maxViewHorizontal={maxViewHorizontal}")

    verticalhypotenuse = altitude/np.cos( np.radians(vFoV/2) )
    maxViewVertical = verticalhypotenuse*np.sin( np.radians(vFoV/2) )
    maxViewVertical = maxViewHorizontal*(height/width)
    if debugGlobalScene:
        print(f"verticalhypotenuse={verticalhypotenuse} maxViewVertical={maxViewVertical}")

    left, right = -maxViewHorizontal + fenceCenterY, maxViewHorizontal + fenceCenterY
    bottom, top = -maxViewVertical + fenceCenterX, maxViewVertical + fenceCenterX

    flipedScene = scene
    if debugGlobalScene:
        print(f"left={left} right={right} bot={bottom} top={top}")

    flipedScene = cv2.cvtColor(flipedScene, cv2.COLOR_BGR2RGB)
    plt.imshow(flipedScene,extent=[left, right, bottom, top])

    colors = ['r','b','m', 'c']
    for ind, ctrl in enumerate(controllers):
        # x,y,z,col = ctrl.getPointCloud(x=10,y=10)
        # print(f"[GlobalView] {ctrl.getName()} has x.size={len(x)} y.size={len(y)}")

        attributed = ctrl.attributed[-1]
        x,y,z = np.reshape(attributed, (3, attributed.shape[0]))

        plt.scatter(y, x, s=0.2, alpha=0.4, label=ctrl.getName(), c = colors[ind])

    # plt.show()

    plt.xlim(left, right)
    plt.ylim(bottom, top)

    # plt.gca().invert_yaxis()

    plt.grid(False)
    plt.axis('off')
    # ax1.set_xlabel("Y-Axis (NetWork)")
    # ax1.set_ylabel("X-Axis (NetWork)")
    # plt.margins(0,0)

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
        argsDict = dict(x = 30, y = 30)
        thread = Thread(target = ctrl.getPointCloud, kwargs=argsDict)
        thread.start()
        threadList.append(thread)
    for thread in threadList:
        thread.join()


def descretization():

    global controllers

    threadList = []
    for ctrl in controllers:
        thread = Thread(target = ctrl.pointCloud2Descrete)
        thread.start()
        threadList.append(thread)
    for thread in threadList:
        thread.join()


def updateIntermediate():

    global controllers

    threadList = []
    for ctrl in controllers:
        thread = Thread(target = ctrl.connectIntermidiate)
        thread.start()
        threadList.append(thread)
    for thread in threadList:
        thread.join()


def updateFrontier():

    global Explored, Obstacles, Frontier

    # Frontier[:,:,:] = False

    # neighbors -> (bot | top | left | right | front | back)
    bot = np.roll(np.logical_not(Explored),1, axis = 0)
    top = np.roll(np.logical_not(Explored),1, axis = 1)
    left = np.roll(np.logical_not(Explored),1, axis = 2)
    right = np.roll(np.logical_not(Explored),-1, axis = 0)
    front = np.roll(np.logical_not(Explored),-1, axis = 1)
    back = np.roll(np.logical_not(Explored),-1, axis = 2)
    neighbors = reduce( np.logical_or, (bot, top, left, right, front, back))

    canditates = np.logical_and(Explored, np.logical_not(Obstacles))

    indexes = np.where( np.logical_and(canditates, neighbors) )

    Frontier[indexes] = True


def attributeFrontierCells():

    """ Returning  attributes= [UAV1 -> [[FC1],[FC2],[FC3],[FC4]]
                                UAV2 -> [[FC1],[FC2],[FC3]]]"""

    global controllers, Frontier, discretizator

    frontierCellsIndexes = np.where(Frontier)
    frontierCellsIndexes = np.stack(frontierCellsIndexes, axis=1)
    frontierCellsIndexes = discretizator.toGroundTruth(frontierCellsIndexes)

    uav = []
    for ctrl in controllers:

        positions = ctrl.getPositions()
        orientation = ctrl.getOrientation()

        x = positions.x_val
        y = positions.y_val
        z = positions.z_val

        uav.append([x,y,z])
    uav = np.array(uav)

    dist = distance.cdist(uav, frontierCellsIndexes)

    argmin = np.argmin(dist, axis=0)

    attributed = [[] for _ in controllers]

    for ind, ctrl in enumerate(controllers):

        cellsInd = np.where(argmin == ind)

        if cellsInd[0].size == 0:
            argmin_single = np.argmin(distance.cdist(frontierCellsIndexes, [uav[ind]]), axis=0)
            attributed[ind].append(frontierCellsIndexes[argmin_single])
        else:
            attributed[ind].append(frontierCellsIndexes[cellsInd])

        # print(f"[ATTRIBUTED] 3d for UAV{ind}")
        # show3DVector(attributed[ind])

    return attributed


def updateMaps():

    global controllers, Explored, Obstacles, DescreteMap

    updateIntermediate()

    for ctrl in controllers:

        # descretePointCloud -> [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4] ...]
        descretePoint = ctrl.descretePointCloud[-1]
        # descretePoint.T -> [np.array([x1,x2,x3,x4]), np.array([y1,y2,y3,y4]), np.array([z1,z2,z3,z4])]
        x,y,z = descretePoint.T

        xIntermediate,yIntermediate,zIntermediate = ctrl.getIntermediate()

        Explored[(xIntermediate,yIntermediate,zIntermediate)] = True

        Obstacles[(xIntermediate,yIntermediate,zIntermediate)] = False
        # Obstacles[(x,y,z)] = True

        DescreteMap[(x,y,z)] = True

    updateFrontier()


def show3DMaps(map):

    indexes = np.where(map)

    x,y,z = indexes
    indexes = np.stack((x,y,z),axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(indexes)
    o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud


def show3DVector(vector):
    # import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vector)
    o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud


def show2DMaps(map):

    indexes = np.where(map)

    x,y,z = indexes

    plt.scatter(x, y, c=z, vmin=0, vmax=12)
    plt.colorbar()

    plt.show()


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
    optParser.add_option("--waypoints", default=200, dest="waypoints",type="int", help="the number of waypoints provided")
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

    global options, controllers, globalHawk, Explored, Obstacles, Frontier, DescreteMap, discretizator

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
                                      estimatorWindow=options.estimatorWindow,
                                      maxDistView = MAX_DIST_VIEW))

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

    print("\nLifting all drones to specified Z altitude")
    tasks = []
    intialAlt = -10
    stepAlt = -0.5
    for i,ctrl in enumerate(controllers):
        t = ctrl.moveToZ(intialAlt + stepAlt*i,2)
        tasks.append(t)
    for t in tasks: t.join()

    discretizator = Discretizator.Discretizator(discrete=baseSet['Discrete'], geofence=fence)
    discretizator.report()

    for ctrl in controllers:
        ctrl.updateDescretizator(discretizator)

    discr = baseSet['Discrete']
    Explored = np.zeros((discr['x'], discr['y'], discr['z']), dtype=bool)
    Obstacles = np.ones((discr['x'], discr['y'], discr['z']), dtype=bool)
    Frontier = np.zeros((discr['x'], discr['y'], discr['z']), dtype=bool)
    DescreteMap = np.zeros((discr['x'], discr['y'], discr['z']), dtype=bool)

    startTime = time.time()

    global costJ
    costJ = []

    for positionIdx in range(0,options.waypoints):
        print(f"position={positionIdx} Explored true = {np.where(Explored)[0].shape}")
        ptime = time.time()

        for ctrl in controllers:
            ctrl.updateState(positionIdx,0)
            if GLOBAL_HAWK_ACTIVE:
                globalHawk.updateState(positionIdx,0)

        getImages()
        getPointClouds()
        descretization()

        updateMaps()
        # data = controllers[0].descretePointCloud[-1]
        # discretizator.show(data)

        attributed = attributeFrontierCells()


        if positionIdx>=1:

            # print(f"Showing Explored")
            # show3DMaps(Explored)

            # print(f"Showing Obstacles")
            # show3DMaps(Obstacles)

            # print(f"Showing Frontier")
            # show3DMaps(Frontier)
            # show2DMaps(Frontier)

            # print(f"Showing DescreteMap")
            # show3DMaps(DescreteMap)
            pass

        tasks = []
        for ind, ctrl in enumerate(controllers):
            t = ctrl.move(frontierCellsAttributed = attributed[ind])
            tasks.append(t)
        for task in tasks[::-1]:
            task.join()

        for ctrl in controllers:

            positions = ctrl.getPositions()
            orientation = ctrl.getOrientation()

            x = positions.x_val
            y = positions.y_val
            z = positions.z_val
            _,_,yaw = airsim.to_eularian_angles(orientation)

            # print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{np.degrees(yaw):.2f}) with Ji:{ctrl.getJi():.2f}")
            print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{np.degrees(yaw):.2f})")

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
