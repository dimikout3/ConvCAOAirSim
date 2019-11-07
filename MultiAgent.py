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

settingsDir = r"C:\Users\dkoutras\Documents\AirSim"
envDir = r"C:\Users\dkoutras\Desktop\CityEnviron"

CAM_YAW = -0.5
CAM_PITCH = 0.
CAM_ROOL = 0.

NORM = {'information':10.0, 'similarity':10.0}
WEIGHT = {'information':1.0, 'similarity':-1.0}

def monitor(droneList, posInd, timeInterval = 1, totalTime = 1):

    global options

    print(f"[MONITORING] position {posInd}")

    detector = yoloDetector.yoloDetector()

    global similarityList, informationScoreList, costJ

    for timeStep in range(0,totalTime,timeInterval):

        cloudPoints = {}
        informationScore = 0.0

        for i,ctrl in enumerate(controllers):

            ctrl.updateState(posInd, timeStep)
            responses = ctrl.getImages(save_raw=True)
            detectionsInfo, detectionsCoordinates = ctrl.detectObjects(detector, save_detected=True)
            # x,y,z,c = ctrl.getPointCloud()
            # pointCloud = np.stack((x,y,z), axis=1)
            pointCloud = detectionsCoordinates

            cloudPoints[ctrl.getName()] = pointCloud

            informationScore += ctrl.getScore(index=-1, absolute=True)

        # sum, avg = similarityOut(cloudPoints, similarityKPI="DistRandom", ip=options.ip)
        sum, avg = similarityOut(cloudPoints, similarityKPI="DistExhaustive", ip=options.ip)
        # TODO: monta einai auto to = prepei na skeftw kati kalytero
        avg = sum

        # Apply boundaries
        # avg = NORM['mutualLow'] if avg<NORM['mutualLow'] else avg
        similarityAvgNorm = avg/NORM['similarity']
        print(f"[INFO] Similarity avg:{avg:.3f}, norm:{similarityAvgNorm:.3f}")
        similarityList.append(similarityAvgNorm)

        informationScoreNorm = informationScore/NORM["information"]
        print(f"[INFO] Information Score combined:{informationScore:.3f}, norm:{informationScoreNorm:.3f}")
        informationScoreList.append(informationScoreNorm)

        J = informationScoreNorm*WEIGHT['information'] + similarityAvgNorm*WEIGHT['similarity']
        costJ.append(J)
        print(f"[INFO] Cost J:{J:.3f}")

        # TODO: computational complex ... simplify
        for ctrl in controllers:

            information = []
            similarity = []
            cloudPoints = {}

            for other in controllers:

                if other.getName() != ctrl.getName():

                    information.append(other.getScore(index=-1, absolute=True))
                    # x,y,z,c = other.getPointCloudList(index=-1)
                    # pointCloud = np.stack((x,y,z), axis=1)
                    pointCloud = other.getDetectionsCoordinates(index=-1)
                    cloudPoints[other.getName()] = pointCloud

                else:

                    information.append(other.getScore(index=-2, absolute=True))
                    # x,y,z,c = other.getPointCloudList(index=-2)
                    # pointCloud = np.stack((x,y,z), axis=1)
                    pointCloud = other.getDetectionsCoordinates(index=-2)
                    cloudPoints[other.getName()] = pointCloud

            # _, avg = similarityOut(cloudPoints, similarityKPI="DistRandom", ip=options.ip)
            sum, avg = similarityOut(cloudPoints, similarityKPI="DistExhaustive", ip=options.ip)
            avg = sum

            # avg = NORM['mutualLow'] if avg<NORM['mutualLow'] else avg
            similarityAvgNorm = avg/NORM["similarity"]
            informationScoreNorm = np.sum(information)/NORM['information']

            J_isolation = informationScoreNorm*WEIGHT['information'] + similarityAvgNorm*WEIGHT['similarity']
            delta = costJ[-1] - J_isolation

            ctrl.appendContribution(delta)
            # ctrl.appendJi(J_isolation)
            if (posInd>=1):
                ctrl.appendJi(ctrl.getJi() + delta)
            else:
                ctrl.appendJi(costJ[-1] + delta)

            print(f"[INFO] {ctrl.getName()} has delta:{delta:.4f} Ji:{ctrl.getJi():.4f}")

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
    optParser.add_option("--ip", dest="ip", help="the ip of the simulations launched")
    options, args = optParser.parse_args()

    return options


def launchAirSim():

    call = f"{envDir}\\CityEnviron"
    sp.Popen(call, shell=True)
    time.sleep(10)

if __name__ == "__main__":

    global options
    options = get_options()

    generatingResultsFolders()

    fillTemplate()

    launchAirSim()

    wayPointsSize = 200

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
        controllers.append(controller(client, drone, OFFSETS[drone], ip=options.ip))

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
        ctrl.setGeoFence(x = 25, y = -25, z = -14, r=75)

    print("\nSetting random Yaw all drones")
    for i,ctrl in enumerate(controllers):
        np.random.seed()
        # yawRandom = np.random.uniform(-180,180,1)
        yawRandom = 30
        ctrl.rotateToYaw(yawRandom)
        # ctrl.rotateToYaw(-5 + i*90)

    startTime = time.time()

    global similarityList, informationScoreList, costJ
    similarityList = []
    informationScoreList = []
    costJ = []

    for positionIdx in range(0,wayPointsSize):

        ptime = time.time()

        for ctrl in controllers:
            # ctrl.randomMoveZ()
            ctrl.move()
            # ctrl.move1DoF()
            # t = Thread(target = ctrl.randomMoveZ)
            # t.start()
            # x,y,z,speed = PATH[ctrl.getName()][positionIdx]
            # ctrl.moveToPostion(x,y,z,speed)

        monitor(dronesID, positionIdx)

        for ctrl in controllers:

            positions = ctrl.getPositions()
            orientation = ctrl.getOrientation()

            x = positions.x_val
            y = positions.y_val
            z = positions.z_val
            _,_,yaw = airsim.to_eularian_angles(orientation)

            print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{np.degrees(yaw):.2f})")


        # for ctrl in controllers: ctrl.updateEstimator()
        for ctrl in controllers:
            # ctrl.updateEstimator1DoF()
            ctrl.updateEstimator()

        # for ctrl in controllers: ctrl.plotEstimator1DoF()

        print(f"----- elapsed time: {time.time() - ptime:.3f} ------")
        print("---------------------------------\n")

        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(costJ, label="Cost J")

        information = [info*WEIGHT["information"] for info in informationScoreList]
        ax1.plot(information, label="information")

        similarity = [sim*WEIGHT["similarity"] for sim in similarityList]
        ax1.plot(similarity, label="similar")

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")

        ax1.legend()

        for ctrl in controllers:
            # stateList = ctrl.getStateList()
            # yawList = [np.degrees(airsim.to_eularian_angles(state[0].kinematics_estimated.orientation)[2]) for state in stateList]
            # ax2.plot(yawList, label=ctrl.getName())
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
            x,y,z,col = ctrl.getPointCloud(x=200,y=200)
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

        for ctrl in controllers:
            detections = ctrl.getDetectionsCoordinates()
            x,y = detections[:,0], detections[:,1]
            plt.scatter(y, x, label=ctrl.getName())

        plt.legend()
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])

        plt.xlabel("Y-Axis (NetWork)")
        plt.ylabel("X-Axis (NetWork)")

        detected_objects_folder = os.path.join(os.getcwd(),f"results_{options.ip}",
                                               "detected_objects", f"detections_{positionIdx}.png")
        plt.savefig(detected_objects_folder)
        plt.close()


    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "similarity_objects",
                            f"similarityList.pickle")
    pickle.dump(similarityList,open(file_out,"wb"))

    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "information",
                            f"scoreAggregated.pickle")
    pickle.dump(informationScoreList,open(file_out,"wb"))

    file_out = os.path.join(os.getcwd(),f"results_{options.ip}", "costJ",
                            f"costJ.pickle")
    pickle.dump(costJ,open(file_out,"wb"))

    print("\n[RESETING] to original state ....")
    for ctrl in controllers: ctrl.quit()
    client.reset()
    print(f"\n --- elapsed time:{startTime - time.time():.2f} [sec] ---")
