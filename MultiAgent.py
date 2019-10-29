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

CAM_YAW = -0.5
CAM_PITCH = 0.
CAM_ROOL = 0.

NORM = {'information':30.0, 'mutualLow':1.0}
WEIGHT = {'information':1.0, 'similarity':40.0}

def monitor(droneList, posInd, timeInterval = 1, totalTime = 1):

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
            x,y,z,c = ctrl.getPointCloud()

            pointCloud = np.stack((x,y,z), axis=1)

            cloudPoints[ctrl.getName()] = pointCloud

            informationScore += ctrl.getScore(index=-1, absolute=True)

        sum, avg = similarityOut(cloudPoints, similarityKPI="DistRandom")
        # Apply boundaries
        avg = NORM['mutualLow'] if avg<NORM['mutualLow'] else avg
        similarityAvgNorm = 1/avg
        print(f"[INFO] Similarity avg:{avg:.3f}, norm:{similarityAvgNorm:.3f}")
        similarityList.append(similarityAvgNorm)

        informationScoreNorm = informationScore/NORM["information"]
        print(f"[INFO] Information Score combined:{informationScore:.3f}, norm:{informationScoreNorm:.3f}")
        informationScoreList.append(informationScoreNorm)

        J = informationScoreNorm*WEIGHT['information'] - similarityAvgNorm*WEIGHT['similarity']
        costJ.append(J)
        print(f"[INFO] Cost J:{J:.3f}")

        # TODO: computational complex ... simplify
        for ctrl in controllers:

            if posInd == 0:
                # the initial value for the output of the estimator is the overall J
                ctrl.appendJi(J)

            information = []
            similarity = []
            cloudPoints = {}

            for other in controllers:

                if other.getName() != ctrl.getName():

                    information.append(other.getScore(index=-1, absolute=True))
                    x,y,z,c = other.getPointCloudList(index=-1)
                    pointCloud = np.stack((x,y,z), axis=1)
                    cloudPoints[other.getName()] = pointCloud

                else:

                    information.append(other.getScore(index=-2, absolute=True))
                    x,y,z,c = other.getPointCloudList(index=-2)
                    pointCloud = np.stack((x,y,z), axis=1)
                    cloudPoints[other.getName()] = pointCloud

            _, avg = similarityOut(cloudPoints, similarityKPI="DistRandom")
            avg = NORM['mutualLow'] if avg<NORM['mutualLow'] else avg
            similarityAvgNorm = 1/avg
            informationScoreNorm = np.sum(information)/NORM['information']

            j_i = informationScoreNorm*WEIGHT['information'] - similarityAvgNorm*WEIGHT['similarity']
            delta = costJ[-1] - j_i

            print(f"[INFO] {ctrl.getName()} has delta:{delta:.4f} Ji:{j_i:.4f}")
            ctrl.appendContribution(delta)
            ctrl.appendJi(j_i)

        time.sleep(timeInterval)

    print(f"[MONITORING] finished position {posInd}" )


# TODO: move it to utilities.utils
def generatingResultsFolders():

    result_folder = os.path.join(os.getcwd(), "results")
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

PATH = {"Drone1":[(x,-10,-12,5) for x in range(100,-100,-5)],
        "Drone2":[(0.0,y,-8,5) for y in range(100,-100,-5)],
        }

OFFSETS = {"Drone1":[-25,0,0],
           "Drone2":[80,0,0]
          }


dronesID = list(PATH.keys())
wayPointsSize = len(PATH[dronesID[0]])
print(f"Detected {dronesID} with {wayPointsSize} positions")

generatingResultsFolders()

client = airsim.MultirotorClient()
client.confirmConnection()

controllers = []
for drone in dronesID:
    controllers.append(controller(client, drone, OFFSETS[drone]))

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
    ctrl.setGeoFence(x=10, y=20, z=-10, r=50)

print("\nSetting random Yaw all drones")
for ctrl in controllers:
    np.random.seed()
    yawRandom = np.random.uniform(-180,180,1)
    ctrl.rotateToYaw(yawRandom)

wayPointsSize = 50

startTime = time.time()

global similarityList, informationScoreList, costJ
similarityList = []
informationScoreList = []
costJ = []

for positionIdx in range(0,wayPointsSize):

    ptime = time.time()

    for ctrl in controllers:
        # ctrl.randomMoveZ()
        # ctrl.move()
        ctrl.move1DoF()
        # t = Thread(target = ctrl.randomMoveZ)
        # t.start()
        # x,y,z,speed = PATH[ctrl.getName()][positionIdx]
        # ctrl.moveToPostion(x,y,z,speed)

    for ctrl in controllers:

        positions = ctrl.getPositions()
        orientation = ctrl.getOrientation()

        x = positions.x_val
        y = positions.y_val
        z = positions.z_val
        _,_,yaw = airsim.to_eularian_angles(orientation)

        print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f} ,z:{z:.2f}, yaw:{np.degrees(yaw):.2f})")

    monitor(dronesID, positionIdx)

    # for ctrl in controllers: ctrl.updateEstimator()
    for ctrl in controllers: ctrl.updateEstimator1DoF()

    print(f"----- elapsed time: {time.time() - ptime:.3f} ------")
    print("---------------------------------\n")

    if (positionIdx % 1) == 0:

        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(costJ, label="Cost J")

        information = [info*WEIGHT["information"] for info in informationScoreList]
        ax1.plot(information, label="information")

        similarity = [sim*WEIGHT["similarity"] for sim in similarityList]
        ax1.plot(similarity, label="similarity")

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")

        ax1.legend()

        for ctrl in controllers:
            stateList = ctrl.getStateList()
            yawList = [np.degrees(airsim.to_eularian_angles(state[0].kinematics_estimated.orientation)[2]) for state in stateList]
            ax2.plot(yawList, label=ctrl.getName())

        ax2.set_ylim(-180,180)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Yaw [degrees]")
        ax2.legend()

        plt.tight_layout()

        report_plot = os.path.join(os.getcwd(),"results", "report",
                                f"report_{positionIdx}.png")
        plt.savefig(report_plot)
        # plt.show(block=False)
        # plt.pause(5)
        plt.close()

file_out = os.path.join(os.getcwd(),"results", "similarity_objects",
                        f"similarityList.pickle")
pickle.dump(similarityList,open(file_out,"wb"))

file_out = os.path.join(os.getcwd(),"results", "information",
                        f"scoreAggregated.pickle")
pickle.dump(informationScoreList,open(file_out,"wb"))

file_out = os.path.join(os.getcwd(),"results", "costJ",
                        f"costJ.pickle")
pickle.dump(costJ,open(file_out,"wb"))

print("\n[RESETING] to original state ....")
for ctrl in controllers: ctrl.quit()
client.reset()
print(f"\n --- elapsed time:{startTime - time.time():.2f} [sec] ---")
