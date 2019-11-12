import numpy as np
import os
import time
from controller import controller
from matplotlib import pyplot as plt
import pickle
import optparse

KW = 10
xLine = 0.8
n_p = 2000
targetPoints = [ [xLine,i] for i in np.linspace(0,1,n_p)]


def plotState(posInd):

    global options

    for ctrl in controllers:
        x,y = ctrl.getPositions()
        plt.scatter(x,y,label=ctrl.getName())

    plt.vlines(xLine,0,1,linestyle=":")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.legend()
    plt.tight_layout()

    out = os.path.join(os.getcwd(),f"results_{options.ip}",
                       f"state_{posInd}.png")

    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def calculateCostJ(droneList, posInd):

    global costJ

    distR2P = np.zeros((n_p,len(droneList)))
    cellsAssigned = np.zeros(len(droneList))

    j = 0.

    for i in range(n_p):

        for r,drone in enumerate(controllers):

            xDrone, yDrone = drone.getPositions()
            xTarget, yTarget = targetPoints[i][0], targetPoints[i][1]

            distR2P[i,r] = np.sqrt((xDrone-xTarget)**2 + (yDrone-yTarget)**2)

        minDist = np.min(distR2P[i,:])
        # imin = np.unravel_index(np.argmin(distR2P[i,:]),distR2P[i,:].shape)
        imin = np.argmin(distR2P[i,:])

        cellsAssigned[imin] = cellsAssigned[imin] + 1

        j += minDist

    for i in range(len(droneList)):
        if cellsAssigned[i] == 0:
            j += KW*np.min(distR2P[:,i])

    costJ.append(j)

    # _________________ Update Ji _________________
    for ctrl in controllers:

        distR2P = np.zeros((n_p,len(droneList)))
        cellsAssigned = np.zeros(len(droneList))
        j_isolation = 0.

        for i in range(n_p):

            for other in controllers:

                if other.getName() != ctrl.getName():
                    xDrone, yDrone = other.getPositions(index=-1)
                    xTarget, yTarget = targetPoints[i][0], targetPoints[i][1]
                    distR2P[i,r] = np.sqrt((xDrone-xTarget)**2 + (yDrone-yTarget)**2)
                else:
                    xDrone, yDrone = other.getPositions(index=-2)
                    xTarget, yTarget = targetPoints[i][0], targetPoints[i][1]
                    distR2P[i,r] = np.sqrt((xDrone-xTarget)**2 + (yDrone-yTarget)**2)

            minDist = np.min(distR2P[i,:])
            # imin = np.unravel_index(np.argmin(distR2P[i,:]),distR2P[i,:].shape)
            imin = np.argmin(distR2P[i,:])
            cellsAssigned[imin] = cellsAssigned[imin] + 1

            j_isolation += minDist

        for i in range(len(droneList)):
            if cellsAssigned[i] == 0:
                j_isolation += KW*np.min(distR2P[:,i])

        delta = costJ[-1] - j_isolation

        if (posInd>=1):
            ctrl.updateJi(ctrl.getJi() + delta)
        else:
            ctrl.updateJi(costJ[-1] + delta)

        print(f"[INFO] {ctrl.getName()} has delta:{delta:.4f} Ji:{ctrl.getJi():.4f}")


def generatingResultsFolders():

    global options

    result_folder = os.path.join(os.getcwd(), f"results_{options.ip}")
    try:
        os.makedirs(result_folder)
    except OSError:
        if not os.path.isdir(result_folder):
            raise


def get_options():

    optParser = optparse.OptionParser()
    optParser.add_option("--ip", dest="ip", help="the ip of the simulations launched")
    options, args = optParser.parse_args()

    return options


if __name__ == "__main__":

    global options
    options = get_options()

    generatingResultsFolders()

    wayPointsSize = 50

    dronesID = ["Drone1", "Drone2", "Drone3", "Drone4"]

    controllers = []
    for drone in dronesID:
        controllers.append(controller(drone,ip=options.ip))

    global similarityList, informationScoreList, costJ
    similarityList = []
    informationScoreList = []
    costJ = []

    for ctrl in controllers:
        ctrl.setGeoFence()

    for positionIdx in range(0,wayPointsSize):

        ptime = time.time()
        print(f"_____ time step: {positionIdx}")
        for ctrl in controllers:
            ctrl.updateState()
            ctrl.move()

        calculateCostJ(dronesID, positionIdx)

        for ctrl in controllers:

            x,y = ctrl.getPositions()
            print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f})")

        for ctrl in controllers:
            # ctrl.updateEstimator1DoF()
            ctrl.updateEstimator()

        print(f"J={costJ[-1]}")

        plotState(positionIdx)
        # for ctrl in controllers: ctrl.plotEstimator1DoF()

        print(f"----- elapsed time: {time.time() - ptime:.3f} ------")
        print("---------------------------------\n")
