import numpy as np
import os
import time
from controller import controller
from matplotlib import pyplot as plt
import pickle
import optparse

xLine = 0.8
n_p = 2000
targetPoints = [ [xLine,i] for i in np.linspace(0,1,n_p)]
KW = n_p/25

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
                       "state",f"state_{posInd}.png")

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
            # print(f"[INFO] {droneList[i]} has no point close, np.min(dist[:,i]:{np.min(distR2P[:,i])})")

    costJ.append(j)

    distStable = distR2P

    # _________________ Update Ji _________________
    for ctrl in controllers:

        distR2P = np.zeros((n_p,len(droneList)))
        cellsAssigned = np.zeros(len(droneList))
        j_isolation = 0.

        for i in range(n_p):

            for r,other in enumerate(controllers):

                xTarget, yTarget = targetPoints[i][0], targetPoints[i][1]

                # print(f"comparing other:{other.getName()} -- ego:{ctrl.getName()}")
                if other.getName() != ctrl.getName():
                    xDrone, yDrone = other.getPositions()
                else:
                    xDrone, yDrone = other.getPositions(index=-2)
                # xDrone, yDrone = other.getPositions()

                distR2P[i,r] = np.sqrt((xDrone-xTarget)**2 + (yDrone-yTarget)**2)
                # if other.getName() != ctrl.getName():
                #     print(f"point:{i} diff:{distR2P[i,r] - distStable[i,r]}")

            minDist = np.min(distR2P[i,:])
            # imin = np.unravel_index(np.argmin(distR2P[i,:]),distR2P[i,:].shape)
            imin = np.argmin(distR2P[i,:])
            cellsAssigned[imin] = cellsAssigned[imin] + 1

            j_isolation += minDist

        for i in range(len(droneList)):
            if cellsAssigned[i] == 0:
                j_isolation += KW*np.min(distR2P[:,i])

        delta = costJ[-1] - j_isolation
        # if delta<0: print("Success !!! negative Delta")

        if (posInd>=1):
            ctrl.updateJi(ctrl.getJi() + delta)
        else:
            ctrl.updateJi(costJ[-1] + delta)

        # if posInd>=2:
        #     # print(f"distStable.shape:{distStable.shape} distR2P.shape{distR2P.shape}")
        #     diff = distStable - distR2P
        #     plt.imshow(diff, aspect=0.5)
        #     plt.title(f"updating {ctrl.getName()}")
        #     plt.colorbar()
        #     plt.show()

        # print(f"[INFO] {ctrl.getName()} has delta:{delta:.4f} Ji:{ctrl.getJi():.4f}")
    print(f"[INFO] time = {posInd} - Cost J = {costJ[-1]}")


def generatingResultsFolders():

    global options

    result_folder = os.path.join(os.getcwd(), f"results_{options.ip}")
    try:
        os.makedirs(result_folder)
    except OSError:
        if not os.path.isdir(result_folder):
            raise

    state_folder = os.path.join(result_folder, f"state")
    try:
        os.makedirs(state_folder)
    except OSError:
        if not os.path.isdir(state_folder):
            raise

    canditate_folder = os.path.join(result_folder, f"canditates")
    try:
        os.makedirs(canditate_folder)
    except OSError:
        if not os.path.isdir(canditate_folder):
            raise


def get_options():

    optParser = optparse.OptionParser()
    optParser.add_option("--ip", dest="ip", help="the ip of the simulations launched")
    optParser.add_option("--ws", dest="size",type=int, help="waypoint size", default=100)
    options, args = optParser.parse_args()

    return options


if __name__ == "__main__":

    global options
    options = get_options()

    generatingResultsFolders()

    wayPointsSize = options.size

    # dronesID = ["Drone1", "Drone2", "Drone3", "Drone4", "Drone5", "Drone6"]
    # dronesID = ["Drone1"]
    dronesID = ["Drone1", "Drone2", "Drone3"]

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
        # print(f"\n_____ time step: {positionIdx}")
        for ctrl in controllers:
            ctrl.updateState(positionIdx)
            ctrl.move(maxDelta = 0.01, plot=False)

        calculateCostJ(dronesID, positionIdx)

        # for ctrl in controllers:
        #
        #     x,y = ctrl.getPositions()
        #     print(f"[INFO] {ctrl.getName()} is at (x:{x:.2f} ,y:{y:.2f})")

        for ctrl in controllers:
            # ctrl.updateEstimator1DoF()
            ctrl.updateEstimator()

        plotState(positionIdx)
        # for ctrl in controllers: ctrl.plotEstimator1DoF()
