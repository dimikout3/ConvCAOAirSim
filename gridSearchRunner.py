# import setup_path
# import airsim

import os
import time
import subprocess as sp
import shutil
import itertools
import numpy as np

# Number of simulations
SAVE_LOG = False

def deleteExistingResults():

    print("Deleting any previous existing 'results'")

    files = os.listdir()

    for file in files:

        if "results_" in file:
            path = os.path.join(os.getcwd(),file)
            shutil.rmtree(path)

def getGridList(repeat=3):

    maxYaw = np.linspace(6.5,8.5,2)
    maxTravelTime = np.linspace(3.5,5.,2)
    estimatorWindow = [75]

    gridList = list(itertools.product(maxYaw,maxTravelTime,estimatorWindow))

    finalList = []
    for yaw, time, window in gridList:
        for i in range(repeat):
            finalList.append((yaw,time,window))

    return finalList

def gridListReport(gridList):

    fout = open(f"gridSearchHyperparameters.txt","w")
    for ip, (maxYaw, maxTravelTime, estimatorWindow) in enumerate(gridList):
        simString = f"Results_{ip+1}:\n  maxYaw={maxYaw}\n  maxTravelTime={maxTravelTime}\n  estimatorWindow={estimatorWindow}\n--------------\n"
        print(simString,file=fout)
    fout.close()


if __name__ == "__main__":

    deleteExistingResults()

    gridList = getGridList()
    gridListReport(gridList)

    for ip, (maxYaw, maxTravelTime, estimatorWindow) in enumerate(gridList):

        ip = ip +1

        stime = time.time()

        print(f"Running simulation {ip}")

        call = ["python","MultiAgentObjective.py",
                "--ip", str(ip),
                "--waypoints", str(300),
                "--maxYaw",str(maxYaw),
                "--maxTravelTime",str(maxTravelTime),
                "--estimatorWindow",str(int(estimatorWindow))]

        if SAVE_LOG:
            fout = open(f"stdOutput_{ip}.txt","w")
            s = sp.Popen(call, shell=True, stdout=fout)
            fout.close()
        else:
            s = sp.Popen(call, shell=True)

        exitCode = s.wait()

        if exitCode != 0:
            # error happened
            print(f"Simulation {ip} failed ...")
            print("Killing any AirSim Enviroment")
            os.system('TASKKILL /F /IM CityEnviron*')
        else:
            print(f"Simulation {ip} finished successfully after {time.time()-stime}[sec]\n")
