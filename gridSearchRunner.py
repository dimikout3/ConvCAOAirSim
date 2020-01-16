# import setup_path
# import airsim

import os
import time
import subprocess as sp
import shutil
import itertools
import numpy as np

# Number of simulations
SAVE_LOG = True

def deleteExistingResults():

    print("Deleting any previous existing 'results'")

    files = os.listdir()

    for file in files:

        if "results_" in file:
            path = os.path.join(os.getcwd(),file)
            shutil.rmtree(path)

def getGridList():

    maxYaw = np.linspace(2.5,7.5,3)
    maxTravelTime = np.linspace(1.,5.,3)
    estimatorWindow = np.linspace(10,30,3)

    gridList = list(itertools.product(maxYaw,maxTravelTime,estimatorWindow))

    return gridList

def gridListReport(gridList):

    fout = open(f"gridSearchHyperparameters.txt","w")
    for ip, (maxYaw, maxTravelTime, estimatorWindow) in enumerate(gridList):
        simString = f"Results_{ip}:\n  maxYaw={maxYaw}\n  maxTravelTime={maxTravelTime}\n  estimatorWindow={estimatorWindow}\n--------------\n"
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

        call = ["python","MultiAgent.py",
                "--ip", str(ip),
                "--waypoints", str(400),
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
