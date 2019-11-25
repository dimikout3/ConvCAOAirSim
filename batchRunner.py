# import setup_path
# import airsim

import os
import time
import subprocess as sp
import shutil


VERBOSE = True
# Number of simulations
Nsim = 10

def deleteExistingResults():

    print("Deleting any previous existing 'results'")

    files = os.listdir()

    for file in files:

        if "results_" in file:
            path = os.path.join(os.getcwd(),file)
            shutil.rmtree(path)


if __name__ == "__main__":

    deleteExistingResults()

    for ip in range(1,Nsim+1):

        print(f"Running simulation {ip}")

        call = ["python","MultiAgent.py",
                "--ip", str(ip),
                "--waypoints", str(600)]
        s = sp.Popen(call, shell=True)

        s.wait()
