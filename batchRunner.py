# import setup_path
# import airsim

import os
import time
import subprocess as sp
import shutil


# Number of simulations
Nsim = 10
SAVE_LOG = False

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
                "--waypoints", str(800)]

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
            print(f"Simulation {ip} finished successfully\n")
