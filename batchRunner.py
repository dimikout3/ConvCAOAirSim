# import setup_path
# import airsim

import os
import time
import subprocess as sp


VERBOSE = True
# Number of simulations
Nsim = 10


if __name__ == "__main__":

    for ip in range(1,Nsim+1):

        print(f"Running simulation {ip}")

        call = ["python","MultiAgent.py", "--ip", str(ip)]
        s = sp.Popen(call, shell=True)

        s.wait()
