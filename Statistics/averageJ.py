import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import scipy
import pandas as pd
import seaborn as sns
# from scipy.signal import savgol_filter

PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\GridSearch\V07"
# PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\2Drones"
DRONES_NUM = "Drones: 4"
PICKLE_OUT = "Drones_4.p"

if __name__ == "__main__":

    global simulation_dir
    if PATH == "":
        simulation_dir = os.path.join(os.getcwd(), "..")
    else:
        simulation_dir = PATH

    directories = os.listdir(simulation_dir)
    print(f"directories {directories}")
    result_dirs = [i for i in directories if 'results_' in i]
    # for safety reasons, last simulation is usally corrupted
    # result_dirs = result_dirs[0:-2]

    data = {}
    data["Simulation"] = []
    data["Time[sec]"] = []
    data["Confidence Interval"] = []
    data["Drones Number"] = []

    for sim in result_dirs:

        sim_dir = os.path.join(simulation_dir, sim)
        pickle_in = os.path.join(sim_dir, "information", "informationAggregated.pickle")
        file = open(pickle_in, "rb")
        informationJ = np.array(pickle.load(file))
        file.close()

        for time, confJ in enumerate(informationJ):
            data["Simulation"].append(sim)
            data["Time[sec]"].append(time*3.)
            data["Confidence Interval"].append(confJ)
            data["Drones Number"].append(DRONES_NUM)
            # data["state"].append(steteList[time])
            # data["droneID"].append(droneID)

    dataFrame = pd.DataFrame(data)
    sns.lineplot(x="Time[sec]", y="Confidence Interval", data=dataFrame)

    print(f"[SAVE] Dumping pickle object")
    pickle.dump(dataFrame,open(PICKLE_OUT, "wb"))

    plt.show()
