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

DRONE2_PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\2Drones"
DRONE3_PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\3Drones"
DRONE4_PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\GridSearch\V07"
DRONE5_PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\5Drones"

DRONES_PATH = {"UAVs 2":DRONE2_PATH,
               "UAVs 3":DRONE3_PATH,
               "UAVs 4":DRONE4_PATH,
               "UAVs 5":DRONE5_PATH}

plotstyle="ggplot"
plt.style.use(f"{plotstyle}")

if __name__ == "__main__":

    dataFrameList = []

    for drones_number,simulation_dir in DRONES_PATH.items():

        directories = os.listdir(simulation_dir)
        # print(f"directories {directories}")
        print(f"___ Drones Number Average J: {drones_number} ____")
        result_dirs = [i for i in directories if 'results_' in i]

        data = {}
        data["Simulation"] = []
        data["Time[sec]"] = []
        data["Confidence Interval"] = []
        data["UAV Number"] = []

        for sim in result_dirs:

            sim_dir = os.path.join(simulation_dir, sim)
            try:
                pickle_in = os.path.join(sim_dir, "information", "informationAggregated.pickle")
                file = open(pickle_in, "rb")
                informationJ = np.array(pickle.load(file))
                file.close()
            except:
                print(f"#### FAILED LOADING SIMULATION {sim_dir}")
                continue

            for time, confJ in enumerate(informationJ):
                data["Simulation"].append(sim)
                data["Time[sec]"].append(time*3.)
                data["Confidence Interval"].append(confJ)
                data["UAV Number"].append(drones_number)

        dataFrame = pd.DataFrame(data)
        dataFrameList.append(dataFrame)

        sns.lineplot(x="Time[sec]", y="Confidence Interval", data=dataFrame)

        print(f"[SAVE] Dumping pickle object")
        pickle.dump(dataFrame,open(f"{drones_number}.p", "wb"))

        # plt.show()
        plt.savefig(f"AverageJ_{drones_number}.png")
        plt.close()

    dataFrameCommon = pd.concat(dataFrameList)
    sns.lineplot(x="Time[sec]",
                 y="Confidence Interval",
                 hue="UAV Number",
                 data=dataFrameCommon[ dataFrameCommon['Time[sec]']<300*3])
    plt.savefig(f"AverageJ_AllUav.png", dpi=1000)
    plt.close()
