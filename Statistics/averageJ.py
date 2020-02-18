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
DRONE6_PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\6Drones"

DRONES_PATH = {"2 UAVs":DRONE2_PATH,
               "3 UAVs":DRONE3_PATH,
               "4 UAVs":DRONE4_PATH,
               "5 UAVs":DRONE5_PATH,
               "6 UAVs":DRONE6_PATH}

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
        data["Time Steps"] = []
        data["Objective Function"] = []
        data["Swarm Size"] = []

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
                data["Time Steps"].append(time)
                data["Objective Function"].append(confJ)
                data["Swarm Size"].append(drones_number)

        dataFrame = pd.DataFrame(data)
        dataFrameList.append(dataFrame)

        sns.lineplot(x="Time Steps", y="Objective Function", data=dataFrame)

        print(f"[SAVE] Dumping pickle object")
        pickle.dump(dataFrame,open(f"{drones_number}.p", "wb"))

        # plt.show()
        plt.savefig(f"AverageJ_{drones_number}.png")
        plt.close()

    plt.figure(num=None, figsize=(6, 4), dpi=80)

    dataFrameCommon = pd.concat(dataFrameList)
    ax = sns.lineplot(x="Time Steps",
                 y="Objective Function",
                 hue="Swarm Size",
                 data=dataFrameCommon[ dataFrameCommon['Time Steps']<300])
    h,l = ax.get_legend_handles_labels()
    # https://stackoverflow.com/questions/58224508/remove-legend-title-from-seaborn-plot
    plt.legend(h[1:],l[1:],ncol=1,
           fancybox=True, shadow=False)

    plt.tight_layout()
    plt.savefig(f"AverageJ_AllUav.png", dpi=1000)
    plt.close()
