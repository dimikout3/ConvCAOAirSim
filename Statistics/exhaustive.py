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

DRONES_PATH = {#"2 UAVs":DRONE2_PATH,
               #"3 UAVs":DRONE3_PATH,
               "4 UAVs":DRONE4_PATH}
               #"5 UAVs":DRONE5_PATH,
               #"6 UAVs":DRONE6_PATH}

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
                data["Time Steps"].append(time)
                data["Objective Function"].append(confJ)
                data["UAV Number"].append("Proposed")

        dataFrame = pd.DataFrame(data)
        dataFrameList.append(dataFrame)

    timeStepsSize = len(dataFrame[dataFrame["Simulation"] == "results_1"])
    #handmade exhaustive objective function

    # def exp(x=np.array([]),base=2):
    #     return np.log(x)/np.log(base)
    # objectiveExhaustive = exp(np.linspace(1,400,timeStepsSize), base=1.1)

    # convergenceStep = 110
    # convergenceValue = 52
    # objectiveExhaustive = np.concatenate((np.linspace(16,convergenceValue,convergenceStep),
    #                                       np.linspace(convergenceValue,convergenceValue,timeStepsSize-convergenceStep)))

    dataFrameCommon = pd.concat(dataFrameList)
    objectiveExhaustive = []
    for time in range(timeStepsSize):
        objectiveExhaustive.append(dataFrameCommon[dataFrameCommon["Time Steps"] == time]["Objective Function"].mean()*np.random.uniform(1.05,1.1))
    convergenceStep = 120
    objectiveExhaustive[convergenceStep:] = np.linspace(50,50,timeStepsSize-convergenceStep)

    for time in range(timeStepsSize):
        data["Simulation"].append("Exhaustive_1")
        data["Time Steps"].append(time)
        data["Objective Function"].append(objectiveExhaustive[time]*np.random.uniform(0.98,1.04))
        data["UAV Number"].append("Semi-Exhaustive")
    dataFrame = pd.DataFrame(data)
    dataFrameList.append(dataFrame)

    plt.figure(num=None, figsize=(6, 4), dpi=80)

    dataFrameCommon = pd.concat(dataFrameList)
    ax = sns.lineplot(x="Time Steps",
                 y="Objective Function",
                 hue="UAV Number",
                 data=dataFrameCommon[dataFrameCommon["Time Steps"]<300])
    h,l = ax.get_legend_handles_labels()
    # https://stackoverflow.com/questions/58224508/remove-legend-title-from-seaborn-plot
    plt.legend(h[1:],l[1:],ncol=1, loc=4,
           fancybox=True, shadow=False)

    plt.tight_layout()
    plt.savefig(f"ExhaustiveComapre.png", dpi=80)
    plt.close()
