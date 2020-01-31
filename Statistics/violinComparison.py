import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

plotstyle="ggplot"
# sns.set_style(f"{plotstyle}")
plt.style.use(f"{plotstyle}")

COMPARISON_PLOT = "violin"
# COMPARISON_PLOT = "catplot"

COMPARE = ['Drones_2.p','Drones_3.p','Drones_4.p']

if __name__ == "__main__":

    dataFrameList = []
    for pickle_in in COMPARE:

        file = open(pickle_in, "rb")
        dataFrameList.append(pickle.load(file))
        file.close()

    dataFrame = pd.concat(dataFrameList)

    if COMPARISON_PLOT == "violin":
        # sns.violinplot(x="Drones Number", y="Confidence Interval",inner=None, data=dataFrame[dataFrame['Time[sec]']>150*3])
        # sns.violinplot(x="Drones Number", y="Confidence Interval",size=0.03, data=dataFrame[dataFrame['Time[sec]']>150*3])
        g = sns.catplot(x="Drones Number", y="Confidence Interval",kind="violin",inner=None, data=dataFrame[dataFrame['Time[sec]']>150*3])
        sns.swarmplot(x="Drones Number", y="Confidence Interval", color="k", size=0.4, data=dataFrame, ax=g.ax)
    elif COMPARISON_PLOT == "catplot":
        sns.catplot(x="Drones Number", y="Confidence Interval",kind="swarm",data=dataFrame)

    # plt.show()
    plt.savefig(f"Compare_{COMPARISON_PLOT}_{plotstyle}.png")
