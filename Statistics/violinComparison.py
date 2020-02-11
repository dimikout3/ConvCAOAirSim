import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

plotstyle="ggplot"
# sns.set_style(f"{plotstyle}")
plt.style.use(f"{plotstyle}")

COMPARE = ['UAVs 2.p','UAVs 3.p','UAVs 4.p', 'UAVs 5.p']

if __name__ == "__main__":

    dataFrameList = []
    for pickle_in in COMPARE:

        file = open(pickle_in, "rb")
        dataFrameList.append(pickle.load(file))
        file.close()

    dataFrame = pd.concat(dataFrameList)

    g = sns.catplot(x="UAV Number", y="Confidence Interval",kind="violin",inner=None, data=dataFrame[dataFrame['Time[sec]']>150*3])
    sns.swarmplot(x="UAV Number", y="Confidence Interval", color="k", size=0.4, data=dataFrame, ax=g.ax)

    # plt.show()
    plt.savefig(f"Compare.png")
