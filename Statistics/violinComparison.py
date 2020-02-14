import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

plotstyle="ggplot"
# sns.set_style(f"{plotstyle}")
plt.style.use(f"{plotstyle}")

COMPARE = ['2 UAVs.p','3 UAVs.p','4 UAVs.p', '5 UAVs.p', '6 UAVs.p']

if __name__ == "__main__":

    dataFrameList = []
    for pickle_in in COMPARE:

        file = open(pickle_in, "rb")
        dataFrameList.append(pickle.load(file))
        file.close()

    dataFrame = pd.concat(dataFrameList)
    
    g = sns.catplot(x="UAV Number", y="Objective Function",kind="violin",inner=None, data=dataFrame[dataFrame['Time Steps']>150])
    sns.swarmplot(x="UAV Number", y="Objective Function", color="k", size=0.4, data=dataFrame, ax=g.ax)

    # plt.show()
    plt.savefig(f"Compare.png")
