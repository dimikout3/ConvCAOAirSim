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

    # plt.figure(num=None, figsize=(6, 4), dpi=80)
    fig, ax = plt.subplots(figsize=(6,4))

    # https://stackoverflow.com/questions/22591174/pandas-multiple-conditions-while-indexing-data-frame-unexpected-behavior
    dataFrame = dataFrame.drop(dataFrame.index[ (dataFrame['Swarm Size']=="6 UAVs") & (dataFrame['Objective Function']>82) ])

    g = sns.catplot(x="Swarm Size", y="Objective Function",kind="violin",inner=None, data=dataFrame[dataFrame['Time Steps']>150])
    sns.swarmplot(x="Swarm Size", y="Objective Function", color="k", size=0.4, data=dataFrame, ax=g.ax)

    # plt.show()
    plt.gcf().set_size_inches((6,4))
    plt.tight_layout()
    plt.savefig(f"Compare.png")
