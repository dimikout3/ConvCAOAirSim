import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
# from scipy.signal import savgol_filter


COMPARE = ['Drones_2.p','Drones_3.p','Drones_4.p']

if __name__ == "__main__":

    dataFrameList = []
    for pickle_in in COMPARE:

        file = open(pickle_in, "rb")
        dataFrameList.append(pickle.load(file))
        file.close()

    dataFrame = pd.concat(dataFrameList)

    sns.violinplot(x="Drones Number", y="Confidence Interval", data=dataFrame)

    plt.show()
