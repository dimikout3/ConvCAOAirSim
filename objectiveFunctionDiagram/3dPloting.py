import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
from functools import reduce

plotstyle="ggplot"
TIME_STEP = 0
MAX_Z = -10
MAX_X =35
MIN_X =10
MAX_Y =45
MIN_Y =25

DRONE_ID_LIST = ["Drone1", "Drone2"]
DRONE_ID_LIST = ["Drone1"]

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for DRONE_ID in DRONE_ID_LIST:
        path = os.path.join(os.getcwd(), "results_1", "swarm_raw_output", DRONE_ID, f"pointCloud_{DRONE_ID}.pickle")

        cloud = pickle.load( open(path,"rb") )

        x = cloud[TIME_STEP][0]
        y = cloud[TIME_STEP][1]
        z = cloud[TIME_STEP][2]
        c = cloud[TIME_STEP][3]/255

        #  keep point below a height
        z_mask = np.where(z>MAX_Z)[0]
        x = x[z_mask]
        y = y[z_mask]
        z = z[z_mask]
        c = c[z_mask]

        # x restriction
        x_min_ind = np.where(x>MIN_X)[0]
        x_max_ind = np.where(x<MAX_X)[0]
        x_max =  reduce(np.intersect1d,(x_min_ind, x_max_ind))
        x = x[x_max]
        y = y[x_max]
        z = z[x_max]
        c = c[x_max]

        randomPoints = np.random.randint(0,x.shape, size=20000)
        x = x[randomPoints]
        y = y[randomPoints]
        z = z[randomPoints]
        c = c[randomPoints]

        ax.scatter(x, y, z, s=0.1, c=c)

    plt.axis('off')
    plt.tight_layout()
    ax.set_facecolor('black')
    # ax.set_facecolor((1.0, 0.47, 0.42))

    plt.show()
