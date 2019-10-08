import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import test3D.utils as utils

# Enters all directories and creates 3d plots (saves them as pickle objects)

if __name__ == "__main__":

    parent_dir = os.path.join(os.getcwd(), "swarm_raw_output")

    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(parent_dir, dronesID[0]))

    xAggregated = np.array([])
    yAggregated = np.array([])
    zAggregated = np.array([])
    colorsAggregated = np.array([[0,0,0]])

    for droneIdx, drone in enumerate(dronesID):
        print(f"\n[DRONE]: {drone}")

        for positionIdx, position in enumerate(wayPointsID):
            print(f"{4*' '}[POSITION]: {position}")

            current_dir = os.path.join(parent_dir, drone, position)
            coordinates_pickle = os.path.join(current_dir, "coordinates3D.pickle")

            coordinates = pickle.load(open(coordinates_pickle,"rb"))

            xAggregated = np.concatenate((xAggregated,coordinates[3]))
            yAggregated = np.concatenate((yAggregated,coordinates[4]))
            zAggregated = np.concatenate((zAggregated,coordinates[5]))
            colorsAggregated = np.concatenate((colorsAggregated,coordinates[6]))

            # utils.plot3dColor(xAggregated,yAggregated,zAggregated,colorsAggregated[1:],
            #                   size=0.3,pose=[30,-60],save_path=)

    coordinates_aggregated = os.path.join(parent_dir, "coordinates3D_Aggregated.pickle")
    coordinates_data = [xAggregated, yAggregated, zAggregated, colorsAggregated[1:]]
    pickle.dump(coordinates_data,open(coordinates_aggregated,"wb"))
