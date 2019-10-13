import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import utilities.utils as utils

# Enters all directories and creates 3d plots (saves them as pickle objects)
# and pickle object with the x,y,z,colors data (relative and absolute)

if __name__ == "__main__":

    parent_dir = os.path.join(os.getcwd(), "swarm_raw_output")

    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(parent_dir, dronesID[0]))

    for droneIdx, drone in enumerate(dronesID):
        print(f"\n[DRONE]: {drone}")

        for positionIdx, position in enumerate(wayPointsID):
            print(f"{4*' '}[POSITION]: position_{positionIdx}")

            current_dir = os.path.join(parent_dir, drone, f"position_{positionIdx}")

            depth_image = os.path.join(current_dir, "depth_time_0.pfm")
            state_pickle = os.path.join(current_dir, "state_time_0.pickle")

            x,y,z,colors = utils.kickstart(random_points=[100,100,"circle"],
                                           file_pfm=depth_image)

            xAbs, yAbs, zAbs = utils.to_absolute_coordinates(x,y,z,state_pickle)

            plot = utils.plot3dColor(xAbs,yAbs,zAbs,colors,size=0.3)

            plot_out = os.path.join(current_dir, "plot3D.pickle")
            pickle.dump(plot,open(plot_out,"wb"))

            coordinates = [x,y,z,xAbs,yAbs,zAbs,colors]
            coordinates_out = os.path.join(current_dir, "coordinates3D.pickle")
            pickle.dump(coordinates,open(coordinates_out,"wb"))
