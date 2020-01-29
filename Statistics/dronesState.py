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
import matplotlib.image as mpimg

PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\GridSearch\V07"

fenceR = 70
fenceX = 25
fenceY = -25
fenceZ = -14
#positions of GlobalHawk
Xglobal = fenceX
Yglobal = fenceY
Zglobal = -90

if __name__ == "__main__":

    global simulation_dir
    if PATH == "":
        simulation_dir = os.path.join(os.getcwd(), "..")
    else:
        simulation_dir = PATH

    directories = os.listdir(simulation_dir)
    print(f"directories {directories}")
    result_dirs = [i for i in directories if 'results_' in i]
    # for safety reasons, last simulation is usally corrupted
    # result_dirs = result_dirs[0:-2]

    data = {}
    data["Simulation"] = []
    data["Time"] = []
    data["State"] = []
    data["droneID"] = []
    data["x_val"] = []
    data["y_val"] = []

    for sim in result_dirs:

        sim_dir = os.path.join(simulation_dir, sim)
        raw_dir = os.path.join(sim_dir, "swarm_raw_output")

        directories = os.listdir(raw_dir)
        # print(f"directories {directories}")
        drone_dirs = [i for i in directories if 'Drone' in i]

        for droneID in drone_dirs:

            print(f"[INFO] simulation={sim} drone={droneID}")

            pickle_in = os.path.join(raw_dir, droneID, f"state_{droneID}.pickle")
            file = open(pickle_in, "rb")
            stateList = np.array(pickle.load(file))
            file.close()

            for time, state in enumerate(stateList):
                #  state = [multirotorState, cameraState]
                data["Simulation"].append(sim)
                data["Time"].append(time*3.)
                data["State"].append(state)

                x_val = state[0].kinematics_estimated.position.x_val
                y_val = state[0].kinematics_estimated.position.y_val
                data["x_val"].append(x_val)
                data["y_val"].append(y_val)

                data["droneID"].append(droneID)

    image_path = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\GridSearch\V07\results_1\globalViewDetections\globalViewDetection_0.png"
    map_img = mpimg.imread(image_path)
    # plt.imshow(map_img)

    df = pd.DataFrame(data)
    # print(df.info)
    df = df[df.Time>(350*3)]
    # https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    hmax = sns.kdeplot(df.y_val , df.x_val, cmap="Reds",
                       shade=False,
                       shade_lowest=False)
    hmax.collections[0].set_alpha(.0)

    # hmax.imshow(map_img,
    #       aspect = hmax.get_aspect(),
    #       extent = hmax.get_xlim() + hmax.get_ylim(),
    #       zorder = 0) #put the map under the heatmap

    # The extent kwarg controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top)
    plt.imshow(map_img, zorder=1, extent=[-100, 100, -100.0, 100.0])
    plt.show()
