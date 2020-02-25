import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import tqdm

# Enters all directories and creates 3d plots (saves them as pickle objects)
# and pickle object with the x,y,z,colors data (relative and absolute)

WIDTH = int(4000*2/4)
HEIGHT = int(3000*2/4)
TOTAL_TIME = 60
# RESUTLS_PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\GridSearch\V07\results_6_1"
RESUTLS_PATH = "results_1"

fenceR = 70
fenceX = 25
fenceY = -25
fenceZ = -14

# ShotCut
#   No white border upon bird's eye view POSITION:975,42
#                                        size:690, 674
#   With white border upon bird's eye view POSITION:960,27
                                        #size:720, 703

if __name__ == "__main__":

    global parent_dir, detected_dir, simulation_dir
    # simulation_dir = os.path.join(os.getcwd(), "..", "results_1")
    simulation_dir = RESUTLS_PATH

    parent_dir = os.path.join(simulation_dir, "swarm_raw_output")
    detected_dir = os.path.join(simulation_dir, "swarm_detected")

    out_dir = os.path.join("Videos", "VideosHelpersPics")

    global dronesID, wayPointsID
    dronesID = os.listdir(parent_dir)

    # setting operastional area boundaries
    # black dashed line
    globalViewDir = os.path.join(parent_dir, "GlobalHawk", "position_0", "scene_time_0.png")
    globalViewImg = cv2.imread(globalViewDir)
    globalViewImg = cv2.cvtColor(globalViewImg, cv2.COLOR_BGR2RGB)
    # square resolution boundaries
    left, right = -114.43489074707031, 64.43489074707031
    bot, top = -64.43489074707031, 114.43489074707031
    plt.imshow(globalViewImg,extent=[left, right, bot, top])

    plt.xlim(left, right)
    plt.ylim(bot, top)

    # Add line with operations area
    theta = np.linspace(0,2*np.pi,500)
    # r = np.sqrt(fenceR)
    r = fenceR
    y = fenceY + r*np.cos(theta)
    x = fenceX + r*np.sin(theta)
    plt.plot(y,x,'k--')

    plt.grid(False)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "BoundariesV02.png"), bbox_inches = 'tight', dpi=1500)

    colors = {"UAV1":'r',
              "UAV2":'b',
              "UAV3":'m',
              "UAV4":'c'}

    for ind,droneID in enumerate(dronesID):

        if droneID == "GlobalHawk":
            continue

        pointCloud = pickle.load(open(os.path.join(parent_dir, droneID, f"pointCloud_{droneID}.pickle"),"rb"))
        x,y,z,col = pointCloud[0]
        plt.scatter(y, x, s=0.2, alpha=0.4, label=droneID, c = colors[droneID])

        plt.legend(markerscale=20, loc=1)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{droneID}.png"), bbox_inches = 'tight', dpi=1500)
