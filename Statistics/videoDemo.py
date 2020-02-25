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


def detectedImages(positionIdx):

    global dronesID, wayPointsID, parent_dir

    dronePics = []
    colors = [(255,0,0),(0,0,255),(255,0,255),(0,255,255)]

    for droneIdx, drone in enumerate(dronesID):

        current_raw_dir = os.path.join(parent_dir, drone, f"position_{positionIdx}")
        current_detected_dir = os.path.join(detected_dir, drone, f"position_{positionIdx}")

        time_steps = os.listdir(current_detected_dir)

        for time,time_step in enumerate(time_steps):

            detected_image_path = os.path.join(current_detected_dir, f"detected_time_{time}.png")

            frame_detected = cv2.imread(detected_image_path)

            textOffsetX, textOffsetY = 50, 50
            rectangleWidth = 155
            rectangleHeight = 48
            # transparency
            # https://gist.github.com/IAmSuyogJadhav/305bfd9a0605a4c096383408bee7fd5c
            cv2.rectangle(frame_detected, (textOffsetX - 5, textOffsetY + 5),
                                          (textOffsetX + rectangleWidth, textOffsetY - rectangleHeight),
                                          (127,127,115), cv2.FILLED)
            cv2.putText(frame_detected, f"{drone}", (textOffsetX, textOffsetY),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)

            bordersize = 40
            frame_detected = cv2.copyMakeBorder(frame_detected,
                                        top=bordersize,
                                        bottom=0,
                                        left=bordersize,
                                        right=0,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=[255, 255, 255])

            dronePics.append(frame_detected)

    frameTop = np.concatenate((dronePics[0],dronePics[1]), axis = 1 )
    frameBot = np.concatenate((dronePics[2],dronePics[3]), axis = 1 )

    frameOut = np.concatenate((frameTop, frameBot))

    return frameOut


def globalView(positionIdx):

    global simulation_dir

    globalView_dir = os.path.join(simulation_dir, "globalView", f"globalViewScene_{positionIdx}.png")

    frame_detected = cv2.imread(globalView_dir)

    return frame_detected


def information(positionIdx, local=True):

    if local:

        frame_detected = cv2.imread(f"Videos/Information/info_{positionIdx}.png")
        # frame_detected = cv2.resize(frame_detected,(int(WIDTH/2),int(HEIGHT/3)))

        bordersize = int(WIDTH/2)
        frame_detected = cv2.copyMakeBorder(frame_detected,
                                    top=0,
                                    bottom=0,
                                    left=bordersize,
                                    right=15,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[255, 255, 255])

    else:

        global simulation_dir

        globalView_dir = os.path.join(simulation_dir, "information", f"information_{positionIdx}.png")

        frame_detected = cv2.imread(globalView_dir)

    return frame_detected


def generateInfos(positions):

    print("Generating Objective Function Time Plots")
    data = pickle.load(open("4 UAVs.p","rb"))

    jList = []
    for pos in tqdm(range(positions)):
        jList.append(data[data["Time Steps"] == pos]["Objective Function"].mean())

        plt.style.use("ggplot")

        plt.plot(jList)

        plt.ylabel("Objective Function")
        plt.xlabel("Time Step")

        plt.gcf().set_size_inches((8,4))
        plt.tight_layout()
        plt.savefig(f"Videos/information/info_{pos}.png")
        plt.close()


if __name__ == "__main__":

    global parent_dir, detected_dir, simulation_dir
    # simulation_dir = os.path.join(os.getcwd(), "..", "results_1")
    simulation_dir = RESUTLS_PATH

    parent_dir = os.path.join(simulation_dir, "swarm_raw_output")
    detected_dir = os.path.join(simulation_dir, "swarm_detected")

    video_dir = "Videos"

    global dronesID, wayPointsID
    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    # wayPointsID = wayPointsID[0:10]
    # generateInfos(len(wayPointsID))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    file_out = os.path.join(video_dir,f"demo.avi")
    out = cv2.VideoWriter(file_out, fourcc, len(wayPointsID)/TOTAL_TIME, (WIDTH,HEIGHT))

    for positionIdx, position in enumerate(tqdm(wayPointsID)):
        # print(f"{4*' '}[POSITION]: position_{positionIdx}")

        frame1 = detectedImages(positionIdx)
        frame2 = globalView(positionIdx)
        frame3 = information(positionIdx)

        frame_detections = cv2.resize(frame1,(int(WIDTH),int(HEIGHT)))
        frame_globalView = cv2.resize(frame2,(int(WIDTH),int(HEIGHT)))
        frame_information = cv2.resize(frame3,(int(WIDTH),int(HEIGHT/3)))

        # print(f"frame_information.shape={frame_information.shape}")
        # print(f"frame_detections.shape={frame_detections.shape}")
        # print(f"frame_globalView.shape={frame_globalView.shape}")
        frame_top = np.concatenate((frame_detections, frame_globalView), axis=1)
        frame_bot = frame_information

        frame_top = cv2.resize(frame_top,(int(WIDTH),int(HEIGHT*(2/3))))
        # frame_concated = np.concatenate((frame_detections, frame_globalView, frame_information), axis=1)
        # frame_concated = np.concatenate((frame_detections, frame_globalView), axis=1)
        frame_concated = np.concatenate((frame_top, frame_bot))
        frame_out = cv2.resize(frame_concated,(WIDTH,HEIGHT))

        bordersize = 40
        frame_out = cv2.copyMakeBorder(frame_out,
                                    top=bordersize,
                                    bottom=0,
                                    left=0,
                                    right=0,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[255, 255, 255])

        frame_out = cv2.resize(frame_out,(WIDTH,HEIGHT))

        out.write(frame_out)

    out.release()
