import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Enters all directories and creates 3d plots (saves them as pickle objects)
# and pickle object with the x,y,z,colors data (relative and absolute)

WIDTH = int(2100*2/4)
HEIGHT = int(900*2/4)
TOTAL_TIME = 120
RESUTLS_PATH = r"E:\Users\DKoutas\ownCloudConvCao\CREST_Shared\results\IROS\GridSearch\V07\results_6_1"

def detectedImages(positionIdx):

    global dronesID, wayPointsID, parent_dir

    dronePics = []

    for droneIdx, drone in enumerate(dronesID):

        current_raw_dir = os.path.join(parent_dir, drone, f"position_{positionIdx}")
        current_detected_dir = os.path.join(detected_dir, drone, f"position_{positionIdx}")

        time_steps = os.listdir(current_detected_dir)

        for time,time_step in enumerate(time_steps):

            detected_image_path = os.path.join(current_detected_dir, f"detected_time_{time}.png")

            frame_detected = cv2.imread(detected_image_path)
            cv2.putText(frame_detected, f"{drone}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (36,255,12), 5)

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


def information(positionIdx):

    global simulation_dir

    globalView_dir = os.path.join(simulation_dir, "information", f"information_{positionIdx}.png")

    frame_detected = cv2.imread(globalView_dir)

    return frame_detected


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

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    file_out = os.path.join(video_dir,f"demo.avi")
    out = cv2.VideoWriter(file_out, fourcc, len(wayPointsID)/TOTAL_TIME, (WIDTH,HEIGHT))

    for positionIdx, position in enumerate(wayPointsID):
        print(f"{4*' '}[POSITION]: position_{positionIdx}")

        frame1 = detectedImages(positionIdx)
        frame2 = globalView(positionIdx)
        frame3 = information(positionIdx)

        frame_detections = cv2.resize(frame1,(int(WIDTH),int(HEIGHT)))
        frame_globalView = cv2.resize(frame2,(int(WIDTH),int(HEIGHT)))
        frame_information = cv2.resize(frame3,(int(WIDTH),int(HEIGHT)))

        frame_concated = np.concatenate((frame_detections, frame_globalView, frame_information), axis=1)
        frame_out = cv2.resize(frame_concated,(WIDTH,HEIGHT))

        out.write(frame_out)

    out.release()
