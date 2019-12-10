import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import utils

# Enters all directories and creates 3d plots (saves them as pickle objects)
# and pickle object with the x,y,z,colors data (relative and absolute)

WIDTH = int(21000/4)
HEIGHT = int(9000/4)

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

            dronePics.append(frame_detected)

    frameTop = np.concatenate((dronePics[0],dronePics[1]), axis = 1 )
    frameBot = np.concatenate((dronePics[1],dronePics[2]), axis = 1 )

    frameOut = np.concatenate((frameTop, frameBot))

    return frameOut


def globalView(positionIdx):

    global simulation_dir

    globalView_dir = os.path.join(simulation_dir, "globalView", f"globalView_{positionIdx}.png")

    frame_detected = cv2.imread(globalView_dir)

    # height, width, colors = frame_detected.shape
    # frame_detected = frame_detected[:,int(width/2):,:]

    return frame_detected


if __name__ == "__main__":

    global parent_dir, detected_dir, simulation_dir
    simulation_dir = os.path.join(os.getcwd(), "..", "results_1")

    parent_dir = os.path.join(simulation_dir, "swarm_raw_output")
    detected_dir = os.path.join(simulation_dir, "swarm_detected")

    video_dir = os.path.join(simulation_dir, "videos")
    try:
        os.makedirs(video_dir)
    except OSError:
        if not os.path.isdir(video_dir):
            raise

    global dronesID, wayPointsID
    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    file_out = os.path.join(video_dir,f"demo.avi")
    out = cv2.VideoWriter(file_out, fourcc, 1.0, (WIDTH,HEIGHT))

    for positionIdx, position in enumerate(wayPointsID):
        print(f"{4*' '}[POSITION]: position_{positionIdx}")

        frame1 = detectedImages(positionIdx)
        frame2 = globalView(positionIdx)

        frame_detections = cv2.resize(frame1,(int(WIDTH),int(HEIGHT)))
        frame_globalView = cv2.resize(frame2,(int(WIDTH),int(HEIGHT)))

        frame_concated = np.concatenate((frame_detections, frame_globalView), axis=1)
        frame_out = cv2.resize(frame_concated,(WIDTH,HEIGHT))

        out.write(frame_out)

    out.release()
