import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

plt.style.use('ggplot')
# Enters all directories and creates 3d plots (saves them as pickle objects)
# and pickle object with the x,y,z,colors data (relative and absolute)

poisitionsSize = 50

def getKPIpng(position):

    score1 = pickle.load(open(os.path.join(os.getcwd(),"swarm_raw_output",
                              "score_detections_Drone1.pickle"),"rb"))
    score2 = pickle.load(open(os.path.join(os.getcwd(),"swarm_raw_output",
                              "score_detections_Drone2.pickle"),"rb"))

    fig = plt.figure(figsize=(10, 5))

    score1 = np.array(score1)
    score2 = np.array(score2)
    scoreTotal = score1 + score2

    x = [i for i in range(position)]

    y1 = score1[0:position]
    plt.plot(x,y1,label="Drone 1")

    y2 = score2[0:position]
    plt.plot(x,y2,label="Drone 2")

    y = scoreTotal[0:position]
    plt.plot(x,y,label="Combined")

    # plt.set_xlabel("Time")
    # plt.set_ylabel("Information")
    # plt.set_ylim([0, np.max(scoreTotal)*1.1])
    # plt.set_xlim([0,poisitionsSize])
    plt.xlabel("Time")
    plt.ylabel("Information")
    plt.ylim([0, np.max(scoreTotal)*1.1])
    plt.xlim([0,poisitionsSize])

    plt.legend()
    plt.tight_layout()

    file_name = os.path.join(os.getcwd(),"results","information_images",
                             f"information_position{position}.png")
    plt.savefig(file_name)

    plt.close()
    return file_name


if __name__ == "__main__":

    parent_dir = os.path.join(os.getcwd(), "swarm_raw_output")
    detected_dir = os.path.join(os.getcwd(), "swarm_detected")

    results_dir = os.path.join(os.getcwd(),"results")
    try:
        os.makedirs(results_dir)
    except OSError:
        if not os.path.isdir(results_dir):
            raise

    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(parent_dir, dronesID[0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    file_out = os.path.join(results_dir,f"video_information.avi")
    out = cv2.VideoWriter(file_out, fourcc, 1.0, (2000,2000))

    for positionIdx, position in enumerate(wayPointsID):
        print(f"{4*' '}[POSITION]: position_{positionIdx}")

        current_detected_dir = os.path.join(detected_dir, dronesID[0], f"position_{positionIdx}")
        time_steps = os.listdir(current_detected_dir)

        for time,time_step in enumerate(time_steps):

            current_detected_dir = os.path.join(detected_dir, dronesID[0], f"position_{positionIdx}")
            raw_image_path1 = os.path.join(current_detected_dir, f"detected_time_{time}.png")

            current_detected_dir = os.path.join(detected_dir, dronesID[1], f"position_{positionIdx}")
            raw_image_path2 = os.path.join(current_detected_dir, f"detected_time_{time}.png")

            frame_raw1 = cv2.imread(raw_image_path1)
            frame_raw1_resized = cv2.resize(frame_raw1,(1000,1000))
            frame_raw2 = cv2.imread(raw_image_path2)
            frame_raw2_resized = cv2.resize(frame_raw2,(1000,1000))

            frame_top = np.concatenate((frame_raw1_resized,frame_raw2_resized), axis=1)
            # h, w, _ = frame_top.shape
            # print(f" frame_top width:{w} -- height:{h}")

            plt_KPI = getKPIpng(positionIdx)
            frame_kpi = cv2.imread(plt_KPI)
            frame_kpi_resized = cv2.resize(frame_kpi,(2000,1000))

            # frame = frame_top + plt_KPI
            frame_top = np.concatenate((frame_top,frame_kpi_resized), axis=0)

            out.write(frame_top)

    out.release()
