import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import tqdm


if __name__ == "__main__":

    parent_dir = os.path.join(os.getcwd(),"..", "swarm_raw_output")
    detected_dir = os.path.join(os.getcwd(),"..", "swarm_detected")

    results_dir = os.path.join(os.getcwd(),"..","results")
    costJ_dir = os.path.join(results_dir,"costJ")

    dronesID = os.listdir(detected_dir)
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    file_out = os.path.join(results_dir,f"videoCostJ.avi")
    out = cv2.VideoWriter(file_out, fourcc, .5, (2000,2000))

    for positionIdx, position in enumerate(tqdm(wayPointsID)):
        # print(f"{4*' '}[POSITION]: position_{positionIdx}")

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

            file_score = os.path.join(costJ_dir, f"costJ_time_{positionIdx+time}.png")
            frame_score = cv2.imread(file_score)
            frame_score_resized = cv2.resize(frame_score,(2000,1000))

            # file_similarity = os.path.join(similarity_dir, f"similarity_time_{positionIdx+time}.png")
            # frame_similarity = cv2.imread(file_similarity)
            # frame_similarity_resized = cv2.resize(frame_similarity,(1000,1000))
            #
            # frame_bot = np.concatenate((frame_score_resized,frame_similarity_resized), axis=1)

            frame = np.concatenate((frame_top,frame_score_resized), axis=0)
            out.write(frame)

    out.release()
