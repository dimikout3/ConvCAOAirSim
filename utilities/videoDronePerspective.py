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

if __name__ == "__main__":

    parent_dir = os.path.join(os.getcwd(),"..", "swarm_raw_output")
    detected_dir = os.path.join(os.getcwd(),"..", "swarm_detected")

    results_dir = os.path.join(os.getcwd(),"..", "results")
    try:
        os.makedirs(results_dir)
    except OSError:
        if not os.path.isdir(results_dir):
            raise

    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(parent_dir, dronesID[0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for droneIdx, drone in enumerate(dronesID):
        print(f"\n[DRONE]: {drone}")

        file_out = os.path.join(results_dir,f"{drone}_perspective_depth.avi")
        out = cv2.VideoWriter(file_out, fourcc, 1.0, (2048,1024))

        for positionIdx, position in enumerate(wayPointsID):
            print(f"{4*' '}[POSITION]: position_{positionIdx}")

            current_raw_dir = os.path.join(parent_dir, drone, f"position_{positionIdx}")
            current_detected_dir = os.path.join(detected_dir, drone, f"position_{positionIdx}")

            time_steps = os.listdir(current_detected_dir)

            for time,time_step in enumerate(time_steps):

                raw_image_path = os.path.join(current_raw_dir, f"scene_time_{time}.png")
                depth_image_path = os.path.join(current_raw_dir, f"depth_time_{time}.pfm")

                frame_raw = cv2.imread(raw_image_path)
                frame_detected,s = airsim.read_pfm(depth_image_path)

                # cut values above 100 meters
                frame_detected[frame_detected>100] = 100.
                # convert from 0->255 fro greyscale
                frame_detected = (frame_detected/100.) * 255.
                # save it
                file_out = os.path.join(current_raw_dir, f"depth_scene_time_{time}.png")
                cv2.imwrite(file_out,frame_detected)

                frame_detected = cv2.imread(file_out)
                fram_vertical = np.concatenate((frame_raw,frame_detected), axis=1)
                # h, w, _ = fram_vertical.shape
                # print(f" width:{w} -- height:{h}")

                out.write(fram_vertical)

        out.release()
