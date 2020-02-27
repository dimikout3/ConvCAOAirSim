import sys
import os
import numpy as np
import pickle
import cv2

"""  """
WIDTH = 1000
HEIGHT = 1000

def generate(path="", time=60):

    parent_dir = os.path.join(path, "swarm_raw_output")
    detected_dir = os.path.join(path, "swarm_detected")

    dronesID = os.listdir(parent_dir)
    dronesID = [drone for drone in dronesID if drone!="GlobalHawk"]
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    for drone in dronesID:

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = os.path.join(path, "swarm_raw_output", f"{drone}", f"scene_{drone}.avi")
        out = cv2.VideoWriter(video_out, fourcc, len(wayPointsID)/time, (WIDTH,HEIGHT))

        for posIndex in range(len(wayPointsID)):

            scene_dir = os.path.join(path, "swarm_raw_output",f"{drone}", f"position_{posIndex}", f"scene_time_0.png")
            frame = cv2.imread(scene_dir)
            out.write(frame)

        out.release()
