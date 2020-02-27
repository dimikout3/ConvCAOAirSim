import sys
import os
import numpy as np
import pickle
import cv2

"""  """

def generate(path="", time=60):

    parent_dir = os.path.join(path, "swarm_raw_output")
    detected_dir = os.path.join(path, "swarm_detected")

    dronesID = os.listdir(parent_dir)
    dronesID = [drone for drone in dronesID if drone!="GlobalHawk"]
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    depth_dir = os.path.join(path, "swarm_raw_output",f"{dronesID[0]}", f"position_{0}", f"depth.png")
    image_reference= cv2.imread(depth_dir)
    height, width, _ = image_reference.shape
    print(f"width={width} height={height}")

    for drone in dronesID:

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = os.path.join(path, "swarm_raw_output", f"{drone}", f"heat_{drone}.avi")
        out = cv2.VideoWriter(video_out, fourcc, len(wayPointsID)/time, (width,height))

        for posIndex in range(len(wayPointsID)):

            depth_dir = os.path.join(path, "swarm_raw_output",f"{drone}", f"position_{posIndex}", f"depth.png")
            frame = cv2.imread(depth_dir)
            out.write(frame)

        out.release()
