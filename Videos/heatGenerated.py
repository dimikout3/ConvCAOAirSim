import sys
import os
import numpy as np
import pickle
import airsim
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Parses all the position and generates a heatmap from depth pfm"""

DEPTH_MAX = 75


def saveDepthPNG(data, depth_dir):

    plt.imshow(data, cmap="gist_heat")

    depth_name = os.path.join(depth_dir, f"depth.png")

    plt.grid(False)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(depth_name)
    plt.close()


if __name__ == "__main__":

    simulation_dir = os.path.join(os.getcwd(), "..","results_Objective")

    parent_dir = os.path.join(simulation_dir, "swarm_raw_output")
    detected_dir = os.path.join(simulation_dir, "swarm_detected")

    dronesID = os.listdir(parent_dir)
    dronesID = [drone for drone in dronesID if drone!="GlobalHawk"]
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    for drone in dronesID:

        print(f"=== Woriking on {drone} ===")

        for posIndex, position in enumerate(tqdm(wayPointsID)):

            depth_dir = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}", f"{position}")

            depth_pfm = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}", f"{position}", f"depth_time_0.pfm")
            data, scale = airsim.read_pfm(depth_pfm)

            data[data>DEPTH_MAX] = DEPTH_MAX

            saveDepthPNG(data, depth_dir)
