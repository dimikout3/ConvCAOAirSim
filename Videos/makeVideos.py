import sys
import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm

import sceneVideo
import optparse
import heatVideo
import pointCloud
import combinedPCD


""" Launches all the videos to be generated"""


def get_options():

    optParser = optparse.OptionParser()
    optParser.add_option("--scene",action="store_true", default=False, dest="scene", help="UAVs scene image going to be generated")
    optParser.add_option("--heat",action="store_true", default=False, dest="heat", help="UAVs depth image going to be generated")
    optParser.add_option("--pointCloud",action="store_true", default=False, dest="pointCloud", help="UAVs 3D image going to be generated")
    optParser.add_option("--combinedPCD",action="store_true", default=False, dest="combinedPCD", help="UAVs combined 3D image going to be generated")
    options, args = optParser.parse_args()

    return options


if __name__ == "__main__":

    options = get_options()

    simulation_dir = os.path.join(os.getcwd(), "..","results_Objective")

    if options.scene:
        sceneVideo.generate(path=simulation_dir, time=60)

    if options.heat:
        # TODO: check if depth.PNG exists, else generate it (heatGenerated.py)
        heatVideo.generate(path=simulation_dir, time=60)

    if options.pointCloud:
        pointCloud.generate(path=simulation_dir, time=60)

    if options.combinedPCD:
        combinedPCD.generate(path=simulation_dir, time=60)
