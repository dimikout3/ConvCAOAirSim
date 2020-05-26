import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import json
import open3d as o3d

class drone:

    def __init__(self, name="UAV_NO_NAME", maxView=50., pose=[0,0,0]):

        self.name = name

        self.maxView = maxView

        self.pose = pose
