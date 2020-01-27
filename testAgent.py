import setup_path
import airsim
import numpy as np
import time
from controller import controller
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

optionsIP = 1

OFFSETS = {"Drone1":[0,0,0]}

ip_id = f"127.0.0.{optionsIP}"
client = airsim.MultirotorClient(ip = ip_id)
client.confirmConnection()

drone = "Drone1"

ctrl = controller(client, drone, OFFSETS[drone], ip=optionsIP, wayPointsSize=100, estimatorWindow=100)

# points[points[:,2]<1]
