import setup_path
import airsim

import numpy as np
import os
import cv2
import time
from tqdm import tqdm
from controller import controller
import yoloDetector

CAM_YAW = -0.5
CAM_PITCH = 0.
CAM_ROOL = 0.



def detectObjects(detector, responses):

    for idx, response in enumerate(responses):

        if response.pixels_as_float:
            # print("PIXEL AS FLOAT")
            airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        elif response.compress: #png format
            # print("PIXEL Compressed [second option]")
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        else: #uncompressed array
            # print("IN UNCOMPRESS ARRAY [third option]")
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) #get numpy array
            img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3

            detector.detect(img_rgb)


def saveImage(subDir, timeStep, responses):

    for idx, response in enumerate(responses):

        filename = os.path.join(subDir, f"type_{idx}_time_{timeStep}" )

        if response.pixels_as_float:
            airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        elif response.compress: #png format
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        else: #uncompressed array
            # img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) #get numpy array
            img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
            cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png


def monitor(droneList, posInd, parentDir, timeInterval = 1, totalTime = 3):

    print(f"[MONITORING] position {posInd}")

    detector = yoloDetector.yoloDetector()

    for timeStep in tqdm(range(0,totalTime,timeInterval)):

        for ctrl in controllers:

            subDir = os.path.join(parentDir, ctrl.getName(), f"position_{posInd}")

            if not os.path.isdir(subDir):
                os.makedirs(subDir)

            # responses = client.simGetImages([
            #     # airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
            #     airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)], vehicle_name=drone)  #scene vision image in uncompressed RGB array
            responses = ctrl.getImages()

            # saveImage(subDir, timeStep, responses)
            detectObjects(detector, responses)

        time.sleep(timeInterval)

# path expressed as x, y, z and speed
PATH = {"Drone1":[(10,0,-10,5), (30,0,-10,5),(50,0,-10,5)],
        "Drone2":[(10,10,-10,5), (10,20,-10,5),(10,30,-10,5)],
        }

dronesID = list(PATH.keys())
wayPointsSize = len(PATH[dronesID[0]])
print(f"Detected {dronesID} with {wayPointsSize} positions")

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

controllers = []
for drone in dronesID:
    controllers.append(controller(client, drone))

# Setting Camera Orientation
for ctrl in controllers: ctrl.setCameraOrientation(CAM_YAW, CAM_PITCH, CAM_ROOL)

# airsim.wait_key('Press any key to takeoff all drones')
print("Taking off all drones")
for ctrl in controllers: ctrl.takeOff()

parentDir = os.path.join(os.getcwd(), "swarm_raw_output")
try:
    os.makedirs(parentDir)
except OSError:
    if not os.path.isdir(parentDir):
        raise

for positionIdx in range(0,wayPointsSize):
    # airsim.wait_key(f"\nPress any key to move drones to position {positionIdx}")
    for ctrl in controllers:
        x, y, z, speed = PATH[ctrl.getName()].pop(0)
        print(f"{2*' '}[MOVING] {ctrl.getName()} to ({x}, {y}, {z}) at {speed} m/s")
        # client.moveToPositionAsync(x, y, z, speed, vehicle_name=drone).join()
        ctrl.moveToPostion(x,y,z,speed)
    print(f"{2*' '}[WAITING] drones to reach their positions")
    # TODO: why this fail ? (check inheritance)
    # client.waitOnLastTask()
    time.sleep(10)
    monitor(dronesID, positionIdx, parentDir)

airsim.wait_key('Press any key to reset to original state')


for ctrl in controllers: ctrl.quit()
client.reset()
