import setup_path
import airsim

import numpy as np
import os
import cv2
import time
from tqdm import tqdm
from controller import controller
import yoloDetector
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CAM_YAW = -0.5
CAM_PITCH = 0.
CAM_ROOL = 0.

def Estimator(response, detections, ctrl):

        width = response.width
        height = response.height
        halfWidth = width/2
        halfHeight= height/2

        image_numpy = np.array(response.image_data_float)
        imageReshaped = np.reshape(image_numpy, (height,width))

        # vehX = ctrl.getPose().position.x_val
        # vehY = ctrl.getPose().position.y_val
        # vehZ = ctrl.getPose().position.z_val

        vehX = ctrl.getState().kinematics_estimated.position.x_val
        vehY = ctrl.getState().kinematics_estimated.position.y_val
        vehZ = ctrl.getState().kinematics_estimated.position.z_val
        vehPitch, _, vehYaw = airsim.to_eularian_angles(ctrl.getPose().orientation)

        camX = response.camera_position.x_val
        camY = response.camera_position.y_val
        camZ = response.camera_position.z_val
        camPitch, _, camYaw = airsim.to_eularian_angles(response.camera_orientation)

        relativePitch = camPitch-vehPitch
        relativeYaw = camYaw-vehYaw

        print(f"\n{ctrl.getName()} is located at (x:{vehX:.2f}, y:{vehY:.2f}, z:{vehZ:.2f})")
        print(f"{2*' '}with orientation (pitch:{vehPitch:.2f}, yaw:{vehYaw:.2f})")
        print(f"{2*' '}Its camera is located at (x:{camX:.2f}, y:{camY:.2f}, z:{camZ:.2f})")
        print(f"{2*' '}with orientation (pitch:{camPitch:.2f}, yaw:{camYaw:.2f})")
        print(f"{2*' '}Relative orientation (pitch:{relativePitch:.2f}, yaw:{relativeYaw:.2f})")

        randomPointsSize = 100*100
        points = np.random.randint(width,size=(2,randomPointsSize))
        # TODO: get the field of view from camInfo()
        FoV = (np.pi/2)

        # points = [[],[]]
        # for key, val in detections.items():
        #     for x,y in val:
        #         points[0].append(int(x))
        #         points[1].append(int(y))
        #
        # points = np.array(points)

        pixelPitch = ((points[0,:]-halfHeight)/halfHeight) * (FoV/2)
        pixelYaw = ((points[1,:]-halfWidth)/halfWidth) * (FoV/2)

        # inclination (in radiants)
        theta = -(np.pi/2) + relativePitch + pixelPitch
        # turn
        phi = relativeYaw + pixelYaw

        r = imageReshaped[ points[0,:] , points[1,:] ]

        d_x = r*np.sin(theta)*np.cos(phi)
        d_y = r*np.sin(theta)*np.sin(phi)
        d_z = r*np.cos(theta)

        return (vehX+d_x, vehY+d_y, vehZ+d_z)


def detectObjects(detector, responses, ctrl):

    for idx, response in enumerate(responses):

        if response.pixels_as_float:
            depthResponse = response
        elif response.compress: #png format
            # print("PIXEL Compressed [second option]")
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        else: #uncompressed array
            # print("IN UNCOMPRESS ARRAY [third option]")
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) #get numpy array
            img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3

    detections = detector.detect(img_rgb, display=False)
    # print(detections)
    abs_x, abs_y, abs_z = Estimator(depthResponse, detections, ctrl)
    return (abs_x, abs_y, abs_z)

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


def monitor(droneList, posInd, parentDir, timeInterval = 1, totalTime = 1):

    print(f"[MONITORING] position {posInd}")

    detector = yoloDetector.yoloDetector()

    c = ['r','b']
    for timeStep in tqdm(range(0,totalTime,timeInterval)):

        absoluteCoordinates = []

        for ctrl in controllers:

            subDir = os.path.join(parentDir, ctrl.getName(), f"position_{posInd}")

            if not os.path.isdir(subDir):
                os.makedirs(subDir)

            responses = ctrl.getImages()

            # saveImage(subDir, timeStep, responses)
            absoluteCoordinates.append(detectObjects(detector, responses, ctrl))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i,(x,y,z) in enumerate(absoluteCoordinates):
            ax.scatter(x, y, z, c=c[i])
        plt.show()
        plt.close()

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

print("\n[RESETING] to original state ....")
for ctrl in controllers: ctrl.quit()
client.reset()
