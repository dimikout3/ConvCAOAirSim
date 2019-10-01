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
        # print(f"{2*' '}with orientation (pitch:{camPitch:.2f}, yaw:{camYaw:.2f})")
        # print(f"{2*' '}Relative orientation (pitch:{relativePitch:.2f}, yaw:{relativeYaw:.2f})")

        # randomPointsSize = 100*100
        # points = np.random.randint(width,size=(2,randomPointsSize))
        # TODO: get the field of view from camInfo()
        FoV = (np.pi/2)

        points = [[],[]]
        labels = []
        for key, val in detections.items():
            for x, y, label in val:
                points[0].append(int(x))
                points[1].append(int(y))
                labels.append(label)


        points = np.array(points)

        pixelPitch = ((points[0,:]-halfHeight)/halfHeight) * (FoV/2)
        pixelYaw = ((points[1,:]-halfWidth)/halfWidth) * (FoV/2)

        # # inclination (in radiants)
        # theta = -(np.pi/2) + relativePitch + pixelPitch
        # # turn
        # phi = relativeYaw + pixelYaw
        # # inclination (in radiants)
        theta = +(np.pi/2) - pixelPitch + camPitch
        # turn
        phi = pixelYaw + camYaw

        r = imageReshaped[ points[0,:] , points[1,:] ]

        d_x = r*np.sin(theta)*np.cos(phi)
        d_y = r*np.sin(theta)*np.sin(phi)
        d_z = r*np.cos(theta)

        print(f"\n average of Z axis:{np.average(d_z)}")

        return (vehX+d_x, vehY+d_y, vehZ+d_z, labels)


def detectObjects(detector, responses, ctrl, subdir=None):

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

    detections = detector.detect(img_rgb, display=False, save=subdir)
    # print(detections)
    abs_x, abs_y, abs_z, labels = Estimator(depthResponse, detections, ctrl)
    return (abs_x, abs_y, abs_z, labels)


def monitor(droneList, posInd, timeInterval = 1, totalTime = 3):

    print(f"[MONITORING] position {posInd}")

    detector = yoloDetector.yoloDetector()

    c = ['r','b']
    for timeStep in tqdm(range(0,totalTime,timeInterval)):

        absoluteCoordinates = []

        for ctrl in controllers:

            ctrl.updateState(posInd, timeStep)
            responses = ctrl.getImages(save_raw=True)
            detections = ctrl.detectObjects(detector, save_detected=True)

            print(detections)
            # absoluteCoordinates.append(detectObjects(detector, responses, ctrl, detectedDir))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # for i,(x,y,z,_) in enumerate(absoluteCoordinates):
        #     ax.scatter(x, y, z, c=c[i])
        #
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        #
        # plt.show()
        # plt.close()

        # for i,(x,y,z, labels) in enumerate(absoluteCoordinates):
        #     for ind, z_val in enumerate(z):
        #         if np.abs(z_val)>15:
        #             continue
        #         else:
        #             plt.scatter(x[ind], y[ind], c=c[i])
        #             plt.annotate(labels[ind],(x[ind],y[ind]))
        #     # for ind, txt in enumerate(labels):
        #     #     plt.annotate(txt,(x[ind],y[ind]))
        #
        # plt.xlabel('X Label')
        # plt.xlim(0,100)
        # plt.ylim(0,100)
        # plt.ylabel('Y Label')
        #
        # # plt.show()
        # plt.savefig(os.path.join(parentDetect, f"Aggregated_pos_{posInd}_time_{timeStep}.png"))
        # plt.close()

        time.sleep(timeInterval)

# path expressed as x, y, z and speed
PATH = {"Drone1":[(10,0,-10,5), (30,0,-10,5),(50,0,-10,5)],
        "Drone2":[(0,10,-10,5), (0,30,-10,5),(0,50,-10,5)],
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
tasks = []
for ctrl in controllers:
    t = ctrl.takeOff()
    tasks.append(t)

for t in tasks: t.join()

for positionIdx in range(0,wayPointsSize):
    # airsim.wait_key(f"\nPress any key to move drones to position {positionIdx}")
    tasks = []
    for ctrl in controllers:
        x, y, z, speed = PATH[ctrl.getName()].pop(0)
        print(f"{2*' '}[MOVING] {ctrl.getName()} to ({x}, {y}, {z}) at {speed} m/s")
        # client.moveToPositionAsync(x, y, z, speed, vehicle_name=drone).join()
        t = ctrl.moveToPostion(x,y,z,speed)
        tasks.append(t)

    for t in tasks: t.join()

    # print(f"{2*' '}[WAITING] drones to reach their positions")
    # TODO: why this fail ? (check inheritance)
    # client.waitOnLastTask()
    # time.sleep(10)
    monitor(dronesID, positionIdx)

print("\n[RESETING] to original state ....")
for ctrl in controllers: ctrl.quit()
client.reset()
