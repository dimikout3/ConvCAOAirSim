# import setup_path
import airsim
import os
import cv2
import numpy as np
import time
import pickle
import utilities.utils as utils
import matplotlib.pyplot as plt

WEIGHTS = {"cars":1.0, "persons":0.5 , "trafficLights":2.0}

class controller:

    def __init__(self, clientIn, droneName):

        self.client = clientIn
        self.name = droneName
        self.scoreDetections = []

        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

        self.state = self.getState()
        self.cameraInfo = self.getCameraInfo()

        self.parentRaw = os.path.join(os.getcwd(), "swarm_raw_output")
        try:
            os.makedirs(self.parentRaw)
        except OSError:
            if not os.path.isdir(self.parentRaw):
                raise

        self.parentDetect = os.path.join(os.getcwd(), "swarm_detected")
        try:
            os.makedirs(self.parentDetect)
        except OSError:
            if not os.path.isdir(self.parentDetect):
                raise


    def takeOff(self):

        return self.client.takeoffAsync(vehicle_name = self.name)


    def moveToPostion(self, x, y, z, speed):

        return self.client.moveToPositionAsync(x,y,z,speed,vehicle_name=self.name)


    def setCameraOrientation(self, cam_yaw, cam_pitch, cam_roll):

        self.client.simSetCameraOrientation("0",
                                            airsim.to_quaternion(cam_yaw, cam_pitch, cam_roll),
                                            vehicle_name = self.name)


    def getName(self):

        return self.name


    def getImages(self, save_raw=None):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True),  #depth visualization image
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        if save_raw != None:

            filenameDepth = os.path.join(self.raw_dir, f"depth_time_{self.timeStep}" )
            airsim.write_pfm(os.path.normpath(filenameDepth + '.pfm'), airsim.get_pfm_array(responses[0]))

            filenameScene = os.path.join(self.raw_dir, f"scene_time_{self.timeStep}" )
            img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8) #get numpy array
            img_rgb = img1d.reshape(responses[1].height, responses[1].width, 3) #reshape array to 3 channel image array H X W X 3
            cv2.imwrite(os.path.normpath(filenameScene + '.png'), img_rgb) # write to png
            self.imageScene = img_rgb

            # TODO: this slows down our programm, consider creating a list of states and save it only one time at the end (before quiting...)
            filenameState = os.path.join(self.raw_dir, f"state_time_{self.timeStep}" )
            pickle.dump([self.state,self.cameraInfo], open(os.path.normpath(filenameState + '.pickle'),"wb"))

        return responses


    # TODO: getDepth() -> from camera "0", similar to rgb

    def getDepthFront(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        self.imageDepthFront = responses[0]

    def getDepthImage(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        self.imageDepthImage = responses[0]

        return responses[0]


    def stabilize(self):

        task = self.client.moveByVelocityAsync(0,0,0,4,vehicle_name=self.name)

        return task


    def moveToZ(self, targetZ, speedClim=3.0):

        task = self.client.moveToZAsync(targetZ,speedClim,vehicle_name=self.name)

        return task


    def rotateToYaw(self, yaw):
        task = self.client.rotateToYawAsync(yaw,vehicle_name=self.name)
        return task

    def randomMoveZ(self):

        minThreshold = 20
        axiZ = -10
        pixeSquare = 150
        speedScalar = 2

        _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)
        # currentYaw = np.degrees(currentYaw)
        changeYaw = 0.0

        tries = 0

        while True:

            tries += 1

            # changeYaw = np.random.ranf()*np.pi
            if ("1" in self.name) or ("2" in self.name):
                randomYaw = np.random.uniform(0,np.pi/6)
            else:
                randomYaw = np.random.uniform(-np.pi/6,0)
            # changeYaw = np.random.uniform(0,180)
            # self.client.rotateToYawAsync(yawRandom,vehicle_name=self.name).join()
            self.client.rotateByYawRateAsync(np.degrees(randomYaw),1,vehicle_name=self.name).join()
            # .join() seems not to work correctly when using python scripts (works on interpreter)
            self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()

            self.getDepthFront()
            imageDepth = airsim.list_to_2d_float_array(self.imageDepthFront.image_data_float,
                                                       self.imageDepthFront.width,
                                                       self.imageDepthFront.height)

            # midVertical = np.vsplit(imageDepth,3)[1]
            # mid = np.hsplit(midVertical,3)[1]
            midW = self.imageDepthFront.width/2
            midH = self.imageDepthFront.width/2
            imageDepthTarget = imageDepth[int(midW-pixeSquare):int(midW+pixeSquare),
                                          int(midH-pixeSquare):int(midH+pixeSquare)]

            changeYaw += randomYaw
            current = np.min(imageDepthTarget)
            print(f"\n {self.name} currentYaw:{np.degrees(currentYaw)}")
            print(f"{self.name} randomYaw:{np.degrees(randomYaw)}")
            print(f"{self.name} changeYaw:{np.degrees(changeYaw)}")
            if current>minThreshold:

                anglesSpeed = changeYaw + currentYaw
                print(f"{self.name} anglesSpeed:{np.degrees(anglesSpeed)}")

                vx = np.cos(anglesSpeed)
                vy = np.sin(anglesSpeed)
                print(f"{self.name} has Vx:{vx} Vy:{vy}")

                task = self.client.moveByVelocityZAsync(speedScalar*vx, speedScalar*vy,axiZ, 5,
                                            airsim.DrivetrainType.ForwardOnly,
                                            airsim.YawMode(False, 0),
                                            vehicle_name=self.name)

                break
            elif tries >= 10:
                # if drone is stucked in tree or building send it to initial position
                # TODO: keep track of position and send it to previous position (it can surely access it).
                task = self.client.moveToPositionAsync(0,0,-10,3)
            else:
                print(f"[WARNING] {self.name} changing yaw due to imminent collision ...")

        return task


    def updateState(self, posIdx, timeStep):

        self.state = self.getState()
        self.cameraInfo = self.getCameraInfo()

        self.posIdx = posIdx
        self.timeStep = timeStep

        self.raw_dir = os.path.join(self.parentRaw, self.name, f"position_{self.posIdx}")
        if not os.path.isdir(self.raw_dir):
            os.makedirs(self.raw_dir)

        self.detected_dir = os.path.join(self.parentDetect, self.name, f"position_{self.posIdx}")
        if not os.path.isdir(self.detected_dir):
            os.makedirs(self.detected_dir)


    def detectObjects(self, detector, save_detected=False):

        detected_file_name = None
        if save_detected:
            detected_file_name = os.path.join(self.detected_dir,
                                              f"detected_time_{self.timeStep}.png")

        detections = detector.detect(self.imageScene, display=False, save=detected_file_name)

        # detections = {'cars':[(pixel_x,pixel_y,detecions_id,confidece),(pixel_x,pixel_y,detecions_id,confidece), ...]}
        # val[4] -> confidence of each detections
        # score = sum([val[4]*WEIGHTS[key] for key,val in detections.items()])
        score = 0.0
        pixelX = []
        pixelY = []
        for detectionClass, objects in detections.items():
            for object in objects:
                # object = (pixel_x,pixel_y,detecions_id,confidece)
                score += object[3]*WEIGHTS[detectionClass]
                pixelX.append(object[0])
                pixelY.append(object[1])

        depthImage = self.getDepthImage()

        xRelative, yRelative, zRelative = utils.to3D(pixelX, pixelY,
                                          self.cameraInfo, depthImage)
        x, y, z = utils.to_absolute_coordinates(xRelative, yRelative, zRelative,
                                                self.cameraInfo)

        detectionsCoordinates = np.stack((x,y,z), axis=1)

        detectionsInfo = []
        for detectionClass, objects in detections.items():
            for object in objects:
                # x,y,z are full lista, use x[i], y[i], z[i]
                detectionsInfo.append([object[2], object[3]])

        self.scoreDetections.append(score)

        # print(f"info: {detectionsInfo} \ncoordinates: {detectionsCoordinates}")
        return detectionsInfo, detectionsCoordinates


    def getPose(self):
        return self.client.simGetVehiclePose(vehicle_name=self.name)


    def getState(self):
        return self.client.getMultirotorState(vehicle_name=self.name)


    def getCameraInfo(self, cam="0"):
        return self.client.simGetCameraInfo(cam,vehicle_name=self.name)


    def quit(self):

        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)

        # TODO: possible np.save() instead of pickle ...
        score_detections_file = os.path.join(os.getcwd(), "swarm_raw_output",
                                             f"score_detections_{self.name}.pickle")

        pickle.dump(self.scoreDetections,open(score_detections_file,"wb"))
