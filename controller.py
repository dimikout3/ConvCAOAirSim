# import setup_path
import airsim
import os
import cv2
import numpy as np
import time
import pickle
import utilities.utils as utils
import matplotlib.pyplot as plt

DEBUG_GEOFENCE = False
DEBUG_RANDOMZ = True
WEIGHTS = {"cars":1.0, "persons":0.5 , "trafficLights":2.0}

class controller:

    def __init__(self, clientIn, droneName):

        self.client = clientIn
        self.name = droneName

        self.scoreDetections = []
        self.detectionsInfo = []
        self.detectionsCoordinates = []
        self.pointCloud = []

        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

        self.state = self.getState()
        self.cameraInfo = self.getCameraInfo()
        self.stateList = [[self.state,self.cameraInfo]]

        # initial contibution is 0.0
        # how much vehicles currrent movement affected the cost Function (delta)
        self.contribution = [0.0]

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

        return responses


    def getPointCloud(self, x=50, y=50):

        randomPointsSize = x*y

        height, width, _ = self.imageScene.shape
        halfWidth = width/2
        halfHeight= height/2

        r = np.random.uniform(0,halfHeight,randomPointsSize)
        thetas = np.random.uniform(0,2*np.pi,randomPointsSize)

        pointsH = r*np.sin(thetas)
        pointsW = r*np.cos(thetas)

        centerH = int(halfHeight)
        centerW = int(halfWidth)

        pointsH = centerH + pointsH.astype(int)
        pointsW = centerW + pointsW.astype(int)

        colors = self.imageScene[pointsH, pointsW]

        xRelative, yRelative, zRelative, colors = utils.to3D(pointsW, pointsH,
                                          self.cameraInfo, self.imageDepthCamera,
                                          color = colors)
        x, y, z = utils.to_absolute_coordinates(xRelative, yRelative, zRelative,
                                                self.cameraInfo)


        # utils.plot3dColor(x,y,z,colors,show=True)

        self.pointCloud.append([x,y,z,colors])

        return x,y,z,colors
    # TODO: getDepth() -> from camera "0", similar to rgb


    def getPointCloudList(self, index=-1):

        x,y,z,colors = self.pointCloud[index]

        return x,y,z,colors


    def appendContribution(self, contrib):

        self.contribution.append(contrib)


    def getDepthFront(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        self.imageDepthFront = responses[0]


    def getDepthImage(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        self.imageDepthCamera = responses[0]

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


    def setGeoFence(self, x=0, y=0, z=-10, r=20):
        """Applying geo fence as a sphere (x,y,z) spheres center, r-> diameter"""

        self.fenceX = x
        self.fenceY = y
        self.fenceZ = z
        self.fenceR = r


    def applyGeoFence(self):

        if DEBUG_GEOFENCE:
            print(f"\n-- Checking Geo Fence for {self.getName()}")

        droneX = self.state.kinematics_estimated.position.x_val
        droneY = self.state.kinematics_estimated.position.y_val
        droneZ = self.state.kinematics_estimated.position.z_val

        dist = np.sqrt( (self.fenceX - droneX)**2 + (self.fenceY - droneY)**2
                    + (self.fenceZ - droneZ)**2)

        if DEBUG_GEOFENCE:
            print(f"   Sphere (x:{self.fenceX:.2f}, y:{self.fenceY:.2f}, z:{self.fenceZ:.2f}, R:{self.fenceR:.2f}) ")
            print(f"   {self.getName()} (x:{droneX:.2f}, y:{droneY:.2f}, z:{droneZ:.2f}, dist:{dist:.2f}) ")

        if dist >= self.fenceR:

            # OA -> X-Y origin to drone position
            # OC -> X-Y origin to sphere center
            # AC -> Drone to sphere

            acX = self.fenceX - droneX
            acY = self.fenceY - droneY

            acYaw = np.arctan2(acY, acX)

            _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)

            # turn -> how much the drone must turn in order to orient itself towards the speher's  center
            turn = acYaw - currentYaw

            if DEBUG_GEOFENCE:
                print(f"   Applying Geo Fence yaw:{np.degrees(currentYaw):.2f} acYaw:{np.degrees(acYaw):.2f} turn:{np.degrees(turn):.2f}")

            # XXX: airsim function rotateToYaw seems not to perform correctly ...
            self.client.rotateByYawRateAsync(np.degrees(turn),1,vehicle_name=self.name).join()
            self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()

            self.state = self.getState()


    def randomMoveZ(self):

        minThreshold = 5
        axiZ = -15
        pixeSquare = 150
        speedScalar = 3

        changeYaw = 0.0

        tries = 0

        while True:

            tries += 1

            self.applyGeoFence()

            _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)

            # restint the seed for more random results (if it is not included it yawRandom
            # is not random anymore ... )
            np.random.seed()

            if tries >=5:
                # if stuck drone should rotatte only towards one side (clockwise)
                # otherwise it will constantly orient towards the obstackel (mean value
                # of many different randomYaw will be 0
                randomYaw = np.random.uniform(0, np.pi/4)
            else:
                randomYaw = np.random.uniform(-np.pi/4, np.pi/4)

            self.client.rotateByYawRateAsync(np.degrees(randomYaw),1,vehicle_name=self.name).join()
            self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()

            self.getDepthFront()
            imageDepth = airsim.list_to_2d_float_array(self.imageDepthFront.image_data_float,
                                                       self.imageDepthFront.width,
                                                       self.imageDepthFront.height)

            midW = self.imageDepthFront.width/2
            midH = self.imageDepthFront.width/2
            imageDepthTarget = imageDepth[int(midW-pixeSquare):int(midW+pixeSquare),
                                          int(midH-pixeSquare):int(midH+pixeSquare)]

            current = np.min(imageDepthTarget)

            travelTime = np.random.uniform(5.,10.)

            if (current - travelTime*speedScalar)<minThreshold:
                travelTime = (current - (minThreshold-2))/speedScalar
            if travelTime<5.0: travelTime = 5.

            if DEBUG_RANDOMZ:
                print(f"\n[DEBUG][RANDOM_MOVE_Z] ----- {self.name} -----")
                print(f"[DEBUG][RANDOM_MOVE_Z] currentYaw:{np.degrees(currentYaw):.4f} [deg]")
                print(f"[DEBUG][RANDOM_MOVE_Z] randomYaw:{np.degrees(randomYaw):.4f} [deg]")
                print(f"[DEBUG][RANDOM_MOVE_Z] available current distance:{current:.2f} [m]")
                print(f"[DEBUG][RANDOM_MOVE_Z] suggested travel time:{travelTime:.2f} [sec]")
                print(f"[DEBUG][RANDOM_MOVE_Z] expected travel dist:{travelTime*speedScalar:.2f} [m]")

            if (current - travelTime*speedScalar)>minThreshold:

                anglesSpeed = currentYaw + randomYaw
                if DEBUG_RANDOMZ:
                    print(f"[DEBUG][RANDOM_MOVE_Z] anglesSpeed:{np.degrees(anglesSpeed):.4f} [deg]")

                vx = np.cos(anglesSpeed)
                vy = np.sin(anglesSpeed)
                if DEBUG_RANDOMZ:
                    print(f"[DEBUG][RANDOM_MOVE_Z] has Vx:{vx*speedScalar:.3f} Vy:{vy*speedScalar:.3f} [m/s]")

                task = self.client.moveByVelocityZAsync(speedScalar*vx, speedScalar*vy, axiZ, travelTime,
                                            airsim.DrivetrainType.ForwardOnly,
                                            airsim.YawMode(False, 0),
                                            vehicle_name=self.name).join()
                self.stabilize().join()

                break
            elif tries >= 10:
                # if drone is stucked in tree or building send it to initial position
                # TODO: keep track of position and send it to previous position (it can surely access it).
                pose = self.client.simGetVehiclePose()
                pose.position.x_val = 0
                pose.position.y_val = 0
                pose.position.z_val = -15

                # task = self.client.moveToPositionAsync(0,0,-10,3)
                # QUESTION: Ignore collision -> what it does ? (now its True)
                task = self.client.simSetVehiclePose(pose, True, self.getName())
                self.stabilize().join()

                print(f"[WARNING] {self.getName()} reached max tries {tries} ...")
                break
            else:
                print(f"[WARNING] {self.name} changing yaw due to imminent collision ...")

            self.state = self.getState()

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
        self.detectionsInfo.append(detectionsInfo)
        self.detectionsCoordinates.append(detectionsCoordinates)
        self.stateList.append([self.getState(), self.getCameraInfo()])

        # print(f"info: {detectionsInfo} \ncoordinates: {detectionsCoordinates}")
        return detectionsInfo, detectionsCoordinates


    def getScore(self, index = None):

        if index == None:
            # if no index is specified the whole list will be returned
            return self.scoreDetections
        else:
            # usually used with index -1
            return self.scoreDetections[index]


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
                                        self.getName(), f"score_detections_{self.name}.pickle")
        pickle.dump(self.scoreDetections,open(score_detections_file,"wb"))
        score_detections_file = os.path.join(os.getcwd(), "results","information",
                                        f"score_detections_{self.name}.pickle")
        pickle.dump(self.scoreDetections,open(score_detections_file,"wb"))

        detections_file = os.path.join(os.getcwd(), "swarm_raw_output",
                                       self.getName(), f"detectionsInfo_{self.name}.pickle")
        pickle.dump(self.detectionsInfo,open(detections_file,"wb"))
        detections_file = os.path.join(os.getcwd(), "results","detected_objects",
                                       f"detectionsInfo_{self.name}.pickle")
        pickle.dump(self.detectionsInfo,open(detections_file,"wb"))

        detections_file = os.path.join(os.getcwd(), "swarm_raw_output",
                                       self.getName(), f"detectionsCoordinates_{self.name}.pickle")
        pickle.dump(self.detectionsCoordinates,open(detections_file,"wb"))
        detections_file = os.path.join(os.getcwd(), "results","detected_objects",
                                       f"detectionsCoordinates_{self.name}.pickle")
        pickle.dump(self.detectionsCoordinates,open(detections_file,"wb"))

        state_file = os.path.join(os.getcwd(), "swarm_raw_output",
                                  self.getName(), f"state_{self.name}.pickle")
        pickle.dump(self.stateList,open(state_file,"wb"))

        pointCloud_file = os.path.join(os.getcwd(), "swarm_raw_output",
                                  self.getName(), f"pointCloud_{self.name}.pickle")
        pickle.dump(self.pointCloud,open(pointCloud_file,"wb"))

        contribution_file = os.path.join(os.getcwd(), "swarm_raw_output",
                                  self.getName(), f"contribution_{self.name}.pickle")
        pickle.dump(self.contribution,open(contribution_file,"wb"))
