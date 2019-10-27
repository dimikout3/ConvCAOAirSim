# import setup_path
import airsim
import os
import cv2
import numpy as np
import time
import pickle
import utilities.utils as utils
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

DEBUG_GEOFENCE = False
DEBUG_RANDOMZ = False
DEBUG_MOVE = False
DEBUG_MOVE1DOF = True
WEIGHTS = {"cars":1.0, "persons":1.0 , "trafficLights":1.0}

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

        self.model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                               ('linear', LinearRegression(fit_intercept=False))])

        self.model1DoF = Pipeline([('poly', PolynomialFeatures(degree=2)),
                               ('linear', LinearRegression(fit_intercept=False))])

        self.model2DoF = Pipeline([('poly', PolynomialFeatures(degree=2)),
                               ('linear', LinearRegression(fit_intercept=False))])

        self.estimator = self.model.fit([np.random.uniform(0,1,3)],[[np.random.uniform(0,1)]])

        self.estimator1DoF = self.model1DoF.fit([np.random.uniform(0,1,1)],[[np.random.uniform(0,1)]])

        self.estimator2DoF = self.model2DoF.fit([np.random.uniform(0,1,2)],[[np.random.uniform(0,1)]])

        self.estimations = []
        self.historyData = []

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

        if abs(index) > len(self.pointCloud):
            x,y,z,colors = self.pointCloud[-1]
        else:
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

        #TODO: replace simGetImages with self.getImages -> save depth image
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        self.imageDepthCamera = responses[0]

        return responses[0]


    def stabilize(self):

        task = self.client.moveByVelocityAsync(0,0,0,4,vehicle_name=self.name)

        return task


    def moveToZ(self, targetZ, speedClim=3.0):

        self.altitude = targetZ
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

        self.maxX = x + r
        self.minX = x - r
        self.maxY = y + r
        self.minY = y - r


    def insideGeoFence(self, d=5., c=[]):

        x,y,z = c
        dist = np.sqrt( (self.fenceX-x)**2 + (self.fenceY-y)**2 + (self.fenceZ-z)**2)

        return (dist+d) < self.fenceR


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

                #BUG: huge bug, duration drone will move after the execution of the duration ....
                #TODO: Find a walk around
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
                self.client.simSetVehiclePose(pose, True, self.getName())
                self.stabilize().join()

                print(f"[WARNING] Reached max tries:{tries} ... teleporting to initial position")
                tries = 0
            else:
                print(f"[WARNING] {self.name} changing yaw due to imminent collision ...")

            self.state = self.getState()



    def move1DoF(self, randomPointsSize=50):

        np.random.seed()

        # camera field of view (degrees)
        camFOV = self.cameraInfo.fov
        leftDeg, rightDeg = -camFOV/2 , camFOV/2

        self.state = self.getState()
        _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)

        self.getDepthFront()
        imageDepth = airsim.list_to_2d_float_array(self.imageDepthFront.image_data_float,
                                                   self.imageDepthFront.width,
                                                   self.imageDepthFront.height)

        height, width = self.imageDepthFront.height, self.imageDepthFront.width
        pixel10H = height*0.1
        lowHeight, highHeight = int(height/2-pixel10H), int(height/2+pixel10H)
        wLow, wHigh = int(width*0.1) ,int(width*0.1)

        yawCanditate = np.random.uniform(leftDeg/2, rightDeg/2, randomPointsSize)

        # validPoint = []
        jPoint = []
        for i in range(len(yawCanditate)):

            if (yawCanditate[i]+np.degrees(currentYaw))>180:yawCanditate[i] = 180 - currentYaw
            if (yawCanditate[i]+np.degrees(currentYaw))<-180:yawCanditate[i] = -180 + currentYaw
            jPoint.append(self.estimate1DoF(yawCanditate[i]+np.degrees(currentYaw)))

        self.estimations.append(jPoint)

        tartgetPointIndex = np.argmax(jPoint)

        self.client.rotateByYawRateAsync(yawCanditate[tartgetPointIndex],1,vehicle_name=self.name).join()
        self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()

        if DEBUG_MOVE1DOF:
            print(f"\n[DEBUG][MOVE1DOF] ----- {self.getName()} -----")
            print(f"[DEBUG][MOVE1DOF] yawCanditate: {yawCanditate[tartgetPointIndex]}")
            print(f"[DEBUG][MOVE1DOF] jPoint: {jPoint[tartgetPointIndex]}")
            print(f"[DEBUG][MOVE1DOF] tartgetPointIndex: {tartgetPointIndex}")


    def move(self, randomPointsSize=50, maxTravelTime=15., minDist=5.):

        axiZ = self.altitude

        speedScalar = 2
        np.random.seed()

        # camera field of view (degrees)
        camFOV = self.cameraInfo.fov
        leftDeg, rightDeg = -camFOV/2 , camFOV/2

        maxTries = 12
        availablePosition = False

        for i in range(maxTries):

            self.state = self.getState()
            _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)

            self.getDepthFront()
            imageDepth = airsim.list_to_2d_float_array(self.imageDepthFront.image_data_float,
                                                       self.imageDepthFront.width,
                                                       self.imageDepthFront.height)

            height, width = self.imageDepthFront.height, self.imageDepthFront.width
            pixel10H = height*0.1
            lowHeight, highHeight = int(height/2-pixel10H), int(height/2+pixel10H)
            wLow, wHigh = int(width*0.1) ,int(width*0.1)

            # boundaries should be avoided for collision avoidance (thats what +/- 3 degrees do ...)
            randomOrientation = np.random.uniform(np.radians(leftDeg+3), np.radians(rightDeg-3), randomPointsSize)
            # travelTime = np.random.uniform(0, maxTravelTime, randomPointsSize)
            travelTime = np.random.uniform(0., maxTravelTime, randomPointsSize)
            yawCanditate = np.random.uniform(-30,30,randomPointsSize)

            # validPoint = []
            jPoint = []
            for i in range(randomPointsSize):

                wCenter = int(  width * ( ( np.degrees(randomOrientation[i]) + camFOV/2 ) / camFOV ) )
                if wCenter - wLow < 0: wCenter = wLow + 1
                if wCenter + wHigh> (width-1): wCenter = wHigh + 1
                # print(f"{self.getName()} wCenter:{wCenter}")

                dist = np.min(imageDepth[ (wCenter-wLow) : (wCenter+wHigh), lowHeight:highHeight])

                safeDist = dist>(travelTime[i]*speedScalar + minDist)

                xCurrent = self.state.kinematics_estimated.position.x_val
                yCurrent = self.state.kinematics_estimated.position.y_val
                zCurrent = self.state.kinematics_estimated.position.z_val

                xCanditate = xCurrent + np.cos(randomOrientation[i] + currentYaw)*speedScalar*travelTime[i]
                yCanditate = yCurrent + np.sin(randomOrientation[i] + currentYaw)*speedScalar*travelTime[i]
                zCanditate = zCurrent

                canditates = [xCanditate,yCanditate,zCanditate]

                inGeoFence = self.insideGeoFence(c = canditates, d = minDist)

                # the estimated score each canditate point has
                if safeDist and inGeoFence:
                    # jPoint.append(dist)
                    angle =  np.radians(yawCanditate[i]) + currentYaw + randomOrientation[i]
                    if angle>np.pi:
                        angle=np.pi
                        yawCanditate[i] = angle
                    elif angle<-np.pi:
                        angle=-np.pi
                        yawCanditate[i] = angle

                    jPoint.append(self.estimate(xCanditate, yCanditate, angle))
                    availablePosition = True
                else:
                    # canditate position is outside geo-fence or on collision
                    jPoint.append(-1000.)

            if availablePosition:
                break
            else:
                # there is no avaoilable point tin the current orientation, change it
                print(f"\n[WARNING][MOVE] There is no available position for {self.getName()}, changing orientation")
                # rotateByYawRateAsync takes as input degres
                self.client.rotateByYawRateAsync(30,1,vehicle_name=self.name).join()
                self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()

        tartgetPointIndex = np.argmax(jPoint)

        anglesSpeed = currentYaw + randomOrientation[tartgetPointIndex]

        vx = np.cos(anglesSpeed)
        vy = np.sin(anglesSpeed)

        # travelTime = np.random.uniform(maxTravelTime)

        self.client.moveByVelocityZAsync(speedScalar*vx, speedScalar*vy, axiZ, travelTime[tartgetPointIndex],
                                    airsim.DrivetrainType.ForwardOnly,
                                    airsim.YawMode(False, 0),
                                    vehicle_name=self.name).join()
        self.stabilize().join()

        self.client.rotateByYawRateAsync(yawCanditate[tartgetPointIndex] + np.degrees(currentYaw),1,vehicle_name=self.name).join()
        self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()

        if DEBUG_MOVE:
            print(f"\n[DEBUG][MOVE] ----- {self.getName()} -----")
            print(f"[DEBUG][MOVE] randomOrientation: {np.degrees(randomOrientation)}")
            print(f"[DEBUG][MOVE] travelTime: {travelTime}")
            print(f"[DEBUG][MOVE] yawCanditate: {yawCanditate}")
            print(f"[DEBUG][MOVE] jPoint: {jPoint}")
            print(f"[DEBUG][MOVE] tartgetPointIndex: {tartgetPointIndex}")


    def estimate(self,x,y,yaw):

        return float(self.estimator.predict([[x,y,np.radians(yaw)]]))


    def estimate1DoF(self,yaw):

        yaw = (yaw - 180)/(180 + 180)
        return float(self.estimator1DoF.predict([[yaw]]))


    def estimate2Dof(self,x,y,yaw):

        return float(self.estimator.predict([[x,y,np.radians(yaw)]]))


    def updateEstimator(self):

        xList = [(state[0].kinematics_estimated.position.x_val-self.minX)/(self.maxX - self.minX) for state in self.stateList]
        yList = [(state[0].kinematics_estimated.position.y_val-self.minY)/(self.maxY-self.minY) for state in self.stateList]
        yawList = [(airsim.to_eularian_angles(state[0].kinematics_estimated.orientation)[2]-np.pi)/(np.pi+np.pi) for state in self.stateList]

        contributionList = [[con] for con in self.contribution]

        data = []
        for i in range(len(xList)):
            data.append([xList[i],yList[i],yawList[i]])

        # print(f" data:{data} contributionList:{contributionList}")
        self.estimator = self.model.fit(data,contributionList)


    def updateEstimator1DoF(self):

        yawList = [[(np.degrees(airsim.to_eularian_angles(state[0].kinematics_estimated.orientation)[2])-180)/(180+180)] for state in self.stateList]

        contributionList = [[con] for con in self.contribution]

        self.historyData.append([yawList,contributionList])

        self.estimator1DoF = self.model1DoF.fit(yawList[-4:],contributionList[-4:])


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
        elif abs(index) > len(self.scoreDetections):
            return 0.0
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

        estimations_file = os.path.join(os.getcwd(), "swarm_raw_output",
                                  self.getName(), f"estimations_{self.name}.pickle")
        pickle.dump(self.estimations,open(estimations_file,"wb"))

        history_file = os.path.join(os.getcwd(), "swarm_raw_output",
                                  self.getName(), f"history_{self.name}.pickle")
        pickle.dump(self.historyData,open(history_file,"wb"))
