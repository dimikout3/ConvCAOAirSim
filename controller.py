# import setup_path
import airsim
import os
import cv2
import numpy as np
import time
import pickle
import utilities.utils as utils
import matplotlib.pyplot as plt
import yoloDetector

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

ESTIMATORWINDOW = 55
# the value wich will devide the field of view (constraing the yaw movement)
CAM_DEV = 4
ORIENTATION_DEV = 4

# how may tries we will try to get images from AirSim
IMAGES_TRIES = 10

SAVE_DETECTED = False

DEBUG_GEOFENCE = False
DEBUG_RANDOMZ = False
DEBUG_MOVE = False
DEBUG_MOVE1DOF = False
DEBUG_MOVE_OMNI = False
DEBUG_ESTIMATOR = False
WEIGHTS = {"cars":1.0, "persons":0.0 , "trafficLights":1.0}

class controller:

    def __init__(self, clientIn, droneName, offSets, ip="1", timeWindow=200):

        self.client = clientIn
        self.name = droneName

        self.ip = ip

        self.detector = yoloDetector.yoloDetector()

        self.scoreDetections = []
        self.scoreDetectionsNum = []
        self.detectionsInfo = []
        self.detectionsCoordinates = []
        self.pointCloud = []

        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

        self.offSetX = offSets[0]
        self.offSetY = offSets[1]
        self.offSetZ = offSets[2]

        self.updateMultirotorState()
        self.updateCameraInfo()
        # self.state = self.getState()
        # self.cameraInfo = self.getCameraInfo()

        self.stateList = []

        self.model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                               ('linear', LinearRegression())])

        self.estimator = self.model.fit([np.random.uniform(0,1,3)],[np.random.uniform(0,1)])

        self.estimations = []
        self.historyData = []

        # how much vehicles currrent movement affected the cost Function (delta)
        self.contribution = []
        self.j_i = []

        self.timeStep = 0
        self.posIdx = 0

        self.restrictingMovement = np.linspace(1,0.1,timeWindow)

        self.parentRaw = os.path.join(os.getcwd(),f"results_{ip}", "swarm_raw_output")
        try:
            os.makedirs(self.parentRaw)
        except OSError:
            if not os.path.isdir(self.parentRaw):
                raise

        self.parentDetect = os.path.join(os.getcwd(),f"results_{ip}", "swarm_detected")
        try:
            os.makedirs(self.parentDetect)
        except OSError:
            if not os.path.isdir(self.parentDetect):
                raise


    def takeOff(self):

        return self.client.takeoffAsync(vehicle_name = self.name)


    def hover(self):

        return self.client.hoverAsync(vehicle_name=self.name)


    def moveToPositionYawModeAsync(self, x, y, z, speed=1, yawmode = 0):
        # moveToPositionAsync works only for relative coordinates, therefore we must
        # subtrack the offset (which corresponds to global coordinates)
        x -= self.offSetX
        y -= self.offSetY
        z -= self.offSetZ

        return self.client.moveToPositionAsync(x, y, z, speed,
                                               yaw_mode = airsim.YawMode(False, yawmode),
                                               vehicle_name=self.name)


    def moveToPositionYawMode(self, x, y, z, speed, yawmode = 0):
        # moveToPositionAsync works only for relative coordinates, therefore we must
        # subtrack the offset (which corresponds to global coordinates)
        x -= self.offSetX
        y -= self.offSetY
        z -= self.offSetZ

        return self.client.moveToPositionAsync(x, y, z, speed,
                                               yaw_mode = airsim.YawMode(False, yawmode),
                                               vehicle_name=self.name).join()


    def moveToPosition(self, x, y, z, speed):

        # moveToPositionAsync works only for relative coordinates, therefore we must
        # subtrack the offset (which corresponds to global coordinates)
        x -= self.offSetX
        y -= self.offSetY
        z -= self.offSetZ

        return self.client.moveToPositionAsync(x,y,z,speed,vehicle_name=self.name).join()


    def setCameraOrientation(self, cam_yaw, cam_pitch, cam_roll):

        self.client.simSetCameraOrientation("0",
                                            airsim.to_quaternion(cam_yaw, cam_pitch, cam_roll),
                                            vehicle_name = self.name)


    def getName(self):

        return self.name


    def getImages(self, save_raw=False):

        for _ in range(IMAGES_TRIES):

            try:
                responses = self.client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True),  #depth visualization image
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True),
                    airsim.ImageRequest("4", airsim.ImageType.DepthPerspective, True)],
                    vehicle_name = self.name)  #scene vision image in uncompressed RGB array

                img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8) #get numpy array
                img_rgb = img1d.reshape(responses[1].height, responses[1].width, 3) #reshape array to 3 channel image array H X W X 3
                self.imageScene = img_rgb
                self.imageDepthCamera = responses[0]

                imageDepthFront = airsim.list_to_2d_float_array(responses[2].image_data_float,
                                                           responses[2].width,
                                                           responses[2].height)
                self.imageDepthFront = imageDepthFront

                imageDepthBack = airsim.list_to_2d_float_array(responses[3].image_data_float,
                                                           responses[3].width,
                                                           responses[3].height)
                self.imageDepthBack = imageDepthBack

                self.imageDepthPeripheralWidth = responses[3].width
                self.imageDepthPeripheralHeight = responses[3].height

                if save_raw:

                    filenameDepth = os.path.join(self.raw_dir, f"depth_time_{self.timeStep}" )
                    airsim.write_pfm(os.path.normpath(filenameDepth + '.pfm'), airsim.get_pfm_array(responses[0]))

                    filenameScene = os.path.join(self.raw_dir, f"scene_time_{self.timeStep}" )
                    cv2.imwrite(os.path.normpath(filenameScene + '.png'), img_rgb) # write to png

                # if code reach here, we should break the loop
                break
            except:
                pass

        return responses


    def getPointCloud(self, x=500, y=500):

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


    def appendJi(self, Ji):

        self.j_i.append(Ji)


    def getJi(self, index=-1):

        return self.j_i[index]


    def getJiList(self):

        return self.j_i


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


    def rotateYawRelative(self, relavtiveYaw):

        self.client.rotateByYawRateAsync(relavtiveYaw,1,vehicle_name=self.name).join()
        self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()


    def rotateToYaw(self, yaw):

        self.updateMultirotorState()
        _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)

        self.client.rotateByYawRateAsync(float(yaw) - np.degrees(currentYaw),1,vehicle_name=self.name).join()
        self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()
        # self.client.rotateToYawAsync(yaw, vehicle_name=self.name).join()


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

            # self.state = self.getState()
            self.updateMultirotorState()


    def move(self, randomPointsSize=70, maxTravelTime=5., minDist=5., plotEstimator=True):

        axiZ = self.altitude

        speedScalar = 2
        np.random.seed()

        # camera field of view (degrees)
        camFOV = self.cameraInfo.fov
        leftDeg, rightDeg = -camFOV/2 , camFOV/2

        maxTries = 12
        availablePosition = False

        for i in range(maxTries):

            # self.state = self.getState()
            self.updateMultirotorState()
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
            randomOrientation = np.random.uniform(np.radians(leftDeg/ORIENTATION_DEV), np.radians(rightDeg/ORIENTATION_DEV), randomPointsSize)
            # travelTime = np.random.uniform(0, maxTravelTime, randomPointsSize)
            travelTime = np.random.uniform(0., maxTravelTime, randomPointsSize)
            yawCanditate = np.random.uniform(leftDeg/CAM_DEV,rightDeg/CAM_DEV,randomPointsSize)

            # validPoint = []
            jPoint = []
            xCanditateList = []
            yCanditateList = []
            yawAngleList = []
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
                angle =  np.radians(yawCanditate[i]) + currentYaw + randomOrientation[i]

                xCanditateList.append(xCanditate)
                yCanditateList.append(yCanditate)
                yawAngleList.append(angle)

                canditates = [xCanditate,yCanditate,zCanditate]

                inGeoFence = self.insideGeoFence(c = canditates, d = minDist)

                # the estimated score each canditate point has
                if safeDist and inGeoFence:

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

        self.client.rotateByYawRateAsync(yawCanditate[tartgetPointIndex],1,vehicle_name=self.name).join()
        self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()

        if plotEstimator:
            self.plotEstimator(xCanditateList, yCanditateList, yawAngleList, jPoint)

        if DEBUG_MOVE:
            print(f"\n[DEBUG][MOVE] ----- {self.getName()} -----")
            print(f"[DEBUG][MOVE] randomOrientation: {np.degrees(randomOrientation)}")
            print(f"[DEBUG][MOVE] travelTime: {travelTime}")
            print(f"[DEBUG][MOVE] yawCanditate: {yawCanditate}")
            print(f"[DEBUG][MOVE] jPoint: {jPoint}")
            print(f"[DEBUG][MOVE] tartgetPointIndex: {tartgetPointIndex}")


    def getPeripheralView(self):

        imageDepth = []

        imageDepthFront = self.imageDepthFront
        imageDepth.append(imageDepthFront)

        imageDepthBack = self.imageDepthBack
        imageDepth.append(imageDepthBack)

        return imageDepth, self.imageDepthPeripheralHeight, self.imageDepthPeripheralWidth


    def getCanditates(self, randomPointsSize=70, maxTravelTime=5., minDist=5., maxYaw=15.):

        speedScalar = 1
        np.random.seed()

        self.updateMultirotorState()
        _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)

        # camera field of view (degrees)
        camFOV = self.cameraInfo.fov
        leftDeg, rightDeg = -camFOV/2 , camFOV/2

        imageDepthList, height, width = self.getPeripheralView()

        # height, width = imageDepthList[0].height, imageDepthList[0].width
        pixel10H = height*0.1
        lowHeight, highHeight = int(height/2-pixel10H), int(height/2+pixel10H)
        wLow, wHigh = int(width*0.1) ,int(width*0.1)

        jEstimated = []
        xCanditateList = []
        yCanditateList = []
        zCanditateList = []
        yawCanditateList = []

        # decreasing the available movement because we are getting closer to convergence point
        # initialy we wan to explore and then exploit more carefully
        a = self.restrictingMovement[self.timeStep]

        counter = 0
        inGeoFenceCounter = 0
        inSafeDistCounter = 0

        while (jEstimated == []) and (counter<38):

            counter += 1

            if counter>1:
                print(f"[CANDITATES] {self.getName()} regeting images ... ")
                print(f"[CANDITATES] {self.getName()} has canditates inGeoFence={inGeoFenceCounter}, inSafeDist={inSafeDistCounter}")
                self.rotateYawRelative(10)
                self.updateMultirotorState()
                _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)
                print(f"[CANDITATES] {self.getName()} rotating in order to find canditates (currentYaw={np.degrees(currentYaw):.3f}) move more freely (no restrictingMovement)")
                a = 2
                self.getImages()
                imageDepthList, height, width = self.getPeripheralView()
                inGeoFenceCounter = 0
                inSafeDistCounter = 0

            for imageIdx,imageDepth in enumerate(imageDepthList):

                # boundaries should be avoided for collision avoidance (thats what +/- 3 degrees do ...)
                randomOrientation = np.random.uniform(np.radians(leftDeg + 3), np.radians(rightDeg - 3), randomPointsSize)
                # travelTime = np.random.uniform(0, maxTravelTime, randomPointsSize)
                travelTime = np.random.uniform(0., maxTravelTime, randomPointsSize)
                yawCanditate = np.random.uniform(np.degrees(currentYaw) - (maxYaw/2)*a, np.degrees(currentYaw) + (maxYaw/2)*a, randomPointsSize)

                for i in range(randomPointsSize):

                    wCenter = int(  width * ( ( np.degrees(randomOrientation[i]) + camFOV/2 ) / camFOV ) )
                    if wCenter - wLow < 0: wCenter = wLow + 1
                    if wCenter + wHigh> (width-1): wCenter = wHigh + 1
                    # print(f"{self.getName()} wCenter:{wCenter}")

                    dist = np.min(imageDepth[ (wCenter-wLow) : (wCenter+wHigh), lowHeight:highHeight])

                    safeDist = dist>(travelTime[i]*speedScalar*a + minDist)

                    xCurrent = self.state.kinematics_estimated.position.x_val
                    yCurrent = self.state.kinematics_estimated.position.y_val
                    zCurrent = self.state.kinematics_estimated.position.z_val

                    xCanditate = xCurrent + np.cos( (randomOrientation[i] + imageIdx*np.pi) + currentYaw)*speedScalar*travelTime[i]*a
                    yCanditate = yCurrent + np.sin( (randomOrientation[i] + imageIdx*np.pi) + currentYaw)*speedScalar*travelTime[i]*a
                    zCanditate = zCurrent

                    canditates = [xCanditate,yCanditate,zCanditate]
                    inGeoFence = self.insideGeoFence(c = canditates, d = minDist)

                    if safeDist: inSafeDistCounter += 1
                    if inGeoFence: inGeoFenceCounter += 1

                    # the estimated score each canditate point has
                    if safeDist and inGeoFence:

                        jEstimated.append(self.estimate(xCanditate, yCanditate, np.radians(yawCanditate[i]) ))
                        xCanditateList.append(xCanditate)
                        yCanditateList.append(yCanditate)
                        zCanditateList.append(zCanditate)
                        yawCanditateList.append(yawCanditate[i])

        return jEstimated, xCanditateList, yCanditateList, zCanditateList, yawCanditateList


    def moveOmniDirectional(self, randomPointsSize=70, maxTravelTime=5.,
                            minDist=5., plotEstimator=True, maxYaw=15.):

        canditatesData = self.getCanditates(randomPointsSize=randomPointsSize,
                                            maxTravelTime=maxTravelTime,
                                            minDist=minDist,
                                            maxYaw=maxYaw)

        jEstimated, xCanditateList, yCanditateList, zCanditateList, yawCanditateList = canditatesData

        # tartgetPointIndex = np.argmax(jEstimated)
        tartgetPointIndex = np.argmin(jEstimated)

        task = self.moveToPositionYawModeAsync(xCanditateList[tartgetPointIndex],
                                   yCanditateList[tartgetPointIndex],
                                   zCanditateList[tartgetPointIndex],
                                   2,
                                   yawmode = yawCanditateList[tartgetPointIndex])

        if plotEstimator:
            self.plotEstimator(xCanditateList, yCanditateList, yawCanditateList, jEstimated)

        if DEBUG_MOVE_OMNI:
            print(f"\n[DEBUG][MOVE_OMNI] ----- {self.getName()} -----")
            print(f"[DEBUG][MOVE_OMNI] target pose (x:{xCanditateList[tartgetPointIndex]:.2f} ,y:{yCanditateList[tartgetPointIndex]:.2f}, z:{zCanditateList[tartgetPointIndex]:.2f}, yaw:{yawCanditateList[tartgetPointIndex]:.2f})")

        return task

    def estimate(self,x,y,yaw):

        if yaw > np.pi:
            yaw = -np.pi*2 + yaw
        if yaw < -np.pi:
            yaw = np.pi*2 - yaw

        return float(self.estimator.predict([[x,y,yaw]]))


    def updateEstimator(self):

        xList = [state[0].kinematics_estimated.position.x_val for state in self.stateList]
        yList = [state[0].kinematics_estimated.position.y_val for state in self.stateList]
        yawList = [airsim.to_eularian_angles(state[0].kinematics_estimated.orientation)[2] for state in self.stateList]

        yawListDegrees = [np.degrees(airsim.to_eularian_angles(state[0].kinematics_estimated.orientation)[2]) for state in self.stateList]

        data = np.stack((xList,yList,yawList),axis=1)
        dataDegrees = np.stack((xList,yList,yawListDegrees),axis=1)

        if DEBUG_ESTIMATOR:
            print(f"\n[ESTIMATOR] {self.getName()} is using data:{[list(i) for i in dataDegrees[-ESTIMATORWINDOW:]]} and Ji:{self.j_i[-ESTIMATORWINDOW:]}")

        weights = np.linspace(1,1,len(data[-ESTIMATORWINDOW:]))

        self.estimator = self.model.fit(data[-ESTIMATORWINDOW:],self.j_i[-ESTIMATORWINDOW:], **{'linear__sample_weight': weights})


    def plotEstimator(self, xCanditate, yCanditate, yawCanditate, JCanditate):

        report_estimator = os.path.join(os.getcwd(),f"results_{self.ip}", f"estimator_{self.getName()}")
        try:
            os.makedirs(report_estimator)
        except OSError:
            if not os.path.isdir(report_estimator):
                raise

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,10))

        yawCanditate = [np.radians(yaw) for yaw in yawCanditate]

        u = np.cos(yawCanditate)
        v = np.sin(yawCanditate)

        # XXX: reverting x,y and therefore u,v to match counterclockwise orientation, defined in AirSim
        ax_1 = ax1.quiver(yCanditate, xCanditate, v, u, JCanditate, cmap=plt.cm.seismic)
        # ax1.colorbar()
        fig.colorbar(ax_1, ax=ax1)

        ax1.set_xlabel("Y-Axis (network)")
        ax1.set_ylabel("X-Axis (network)")

        x = [state[0].kinematics_estimated.position.x_val for state in self.stateList ]
        y = [state[0].kinematics_estimated.position.y_val for state in self.stateList ]
        yaw = [state[1].pose.orientation for state in self.stateList ]
        # to_eularian_angles -> (picth, roll, yaw)
        yaw = [airsim.to_eularian_angles(i)[2] for i in yaw]
        u = np.cos(yaw)
        v = np.sin(yaw)
        ji = self.j_i

        ax_2 = ax2.quiver(y[-ESTIMATORWINDOW:],x[-ESTIMATORWINDOW:],v[-ESTIMATORWINDOW:],
                   u[-ESTIMATORWINDOW:],ji[-ESTIMATORWINDOW:], cmap=plt.cm.seismic)
        # ax2.colorbar()
        fig.colorbar(ax_2, ax=ax2)
        ax2.set_xlabel("Y-Axis (network)")
        ax2.set_ylabel("X-Axis (network)")

        try:
            estimator_file = os.path.join(report_estimator, f"estimaton_{self.posIdx}.png")
            # plt.title(f"{self.getName()} - Time:{self.timeStep} - Canditates")
        except:
            estimator_file = os.path.join(report_estimator, f"estimaton_{0}.png")
            # plt.title(f"{self.getName()} - Time:{0} - Canditates")

        # ax1.tight_layout()
        # ax2.tight_layout()
        plt.tight_layout()

        plt.savefig(estimator_file)
        plt.close()


    def updateState(self, posIdx, timeStep):

        # self.state = self.getState()
        self.updateMultirotorState()

        # self.cameraInfo = self.getCameraInfo()
        self.updateCameraInfo()

        self.posIdx = posIdx
        self.timeStep = timeStep

        self.raw_dir = os.path.join(self.parentRaw, self.name, f"position_{self.posIdx}")
        if not os.path.isdir(self.raw_dir):
            os.makedirs(self.raw_dir)

        self.detected_dir = os.path.join(self.parentDetect, self.name, f"position_{self.posIdx}")
        if not os.path.isdir(self.detected_dir):
            os.makedirs(self.detected_dir)


    def detectObjects(self, save_detected=SAVE_DETECTED):

        # detector = yoloDetector.yoloDetector()

        detected_file_name = None
        if save_detected:
            detected_file_name = os.path.join(self.detected_dir,
                                              f"detected_time_{self.timeStep}.png")

        detections = self.detector.detect(self.imageScene, display=False, save=detected_file_name)

        # detections = {'cars':[(pixel_x,pixel_y,detecions_id,confidece),(pixel_x,pixel_y,detecions_id,confidece), ...]}
        # val[4] -> confidence of each detections
        # score = sum([val[4]*WEIGHTS[key] for key,val in detections.items()])
        # score = 0.0
        pixelX = []
        pixelY = []
        # scoreNum -> number of detected objects (int)
        scoreNum = 0
        for detectionClass, objects in detections.items():
            for object in objects:
                # object = (pixel_x,pixel_y,detecions_id,confidece)
                # score += object[3]*WEIGHTS[detectionClass]
                pixelX.append(object[0])
                pixelY.append(object[1])
                # scoreNum += WEIGHTS[detectionClass]

        # depthImage = self.getDepthImage()
        depthImage = self.imageDepthCamera

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

        # self.scoreDetections.append(score)
        # self.scoreDetectionsNum.append(scoreNum)
        self.detectionsInfo.append(detectionsInfo)
        self.detectionsCoordinates.append(detectionsCoordinates)
        # self.stateList.append([self.getState(), self.getCameraInfo()])
        self.stateList.append([self.state, self.cameraInfo])


    def getDetections(self):

        return self.detectionsCoordinates[-1], self.detectionsInfo[-1]


    def getDistanceClosestDetection(self, currentDetection={}):

        min = 2*self.fenceR

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        for drone, detections in currentDetection.items():
            # print(f"detections:{detections}")
            if (detections == []) or (self.getName() == drone):
                continue

            for xDetection , yDetection, zDetection in detections[0]:

                dist = np.sqrt( (xCurrent - xDetection)**2 + (yCurrent - yDetection)**2 + (zCurrent - zDetection)**2)

                if dist<min:
                    min = dist

        if min < 2*self.fenceR:
            return min

        for detections in self.detectionsCoordinates:
            for xDetection , yDetection, zDetection in detections:

                dist = np.sqrt( (xCurrent - xDetection)**2 + (yCurrent - yDetection)**2 + (zCurrent - zDetection)**2)

                if dist<min:
                    min = dist

        return min


    def scoreExcludingDetections(self, index=-1, excludedList=[], minusDuplicates = True):
        """ Excluding dections that have been better detected (higher confidence)
            from other drones """

        if abs(index)>len(self.detectionsCoordinates):
            index=-1

        tempCoordinates = self.detectionsCoordinates[index]
        tempInfo = self.detectionsInfo[index]

        score = 0.

        for i, info in enumerate(tempInfo):

            if (i in excludedList) :
                if minusDuplicates:
                    score -= info[1]
            else:
                score += info[1]
            # del tempCoordinates[i]
            # del tempInfo[i]

        return score


    def updateScore(self):

        self.scoreDetections.append( np.sum(self.detectionsInfo[-1][1]) )
        self.scoreDetectionsNum.append( len(self.detectionsInfo[-1][1]) )


    def getScore(self, index = None, absolute = False):

        # absolute -> return score as absolute number of detected objects

        if not absolute:
            if index == None:
                # if no index is specified the whole list will be returned
                return self.scoreDetections
            elif abs(index) > len(self.scoreDetections):
                return 0.0
            else:
                # usually used with index -1
                return self.scoreDetections[index]
        else:
            if index == None:
                # if no index is specified the whole list will be returned
                return self.scoreDetectionsNum
            elif abs(index) > len(self.scoreDetectionsNum):
                return 0.0
            else:
                # usually used with index -1
                return self.scoreDetectionsNum[index]


    def getPose(self):
        return self.client.simGetVehiclePose(vehicle_name=self.name)


    def getState(self):

        self.updateMultirotorState()
        return self.state

#
    def updateMultirotorState(self):

        self.state = self.client.getMultirotorState(vehicle_name=self.name)

        self.state.kinematics_estimated.position.x_val += self.offSetX
        self.state.kinematics_estimated.position.y_val += self.offSetY
        self.state.kinematics_estimated.position.z_val += self.offSetZ

        self.orientation = self.state.kinematics_estimated.orientation
        self.position  = self.state.kinematics_estimated.position


    def updateCameraInfo(self, cam="0"):

        self.cameraInfo = self.client.simGetCameraInfo(cam,vehicle_name=self.name)

        self.cameraInfo.pose.position.x_val += self.offSetX
        self.cameraInfo.pose.position.y_val += self.offSetY
        self.cameraInfo.pose.position.z_val += self.offSetZ


    def getPositions(self, index=None):

        if index is None:

            return self.position

        else:

            stateList = self.getStateList()
            stateDrone, _ = stateList[index]

            return stateDrone.kinematics_estimated.position


    def getOrientation(self):

        return self.orientation


    def getStateList(self):

        return self.stateList


    def getCameraInfo(self, cam="0"):

        return self.client.simGetCameraInfo(cam,vehicle_name=self.name)


    def getDetectionsCoordinates(self, index=-1):

        if abs(index) > len(self.detectionsCoordinates):
            return self.detectionsCoordinates[-1]
        else:
            return self.detectionsCoordinates[index]


    def getDetectionsInfo(self, index=-1):

        if abs(index) > len(self.detectionsInfo):
            return self.detectionsInfo[-1]
        else:
            return self.detectionsInfo[index]


    def quit(self):

        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)

        # TODO: possible np.save() instead of pickle ...
        score_detections_file = os.path.join(self.parentRaw,
                                        self.getName(), f"score_detections_{self.name}.pickle")
        pickle.dump(self.scoreDetections,open(score_detections_file,"wb"))
        score_detections_file = os.path.join(os.getcwd(), f"results_{self.ip}","information",
                                        f"score_detections_{self.name}.pickle")
        pickle.dump(self.scoreDetections,open(score_detections_file,"wb"))

        detections_file = os.path.join(self.parentRaw,
                                       self.getName(), f"detectionsInfo_{self.name}.pickle")
        pickle.dump(self.detectionsInfo,open(detections_file,"wb"))
        detections_file = os.path.join(os.getcwd(), f"results_{self.ip}","detected_objects",
                                       f"detectionsInfo_{self.name}.pickle")
        pickle.dump(self.detectionsInfo,open(detections_file,"wb"))

        detections_file = os.path.join(self.parentRaw,
                                       self.getName(), f"detectionsCoordinates_{self.name}.pickle")
        pickle.dump(self.detectionsCoordinates,open(detections_file,"wb"))
        detections_file = os.path.join(os.getcwd(), f"results_{self.ip}","detected_objects",
                                       f"detectionsCoordinates_{self.name}.pickle")
        pickle.dump(self.detectionsCoordinates,open(detections_file,"wb"))

        state_file = os.path.join(self.parentRaw,
                                  self.getName(), f"state_{self.name}.pickle")
        pickle.dump(self.stateList,open(state_file,"wb"))

        pointCloud_file = os.path.join(self.parentRaw,
                                  self.getName(), f"pointCloud_{self.name}.pickle")
        pickle.dump(self.pointCloud,open(pointCloud_file,"wb"))

        contribution_file = os.path.join(self.parentRaw,
                                  self.getName(), f"contribution_{self.name}.pickle")
        pickle.dump(self.contribution,open(contribution_file,"wb"))

        estimations_file = os.path.join(self.parentRaw,
                                  self.getName(), f"estimations_{self.name}.pickle")
        pickle.dump(self.estimations,open(estimations_file,"wb"))

        history_file = os.path.join(self.parentRaw,
                                  self.getName(), f"history_{self.name}.pickle")
        pickle.dump(self.historyData,open(history_file,"wb"))
