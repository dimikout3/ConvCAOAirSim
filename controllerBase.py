# import setup_path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import airsim
import os
import cv2
import numpy as np
import time
import pickle
import utilities.utils as utils
import matplotlib.pyplot as plt

import scipy
from scipy.spatial import Delaunay

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# the value wich will devide the field of view (constraing the yaw movement)
CAM_DEV = 4
ORIENTATION_DEV = 4

# how may tries we will try to get images from AirSim
IMAGES_TRIES = 10

SAVE_DETECTED = True

DEBUG_GEOFENCE = False
DEBUG_RANDOMZ = False
DEBUG_MOVE = False
DEBUG_MOVE1DOF = False
DEBUG_MOVE_OMNI = False
DEBUG_ESTIMATOR = False
DEBUG_CANDITATE_LIDAR = False
DEBUG_CLEAR_LIDAR = False
DEBUG_LIDAR_DIST = False
PLOT_CANDITATES = True
WEIGHTS = {"cars":1.0, "persons":0.0 , "trafficLights":1.0}

class controller:

    def __init__(self, clientIn, droneName, offSets, ip="1",
                 wayPointsSize=200, estimatorWindow=55, maxDistView=None):

        self.client = clientIn
        self.name = droneName

        self.lidarName = "Lidar1"

        self.ip = ip

        self.estimatorWindow = estimatorWindow

        self.pointCloud = []

        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

        self.offSetX = offSets[0]
        self.offSetY = offSets[1]
        self.offSetZ = offSets[2]

        self.wayPointSize = wayPointsSize

        self.maxDistView = maxDistView

        self.updateMultirotorState()
        self.updateCameraInfo()

        self.stateList = []

        self.model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                               ('linear', LinearRegression())])

        self.estimator = self.model.fit([np.random.uniform(0,1,3)],[np.random.uniform(0,1)])

        self.estimations = []
        self.historyData = []

        # how much vehicles currrent movement affected the cost Function (delta)
        self.contribution = []
        self.j_i = []

        self.informationJ = []
        self.informationJi = []

        self.timeStep = 0
        self.posIdx = 0

        self.restrictingMovement = np.linspace(1,0.1,wayPointsSize)
        self.estimatorWeights = np.linspace(1,1,self.estimatorWindow)

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


    def setPose(self, x, y, z, pitch, roll, yaw):

        x -= self.offSetX
        y -= self.offSetY
        z -= self.offSetZ

        position = airsim.Vector3r(x , y, z)
        heading = airsim.to_quaternion(pitch, roll, yaw)
        pose = airsim.Pose(position, heading)

        return self.client.simSetVehiclePose(pose, True,vehicle_name=self.name)


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

        return self.client.moveToPositionAsync(x,y,z,speed,vehicle_name=self.name)


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
                    airsim.ImageRequest("4", airsim.ImageType.DepthPerspective, True),
                    airsim.ImageRequest("1", airsim.ImageType.Segmentation, True), #depth in perspective projection
                    airsim.ImageRequest("4", airsim.ImageType.Segmentation, True)],
                    vehicle_name = self.name)  #scene vision image in uncompressed RGB array

                img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8) #get numpy array
                if os.name=='nt':
                    img_rgb = img1d.reshape(responses[1].height, responses[1].width, 3) #reshape array to 3 channel image array H X W X 3
                else:
                    img_rgb = img1d.reshape(responses[1].height, responses[1].width, 4) #reshape array to 3 channel image array H X W X 3
                    img_rgb = img_rgb[:,:,0:3]
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

                self.imageScene = img_rgb
                self.imageDepthCamera = responses[0]

                imageDepthFront = airsim.list_to_2d_float_array(responses[2].image_data_float,
                                                           responses[2].width,
                                                           responses[2].height)
                self.imageDepthFront = imageDepthFront
                self.imageDepthFrontRaw = responses[2]
                self.frontSegmented  = airsim.list_to_2d_float_array(responses[4].image_data_float,
                                                           responses[4].width,
                                                           responses[4].height)

                imageDepthBack = airsim.list_to_2d_float_array(responses[3].image_data_float,
                                                           responses[3].width,
                                                           responses[3].height)
                self.imageDepthBack = imageDepthBack
                self.imageDepthBackRaw = responses[3]
                self.backSegmented  = airsim.list_to_2d_float_array(responses[5].image_data_float,
                                                           responses[5].width,
                                                           responses[5].height)

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

        r = np.random.uniform(0,min(halfHeight,halfWidth),randomPointsSize)
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
                                          color = colors, maxDistView = self.maxDistView)
        x, y, z = utils.to_absolute_coordinates(xRelative, yRelative, zRelative,
                                                self.cameraInfo)


        # utils.plot3dColor(x,y,z,colors,show=True)

        self.pointCloud.append([x,y,z,colors])

        return x,y,z,colors
    # TODO: getDepth() -> from camera "0", similar to rgb


    def getCanditatesSegmented(self, maxDistTravel=5., size=500, safeDist=2.):

        depthImageList = [(self.imageDepthFrontRaw, self.imageDepthFront, self.frontSegmented, self.cameraInfoFront),
                          (self.imageDepthBackRaw, self.imageDepthBack, self.backSegmented, self.cameraInfoBack)]

        height, width, _ = self.imageScene.shape
        halfWidth = width/2
        halfHeight= height/2

        centerH = int(halfHeight)
        centerW = int(halfWidth)

        randomBatchSize = int(size/len(depthImageList))

        x = np.array([])
        y = np.array([])
        z = np.array([])

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val
        uav = np.array([xCurrent, yCurrent, zCurrent])

        for depthImageRaw, depthImage, segmentedImage, camInfo in depthImageList:

            r = np.random.uniform(0,min(halfHeight,halfWidth),randomBatchSize)
            thetas = np.random.uniform(0,2*np.pi,randomBatchSize)
            distTravel = np.random.uniform(0, maxDistTravel, randomBatchSize)

            # Identify how many unique objects are in the segmented image
            uni = np.unique(segmentedImage)
            # find the closest point of each segment from the drone and map them
            # as dictionary. segMinDist -> { objectID_1 -> 13.4 [meters],
                                           # objectID_2 -> 33.4 [meters]}
            segMinDist = {}
            for objectID in uni:
                # import pdb; pdb.set_trace()
                segMinDist[objectID] = np.min( depthImage[segmentedImage == objectID] )

            pointsHFloat = r*np.sin(thetas)
            pointsWFloat = r*np.cos(thetas)

            pointsH = centerH + pointsHFloat.astype(int)
            pointsW = centerW + pointsWFloat.astype(int)

            # Points from center of image to target pixels (= the line from current
            # position towards the target position, in pixel coordinates)
            widthSpaced = np.linspace(np.repeat(centerW,randomBatchSize), pointsW, int(halfWidth))
            heightSpaced = np.linspace(np.repeat(centerH,randomBatchSize), pointsH, int(halfHeight))

            # segmentID of the points in the line
            #  axis_0 -> line step
            #  axis_1 -> pertubation
            segmentsSpaced = segmentedImage[widthSpaced.astype(int), heightSpaced.astype(int)]

            segmentClosest = np.zeros(segmentsSpaced.shape)
            for objectID, closestPoint in segMinDist.items():
                segmentClosest[segmentsSpaced == objectID] = closestPoint

            pertubationClosest = np.min( segmentClosest, axis=0)

            valid = np.where( pertubationClosest > distTravel + safeDist)

            xRelative, yRelative, zRelative = utils.vectorTo3D(pointsW, pointsH,
                                              camInfo, depthImage,
                                              maxDistView = self.maxDistView,
                                              vectorDistances = distTravel)
            xCommon, yCommon, zCommon = utils.to_absolute_coordinates(xRelative, yRelative, zRelative,
                                                    camInfo)

            x = np.append(x, xCommon[valid])
            y = np.append(y, yCommon[valid])
            z = np.append(z, zCommon[valid])

        safeCanditates = np.stack((x,y,z), axis=1)

        return safeCanditates


    def getPseudoLidar(self, size=1000):

        depthImageList = [(self.imageDepthFrontRaw, self.cameraInfoFront),
                          (self.imageDepthBackRaw, self.cameraInfoBack)]

        height, width, _ = self.imageScene.shape
        halfWidth = width/2
        halfHeight= height/2

        randomPointsSize = int(size/len(depthImageList))

        x = np.array([])
        y = np.array([])
        z = np.array([])

        hullList = []

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val
        uav = np.array([xCurrent, yCurrent, zCurrent])

        for depthImage, camInfo in depthImageList:

            r = np.random.uniform(0,min(halfHeight,halfWidth),randomPointsSize)
            thetas = np.random.uniform(0,2*np.pi,randomPointsSize)

            pointsH = r*np.sin(thetas)
            pointsW = r*np.cos(thetas)

            centerH = int(halfHeight)
            centerW = int(halfWidth)

            pointsH = centerH + pointsH.astype(int)
            pointsW = centerW + pointsW.astype(int)

            xRelative, yRelative, zRelative = utils.to3D(pointsW, pointsH,
                                              camInfo, depthImage,
                                              maxDistView = self.maxDistView)
            xCommon, yCommon, zCommon = utils.to_absolute_coordinates(xRelative, yRelative, zRelative,
                                                    camInfo)

            x = np.append(x, xCommon)
            y = np.append(y, yCommon)
            z = np.append(z, zCommon)

            pointsHull = np.stack((xCommon, yCommon, zCommon), axis=1)
            pointsHull = np.append(pointsHull, uav.reshape(1,3), axis=0)
            hull = Delaunay(pointsHull)
            hullList.append(hull)

        lidarPoints = np.stack((x,y,z), axis=1)

        return lidarPoints, hullList


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


    def resetJi(self, resetStyle=""):

        if resetStyle == "gradientInformationJ":
            self.j_i = np.gradient(self.informationJ)
        elif resetStyle == "directInformationJ":
            self.j_i = self.informationJ.copy()
        elif resetStyle == "deltaInformationJi":
            self.j_i = self.informationJi.copy()


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

        """ Relative rotate in degrees"""

        self.client.rotateByYawRateAsync(relavtiveYaw,1,vehicle_name=self.name).join()
        self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()


    def rotateToYaw(self, yaw):

        self.updateMultirotorState()
        _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)

        self.client.rotateByYawRateAsync(float(yaw) - np.degrees(currentYaw),1,vehicle_name=self.name).join()
        self.client.rotateByYawRateAsync(0,1,vehicle_name=self.name).join()
        # self.client.rotateToYawAsync(yaw, vehicle_name=self.name).join()


    def setGeoFence(self, geofence):
        """Applying geo fence as object"""
        self.geoFence = geofence



    def insideGeoFence(self, d=5., c=[]):

        if type(c) is list:
            x,y,z = c
            dist = np.sqrt( (self.fenceX-x)**2 + (self.fenceY-y)**2 + (self.fenceZ-z)**2)
        elif type(c) is np.ndarray:
            # in that case c=[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3] ...]
            dist = scipy.spatial.distance.cdist(c, [np.array([self.fenceX, self.fenceY, self.fenceZ])])

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
        a = self.restrictingMovement[self.posIdx]

        counter = 0
        inGeoFenceCounter = 0
        inSafeDistCounter = 0

        while (jEstimated == []) and (counter<360):

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
                randomOrientation = np.random.uniform(np.radians(leftDeg + 5), np.radians(rightDeg - 5), randomPointsSize)
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


    def plotCanditates(self, xCanditate, yCanditate, zCanditate, isSafeDist, lidarPoints):

        report_canditates = os.path.join(os.getcwd(),f"results_{self.ip}", f"canditates_{self.getName()}")
        try:
            os.makedirs(report_canditates)
        except OSError:
            if not os.path.isdir(report_canditates):
                raise

        # add the lidar point cloud
        plt.scatter( lidarPoints[:,1], lidarPoints[:,0], c="green", label="Lidar")

        # safe canditates plot
        safeDistTrue = np.where(np.array(isSafeDist)==True)[0]
        # print(f"[DEBUG_CANDITATES]{self.getName()} has safeDist={len(safeDistTrue)}")
        plt.scatter(yCanditate[safeDistTrue], xCanditate[safeDistTrue], c="blue", label="Safe")

        # not safe canditates
        safeDistFalse = np.where(np.array(isSafeDist)==False)[0]
        # print(f"[DEBUG_CANDITATES]{self.getName()} has not safe ={len(safeDistFalse)}")
        plt.scatter(yCanditate[safeDistFalse], xCanditate[safeDistFalse], c="red", label="Not Safe")


        plt.xlabel("Y-Axis (network)")
        plt.ylabel("X-Axis (network)")

        try:
            canditates_file = os.path.join(report_canditates, f"canditates_{self.posIdx}.png")
            # plt.title(f"{self.getName()} - Time:{self.timeStep} - Canditates")
        except:
            canditates_file = os.path.join(report_canditates, f"canditates_{0}.png")

        plt.tight_layout()
        plt.legend()

        plt.savefig(canditates_file)
        plt.close()


    def getLidarData(self, save_lidar=False):

        # print(f"Getting lidar data for {self.getName()} from {self.lidarName}")
        for test in range(10):
            lidarData = self.client.getLidarData(lidar_name = self.lidarName,
                                             vehicle_name = self.getName())

            points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
            if points.size != 0:
                break
            # getLidarData failed ... wait and try to get lidar data again
            time.sleep(1)
        points = np.reshape(points, (int(points.shape[0]/3), 3))

        if save_lidar:
            filenameLidar = os.path.join(self.raw_dir, f"lidar_time_{self.timeStep}" )
            np.save(filenameLidar, points)

        return points


    def clearLidarPoints(self, lidarPoints=[], maxTravelTime=5., controllers=[]):

        initialSize = len(lidarPoints)

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        droneCurrent = np.array([xCurrent, yCurrent, zCurrent])

        distLidar2Drone = scipy.spatial.distance.cdist(lidarPoints, [droneCurrent])

        lidarPoints = lidarPoints[np.where(distLidar2Drone<=maxTravelTime*10.)[0]]

        for ctrl in controllers:
            if ctrl.getName() != self.getName():

                state = ctrl.getState()
                xCurrent = state.kinematics_estimated.position.x_val
                yCurrent = state.kinematics_estimated.position.y_val
                zCurrent = state.kinematics_estimated.position.z_val
                ctrlCurrent = np.array([xCurrent, yCurrent, zCurrent])

                distLidar2Drone = scipy.spatial.distance.cdist(lidarPoints, [ctrlCurrent])

                lidarPoints = lidarPoints[np.where(distLidar2Drone>=3.)[0]]

                # if DEBUG_CLEAR_LIDAR:
                #     print(f"{self.getName()} excluding lidar from {ctrl.getName()} excluded size {len(np.where(distLidar2Drone<2.)[0])}")

        if DEBUG_CLEAR_LIDAR:
            print(f"[DEBUG_CLEAR_LIDAR]{self.getName()} initial lidar points {initialSize} cleared {len(lidarPoints)}")

        return lidarPoints


    def addOffsetLidar(self, lidarPoints=[]):

        offset = np.array([self.offSetX, self.offSetY, self.offSetZ])
        points = lidarPoints + offset

        return points


    def distLine2Point(self, p1, p2, p3):
        """Distance from line p1p2 and point p3"""
        # return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        """Distance from line p2(canditate) and point p3(lidar point)"""
        return np.linalg.norm(p2-p3)


    def isSafeDist(self,canditate=[], lidarPoints=[], minDist=5.):

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        droneCurrent = np.array([xCurrent, yCurrent, zCurrent])

        if canditate.shape==(3,):
            for point in lidarPoints:
                dist = self.distLine2Point(droneCurrent, canditate, point)
                if dist<minDist:
                    return False
            return True
        else:
            # canditate = [[x1,y1,z1],[x2,y2,z2],[]]
            # canditateList = []
            # for x,y,z in canditate:
            #     cand = np.array([x,y,z])
            #     for point in lidarPoints:
            #         dist = self.distLine2Point(droneCurrent, cand, point)
            #         if dist<minDist:
            #             canditateList.append(False)
            #             break
            #     canditateList.append(True)
            # return canditateList
            dist = scipy.spatial.distance.cdist(lidarPoints, canditate)
            distMin = np.min(dist, axis=0)
            return distMin>minDist


    def getCanditatesLidar(self, randomPointsSize=70, maxTravelTime=5.,
                           minDist=5., maxYaw=15., controllers=[],
                           saveLidar=False):

        speedScalar = 1
        np.random.seed()

        self.updateMultirotorState()
        _,_,currentYaw = airsim.to_eularian_angles(self.state.kinematics_estimated.orientation)

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        # decreasing the available movement because we are getting closer to convergence point
        # initialy we wan to explore and then exploit more carefully
        a = self.restrictingMovement[self.posIdx]

        for helperIcreasedMove in np.linspace(1,5,40):

            if DEBUG_CANDITATE_LIDAR:
                print(f"{self.getName()} has helperIcreasedMove={helperIcreasedMove}")

            # [-np.pi, np.pi] canditates are inside a shpere with radius=maxTravelTime
            randomOrientation = np.random.uniform(-np.pi, np.pi, randomPointsSize)
            travelTime = np.random.uniform(0., maxTravelTime, randomPointsSize)
            yawCanditate = np.random.uniform(np.degrees(currentYaw) - (maxYaw/2)*a, np.degrees(currentYaw) + (maxYaw/2)*a, randomPointsSize)

            lidarPoints = self.getLidarData(save_lidar=saveLidar)
            lidarPoints = self.clearLidarPoints(lidarPoints=lidarPoints,
                                                maxTravelTime=maxTravelTime,
                                                controllers=controllers)
            lidarPoints = self.addOffsetLidar(lidarPoints=lidarPoints)

            if DEBUG_LIDAR_DIST:
                droneCurrent = np.array([xCurrent, yCurrent, zCurrent])
                dist = scipy.spatial.distance.cdist(lidarPoints, [droneCurrent])
                print(f"[DEBUG_LIDAR_DIST]{self.getName()} has minDist={minDist} min.dist.(drone,lidar)={np.min(dist)}")

            xCanditate = xCurrent + np.cos(randomOrientation)*speedScalar*travelTime*a*helperIcreasedMove
            yCanditate = yCurrent + np.sin(randomOrientation)*speedScalar*travelTime*a*helperIcreasedMove
            zCanditate = np.repeat(zCurrent,len(xCanditate))
            canditates = np.stack((xCanditate,yCanditate,zCanditate),axis=1)

            inGeoFence = self.insideGeoFence(c = canditates, d = minDist)
            isSafeDist = self.isSafeDist(canditate = canditates,
                                         lidarPoints = lidarPoints,
                                         minDist = minDist)
            # print(f"{self.getName()} isSafeDist={len(isSafeDist)}")
            # print(f"{self.getName()} inGeoFence={len(inGeoFence)}")

            geoFenceSafe = np.where(inGeoFence==True)[0]
            safeDistTrue = np.where(np.array(isSafeDist)==True)[0]

            validCandidatesIndex = np.intersect1d(geoFenceSafe, safeDistTrue)
            # print(f"{self.getName()} validCandidatesIndex={validCandidatesIndex} ")

            if PLOT_CANDITATES:
                self.plotCanditates(xCanditate, yCanditate, zCanditate, isSafeDist, lidarPoints)

            if validCandidatesIndex.size == 0:
                # something went wrong ...
                if helperIcreasedMove<4.:
                    # increase helperIcreasedMove, check further canditates
                    continue
                else:
                    # if further canditates also fail, go to debug mode ...
                    import pdb
                    pdb.set_trace()

            xCanditate = xCanditate[validCandidatesIndex]
            yCanditate = yCanditate[validCandidatesIndex]
            zCanditate = zCanditate[validCandidatesIndex]
            yawCanditate = yawCanditate[validCandidatesIndex]

            # print(f"{self.getName()} type(xCanditate)={xCanditate}")
            jEstimated = self.estimate(xCanditate, yCanditate, np.radians(yawCanditate))

            # print(f"{self.getName()} jEstimated={jEstimated}")

            return jEstimated,xCanditate,yCanditate,zCanditate,yawCanditate


    def moveOmniDirectional(self, randomPointsSize=70, maxTravelTime=5.,
                            minDist=5., plotEstimator=True, maxYaw=15.,
                            lidar=False, controllers=[], saveLidar=False):

        if lidar:
            canditatesData = self.getCanditatesLidar(randomPointsSize=randomPointsSize,
                                                     maxTravelTime=maxTravelTime,
                                                     minDist=minDist,
                                                     maxYaw=maxYaw,
                                                     controllers=controllers,
                                                     saveLidar=saveLidar)
        else:
            canditatesData = self.getCanditates(randomPointsSize=randomPointsSize,
                                                maxTravelTime=maxTravelTime,
                                                minDist=minDist,
                                                maxYaw=maxYaw)

        jEstimated, xCanditateList, yCanditateList, zCanditateList, yawCanditateList = canditatesData

        tartgetPointIndex = np.argmax(jEstimated)
        # tartgetPointIndex = np.argmin(jEstimated)

        task = self.moveToPositionYawModeAsync(xCanditateList[tartgetPointIndex],
                                   yCanditateList[tartgetPointIndex],
                                   zCanditateList[tartgetPointIndex],
                                   1,
                                   yawmode = yawCanditateList[tartgetPointIndex])

        if plotEstimator:
            self.plotEstimator(xCanditateList, yCanditateList, yawCanditateList, jEstimated)

        if DEBUG_MOVE_OMNI:
            print(f"\n[DEBUG][MOVE_OMNI] ----- {self.getName()} -----")
            print(f"[DEBUG][MOVE_OMNI] target pose (x:{xCanditateList[tartgetPointIndex]:.2f} ,y:{yCanditateList[tartgetPointIndex]:.2f}, z:{zCanditateList[tartgetPointIndex]:.2f}, yaw:{yawCanditateList[tartgetPointIndex]:.2f})")

        return task

    def estimate(self,x,y,yaw):

        if type(x) is np.ndarray:

            yaw[yaw > np.pi] = -np.pi*2 + yaw[yaw > np.pi]
            yaw[yaw < -np.pi] = np.pi*2 - yaw[yaw < -np.pi]

            canditates = np.stack((x,y,yaw),axis=1)
            return self.estimator.predict(canditates)

        else:

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
            print(f"\n[ESTIMATOR] {self.getName()} is using data:{[list(i) for i in dataDegrees[-self.estimatorWindow:]]} and Ji:{self.j_i[-self.estimatorWindow:]}")

        # weights = np.linspace(1,1,len(data[-self.estimatorWindow:]))
        weights = self.estimatorWeights[-len(data):]

        self.estimator = self.model.fit(data[-self.estimatorWindow:],self.j_i[-self.estimatorWindow:], **{'linear__sample_weight': weights})


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

        ax_2 = ax2.quiver(y[-self.estimatorWindow:],x[-self.estimatorWindow:],v[-self.estimatorWindow:],
                   u[-self.estimatorWindow:],ji[-self.estimatorWindow:], cmap=plt.cm.seismic)
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


    def updateState(self, posIdx, timeStep, addInList=False):

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

        if addInList:
            self.stateList.append([self.state, self.cameraInfo])


    def detectObjects(self, save_detected=SAVE_DETECTED):

        # detector = yoloDetector.yoloDetector()

        detected_file_name = None
        if save_detected:
            detected_file_name = os.path.join(self.detected_dir,
                                              f"detected_time_{self.timeStep}.png")

        detections = self.detector.detect(self.imageScene, display=False, save=detected_file_name)

        # detections = {'cars':[(pixel_x,pixel_y,detecions_id,confidece,width, height),(pixel_x,pixel_y,detecions_id,confidece), ...]}
        pixelX = []
        pixelY = []
        width = []
        height = []
        # scoreNum -> number of detected objects (int)
        scoreNum = 0
        detection_copy = {}
        for detectionClass, objects in detections.items():
            detection_copy[detectionClass] = []
            for object in objects:
                # object = (pixel_x,pixel_y,detecions_id,confidece)
                # score += object[3]*WEIGHTS[detectionClass]
                pixelX.append(object[0])
                pixelY.append(object[1])
                width.append(object[4])
                height.append(object[5])
                detection_copy[detectionClass].append((object[0],object[1],object[2],object[3]))
                # scoreNum += WEIGHTS[detectionClass]

        # XXX: bad technique here, but width and height were added on later stages,
        # there was no compatibility. Defenitely we must improve that
        # detection_copy = {}
        # for detectionClass, objects in detections.items():
        #     detection_copy[detectionClass] = []
        #     for object in objects:
        #         detection_copy[detectionClass].append(object[0:4])
        # detections = detection_copy.copy()

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
        self.detectionsWidthHeight.append([pixelX, pixelY, width,height])
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


    def setExcludedList(self, excludedList=[]):
        self.excludedList.append(excludedList)


    def storeInformationJ(self, KW=1., detectionsDict={} ):

        score = self.scoreExcludingDetections(excludedList=self.excludedList[-1], minusDuplicates=False)

        closestDetection = 0.

        if score == 0.:
            detectionsCoordinates = self.getDetectionsCoordinates()
            closestDetection = self.getDistanceClosestDetection(detectionsDict)

        update = score - KW*closestDetection

        self.informationJ.append(score)


    def getInformationJ(self,index=-1):
        return self.informationJ[index]


    def appendInforamtionJi(self,update):
        self.informationJi.append(update)


    def getinformationJi(self, index=-1):
        return self.informationJi[index]


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

        # self.cameraInfo.fov = max(120,self.cameraInfo.fov)

        self.cameraInfoFront = self.client.simGetCameraInfo(1, vehicle_name=self.name)
        self.cameraInfoFront.pose.position.x_val += self.offSetX
        self.cameraInfoFront.pose.position.y_val += self.offSetY
        self.cameraInfoFront.pose.position.z_val += self.offSetZ

        self.cameraInfoBack = self.client.simGetCameraInfo(4, vehicle_name=self.name)
        self.cameraInfoBack.pose.position.x_val += self.offSetX
        self.cameraInfoBack.pose.position.y_val += self.offSetY
        self.cameraInfoBack.pose.position.z_val += self.offSetZ


    def getPositions(self, index=None):

        if index is None:

            return self.position

        else:

            stateList = self.getStateList()
            stateDrone, _ = stateList[index]

            return stateDrone.kinematics_estimated.position


    def getTimeStep(self):
        return self.timeStep


    def getWayPoint(self):
        return self.posIdx


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

        detections_file = os.path.join(self.parentRaw,
                                       self.getName(), f"detectionsWidthHeight_{self.name}.pickle")
        pickle.dump(self.detectionsWidthHeight,open(detections_file,"wb"))

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
