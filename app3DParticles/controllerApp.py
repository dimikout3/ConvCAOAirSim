# import setup_path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from controllerBase import controller

import airsim
import numpy as np
from scipy.spatial import distance
import pickle

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

DEBUG_ESTIMATOR = False
DEBUG_ESTIMATE = True
DEBUG_INSIDE_GEOFENCE = False
DEBUG_CANDITATES = False

class controllerApp(controller):

    def __init__(self, clientIn, droneName, offSets, ip="1",
                 wayPointsSize=200, estimatorWindow=55,
                 maxDistView = 10.0):

        super().__init__(clientIn, droneName, offSets, ip=ip,
                         wayPointsSize = wayPointsSize,
                         estimatorWindow = estimatorWindow,
                         maxDistView = maxDistView)

        self.descretePointCloud = []

        self.attributed = []

        self.resetEstimator()


    def resetEstimator(self, DoF=4):

        # DoF -> Degrees of Freedom (in this App x,y,z,yaw)

        self.model = Pipeline([('poly', PolynomialFeatures(degree=DoF)),
                               ('linear', LinearRegression())])

        self.estimator = self.model.fit([np.random.uniform(0,1,DoF)],[np.random.uniform(0,1)])


    def updateEstimator(self):

        """ Virtual function, redefines here from controlelrBase"""

        xList = [state[0].kinematics_estimated.position.x_val for state in self.stateList]
        yList = [state[0].kinematics_estimated.position.y_val for state in self.stateList]
        zList = [state[0].kinematics_estimated.position.z_val for state in self.stateList]
        yawList = [airsim.to_eularian_angles(state[0].kinematics_estimated.orientation)[2] for state in self.stateList]

        yawListDegrees = [np.degrees(airsim.to_eularian_angles(state[0].kinematics_estimated.orientation)[2]) for state in self.stateList]

        data = np.stack((xList,yList,zList,yawList),axis=1)
        dataDegrees = np.stack((xList,yList,yawListDegrees),axis=1)

        if DEBUG_ESTIMATOR:
            print(f"\n[ESTIMATOR] {self.getName()} is using data:{[list(i) for i in dataDegrees[-self.estimatorWindow:]]} and Ji:{self.j_i[-self.estimatorWindow:]}")

        # weights = np.linspace(1,1,len(data[-self.estimatorWindow:]))
        weights = self.estimatorWeights[-len(data):]

        # import pdb; pdb.set_trace()
        # self.estimator = self.model.fit(data[-self.estimatorWindow:],self.j_i[-self.estimatorWindow:], **{'linear__sample_weight': weights})
        self.estimator = self.model.fit(data[-self.estimatorWindow:],self.j_i[-self.estimatorWindow:])


    def estimate(self, x, y, z, yaw):

        if type(x) is np.ndarray:

            yaw[yaw > np.pi] = -np.pi*2 + yaw[yaw > np.pi]
            yaw[yaw < -np.pi] = np.pi*2 - yaw[yaw < -np.pi]

            canditates = np.stack((x,y,z,yaw),axis=1)

            if DEBUG_ESTIMATE:
                print(f"[ESTIMATE] {self.getName()} canditates.shape={canditates.shape}")

            return self.estimator.predict(canditates)

        else:

            if yaw > np.pi:
                yaw = -np.pi*2 + yaw
            if yaw < -np.pi:
                yaw = np.pi*2 - yaw

            return float(self.estimator.predict([[x,y,z,yaw]]))


    def getForcePoints(self, line_points=10):

        """ Put points between uav camera and final detected pixel """

        x, y, z, colors = self.pointCloud[-1]

        points = np.stack((x,y,z),axis=1)

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val
        uav = np.array([xCurrent, yCurrent, zCurrent])

        lines = np.linspace(points, uav, line_points)
        line_points  = np.reshape(lines, (lines.shape[0]*lines.shape[1],3))

        return line_points


    def insideGeoFence(self, points):

        """ Not needed since Frontier Cells will be inside the GeoFence anyway"""

        inGeoFence = self.geoFence.isInside(points)

        if DEBUG_INSIDE_GEOFENCE:
            print(f"[INSIDE_GEOFENCE] {self.getName()} inGeoFence.size={len(inGeoFence)}")

        return inGeoFence


    def isSafeDist(self, canditates=[], lidarPoints=[], minDist=1.):

        xCurrent = self.state.kinematics_estimated.position.x_val
        yCurrent = self.state.kinematics_estimated.position.y_val
        zCurrent = self.state.kinematics_estimated.position.z_val

        canditatesTrue = []

        # TODO: Avoid this loop if possible

        for ind, (x, y, z) in enumerate(canditates):

            xInter = np.linspace(xCurrent, x ,10)
            yInter = np.linspace(yCurrent, y ,10)
            zInter = np.linspace(zCurrent, z, 10)

            pathPoints = np.stack((xInter, yInter, zInter),axis=1)

            dist = distance.cdist(lidarPoints, pathPoints)

            min = np.min(dist)

            if min>=minDist:

                canditatesTrue.append(ind)

        return canditatesTrue


    def getCanditates(self, pertubations=300, saveLidar=False, minDist = 1.,
                            maxTravelTime=1., maxYaw=15., controllers=[]):

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

            # [-np.pi, np.pi] canditates are inside a shpere with radius=maxTravelTime
            randomOrientation = np.random.uniform(-np.pi, np.pi, pertubations)
            travelTime = np.random.uniform(0., maxTravelTime, pertubations)
            yawCanditate = np.random.uniform(np.degrees(currentYaw) - (maxYaw/2)*a, np.degrees(currentYaw) + (maxYaw/2)*a, pertubations)

            lidarPoints = self.getLidarData(save_lidar=saveLidar)
            lidarPoints = self.clearLidarPoints(lidarPoints=lidarPoints,
                                                maxTravelTime=maxTravelTime,
                                                controllers=controllers)
            lidarPoints = self.addOffsetLidar(lidarPoints=lidarPoints)
            print(f"[LIDAR] {self.getName()} has lidarPoints.shape={lidarPoints.shape}")

            xCanditate = xCurrent + np.cos(randomOrientation)*speedScalar*travelTime*helperIcreasedMove
            yCanditate = yCurrent + np.sin(randomOrientation)*speedScalar*travelTime*a*helperIcreasedMove
            # zCanditate = zCurrent + (np.random.random(pertubations)*2-1)*speedScalar*travelTime
            zCanditate = np.array([zCurrent]).repeat(pertubations)
            canditates = np.stack((xCanditate,yCanditate,zCanditate),axis=1)

            inGeoFence = self.insideGeoFence(canditates)
            isSafeDist = self.isSafeDist(canditates = canditates,
                                         lidarPoints = lidarPoints,
                                         minDist = minDist)

            geoFenceSafe = inGeoFence
            safeDistTrue = isSafeDist

            # import pdb; pdb.set_trace()
            validCandidatesIndex = np.intersect1d(geoFenceSafe, safeDistTrue)

            if validCandidatesIndex.size == 0:
                # something went wrong ...
                if helperIcreasedMove<5.:
                    # increase helperIcreasedMove, check further canditates
                    continue
                else:
                    # if further canditates also fail, go to debug mode ...
                    print(f"[ERROR] {self.getName()}")
                    print(f"    geoFenceSafe.size={len(geoFenceSafe)}")
                    print(f"    safeDistTrue.size={len(safeDistTrue)}")
                    print(f"    validCandidatesIndex.size={len(validCandidatesIndex)}")
                    print(f"    helperIcreasedMove={helperIcreasedMove}")

                    # BUG: it is possible that lidarPoints dont get slim vertical obstacles (DEH)

                    import pdb
                    pdb.set_trace()

            else:
                # There are validate canditates, tehrfore break the searchi process
                break

        if DEBUG_CANDITATES:
            print(f"[CANDITATES] {self.getName()}")
            print(f"    geoFenceSafe.size={len(geoFenceSafe)}")
            print(f"    safeDistTrue.size={len(safeDistTrue)}")
            print(f"    validCandidatesIndex.size={len(validCandidatesIndex)}")
            print(f"    helperIcreasedMove={helperIcreasedMove}")

        xCanditate = xCanditate[validCandidatesIndex]
        yCanditate = yCanditate[validCandidatesIndex]
        zCanditate = zCanditate[validCandidatesIndex]
        yawCanditate = yawCanditate[validCandidatesIndex]

        jEstimated = self.estimate(xCanditate, yCanditate, zCanditate, np.radians(yawCanditate))

        return jEstimated,xCanditate,yCanditate,zCanditate,yawCanditate


    def move(self,controllers=[]):

        canditatesPoints = self.getCanditates(controllers = controllers)

        jEstimated, xCanditateList, yCanditateList, zCanditateList, yawCanditateList = canditatesPoints

        tartgetPointIndex = np.argmax(jEstimated)

        task = self.moveToPositionYawModeAsync(xCanditateList[tartgetPointIndex],
                                               yCanditateList[tartgetPointIndex],
                                               zCanditateList[tartgetPointIndex],
                                               1,
                                               yawmode = yawCanditateList[tartgetPointIndex])

        return task


    def quit(self):

        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)

        state_file = os.path.join(self.parentRaw,
                                  self.getName(), f"state_{self.name}.pickle")
        pickle.dump(self.stateList,open(state_file,"wb"))
