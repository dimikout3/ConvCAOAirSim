# import setup_path
import airsim

class controller:

    def __init__(self, clientIn, droneName):

        self.client = clientIn
        self.name = droneName

        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

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

    def getImages(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True),  #depth visualization image
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        return responses

    def getPose(self):
        return self.client.simGetVehiclePose(vehicle_name=self.name)


    def getState(self):
        return self.client.getMultirotorState(vehicle_name=self.name)


    def quit(self):

        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)
