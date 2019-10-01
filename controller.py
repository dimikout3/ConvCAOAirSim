# import setup_path
import airsim
import os
import cv2
import numpy as np

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

    def getImages(self, save_raw=None):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True),  #depth visualization image
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
            vehicle_name = self.name)  #scene vision image in uncompressed RGB array

        if save_raw != None:

            subDir = save_raw[0]
            timeStep = save_raw[1]

            filenameDepth = os.path.join(subDir, f"depth_time_{timeStep}" )
            airsim.write_pfm(os.path.normpath(filenameDepth + '.pfm'), airsim.get_pfm_array(responses[0]))

            filenameScene = os.path.join(subDir, f"scene_time_{timeStep}" )
            img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8) #get numpy array
            img_rgb = img1d.reshape(responses[1].height, responses[1].width, 3) #reshape array to 3 channel image array H X W X 3
            cv2.imwrite(os.path.normpath(filenameScene + '.png'), img_rgb) # write to png



        return responses

    def getPose(self):
        return self.client.simGetVehiclePose(vehicle_name=self.name)


    def getState(self):
        return self.client.getMultirotorState(vehicle_name=self.name)


    def quit(self):

        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)
