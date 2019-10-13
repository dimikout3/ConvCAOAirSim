import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import utils

POSE_INITIAL = [60, -30]
# Enters all directories and creates 3d plots (saves them as pickle objects)

if __name__ == "__main__":

    parent_dir = os.path.join(os.getcwd(),"..", "swarm_raw_output")

    results_dir = os.path.join(os.getcwd(),"..", "results")
    try:
        os.makedirs(results_dir)
    except OSError:
        if not os.path.isdir(results_dir):
            raise

    imagesEnv3D_dir = os.path.join(results_dir, "images_enviroment_3D")
    try:
        os.makedirs(imagesEnv3D_dir)
    except OSError:
        if not os.path.isdir(imagesEnv3D_dir):
            raise
    imageIndex = 0

    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(parent_dir, dronesID[0]))

    xAggregated = np.array([])
    yAggregated = np.array([])
    zAggregated = np.array([])
    colorsAggregated = np.array([[0,0,0]])

    for positionIdx, position in enumerate(wayPointsID):
        print(f"{4*' '}[POSITION]: position_{positionIdx}")

        for droneIdx, drone in enumerate(dronesID):
            print(f"[DRONE]: {drone}")

            current_dir = os.path.join(parent_dir, drone, f"position_{positionIdx}")
            coordinates_pickle = os.path.join(current_dir, "coordinates3D.pickle")

            coordinates = pickle.load(open(coordinates_pickle,"rb"))

            xAggregated = np.concatenate((xAggregated,coordinates[3]))
            yAggregated = np.concatenate((yAggregated,coordinates[4]))
            zAggregated = np.concatenate((zAggregated,coordinates[5]))
            colorsAggregated = np.concatenate((colorsAggregated,coordinates[6]))

            image_name = os.path.join(imagesEnv3D_dir, f"image_{imageIndex}.png")
            utils.plot3dColor(xAggregated,yAggregated,zAggregated,colorsAggregated[1:],
                              x_lim=[200,-200], pose=[POSE_INITIAL[0],POSE_INITIAL[1]],
                              y_lim=[-200,250], save_path=image_name, z_lim=[0,-100])
            imageIndex += 1

    #rotate azimuth
    print("Rotaing by azimuth")
    for i in range(POSE_INITIAL[1],POSE_INITIAL[1]+180,10):
        image_name = os.path.join(imagesEnv3D_dir, f"image_{imageIndex}.png")
        utils.plot3dColor(xAggregated,yAggregated,zAggregated,colorsAggregated[1:],
                          x_lim=[200,-200], pose=[POSE_INITIAL[0],i],
                          y_lim=[-200,250], save_path=image_name,z_lim=[0,-100])
        imageIndex += 1
    #rotate elevation
    print("Rotaing by elevation")
    for i in range(POSE_INITIAL[0],90,10):
        image_name = os.path.join(imagesEnv3D_dir, f"image_{imageIndex}.png")
        utils.plot3dColor(xAggregated,yAggregated,zAggregated,colorsAggregated[1:],
                          x_lim=[200,-200], pose=[i,POSE_INITIAL[1]],
                          y_lim=[-200,250], save_path=image_name, z_lim=[0,-100])
        imageIndex += 1

    image_random_path = os.path.join(imagesEnv3D_dir, "image_0.png")
    image_random = cv2.imread(image_random_path)
    w,h,_ = image_random.shape
    print(f"Images have width:{w} and height:{h}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"enviroment_3D.avi", fourcc, 1.0, (h,w))

    images_3D = os.listdir(imagesEnv3D_dir)
    print(f"Images for video are {images_3D}")

    for i,image in enumerate(images_3D):

        image_path = os.path.join(imagesEnv3D_dir, f"image_{i}.png")
        frame = cv2.imread(image_path)
        out.write(frame)

    out.release()

    coordinates_aggregated = os.path.join(results_dir, "coordinates3D_Aggregated.pickle")
    coordinates_data = [xAggregated, yAggregated, zAggregated, colorsAggregated[1:]]
    pickle.dump(coordinates_data,open(coordinates_aggregated,"wb"))
