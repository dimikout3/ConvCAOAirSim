import airsim
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Enters all directories and creates 3d plots (saves them as pickle objects)
# and pickle object with the x,y,z,colors data (relative and absolute)

poisitionsSize = 50

def getKPIpng(position):

    singVector = np.load(os.path.join(os.getcwd(),"results", "singleVector.npy"))
    hull = np.load(os.path.join(os.getcwd(),"results", "convexHull.npy"))
    distRandom = np.load(os.path.join(os.getcwd(),"results", "dist_avg.npy"))
    distIdeal = np.load(os.path.join(os.getcwd(),"results", "dist_avg_ideal.npy"))
    sift = np.load(os.path.join(os.getcwd(),"results", "good_points.npy"))

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # x = [i for i in range(poisitionsSize)]


    y = singVector[0,0:position]
    x = [i for i in range(position)]
    ax1.plot(x,y)
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Average Distance [Vector]")
    ax1.set_ylim([0, np.max(singVector)])
    ax1.set_xlim([0,poisitionsSize])

    y = hull[0,0:position,0]
    x = [i for i in range(position)]
    ax2.plot(x,y)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Number of points inside ConvexHull")
    ax2.set_ylim([0, np.max(hull)])
    ax2.set_xlim([0,poisitionsSize])

    y = distRandom[0,0:position]
    x = [i for i in range(position)]
    ax3.plot(x,y,label="Random")
    y = distIdeal[0,0:position]
    ax3.plot(x,y,'r',label="Exhaustive")
    ax3.legend()
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Average Distance")
    ax3.set_ylim([0, max(np.max(distRandom),np.max(distIdeal))])
    ax3.set_xlim([0,poisitionsSize])

    y = sift[0,0:position]
    x = [i for i in range(position)]
    ax4.plot(x,y)
    ax4.set_xlabel("Position")
    ax4.set_ylabel("Number of Good Points")
    ax4.set_ylim([0, np.max(sift)])
    ax4.set_xlim([0,poisitionsSize])

    plt.tight_layout()

    file_name = os.path.join(os.getcwd(),"results","similarity_images",
                             f"KPI_position{position}.png")
    plt.savefig(file_name)

    plt.close()
    return file_name


if __name__ == "__main__":

    parent_dir = os.path.join(os.getcwd(), "swarm_raw_output")
    detected_dir = os.path.join(os.getcwd(), "swarm_detected")

    results_dir = os.path.join(os.getcwd(),"results")
    try:
        os.makedirs(results_dir)
    except OSError:
        if not os.path.isdir(results_dir):
            raise

    dronesID = os.listdir(parent_dir)
    wayPointsID = os.listdir(os.path.join(parent_dir, dronesID[0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    file_out = os.path.join(results_dir,f"videoSimilarity.avi")
    out = cv2.VideoWriter(file_out, fourcc, 1.0, (2000,2000))

    for positionIdx, position in enumerate(wayPointsID):
        print(f"{4*' '}[POSITION]: position_{positionIdx}")

        current_detected_dir = os.path.join(detected_dir, dronesID[0], f"position_{positionIdx}")
        time_steps = os.listdir(current_detected_dir)

        for time,time_step in enumerate(time_steps):

            current_raw_dir = os.path.join(parent_dir, dronesID[0], f"position_{positionIdx}")
            raw_image_path1 = os.path.join(current_raw_dir, f"scene_time_{time}.png")

            current_raw_dir = os.path.join(parent_dir, dronesID[1], f"position_{positionIdx}")
            raw_image_path2 = os.path.join(current_raw_dir, f"scene_time_{time}.png")

            frame_raw1 = cv2.imread(raw_image_path1)
            frame_raw1_resized = cv2.resize(frame_raw1,(1000,1000))
            frame_raw2 = cv2.imread(raw_image_path2)
            frame_raw2_resized = cv2.resize(frame_raw2,(1000,1000))

            frame_top = np.concatenate((frame_raw1_resized,frame_raw2_resized), axis=1)
            # h, w, _ = frame_top.shape
            # print(f" frame_top width:{w} -- height:{h}")

            plt_KPI = getKPIpng(positionIdx)
            frame_kpi = cv2.imread(plt_KPI)
            frame_kpi_resized = cv2.resize(frame_kpi,(2000,1000))

            # frame = frame_top + plt_KPI
            frame_top = np.concatenate((frame_top,frame_kpi_resized), axis=0)

            out.write(frame_top)

    out.release()
