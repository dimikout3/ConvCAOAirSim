# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import cv2
import os
import itertools
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import time
import sys
from scipy.spatial import Delaunay

def compareImages(image1, image2):

    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(image1, None)
    kp_2, desc_2 = sift.detectAndCompute(image2, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_p = []
    ratio = 0.6
    for m, n in matches:
    	if m.distance < ratio*n.distance:
    		good_p.append(m)

    return len(good_p)


def compareCloudsDist(cloud1, cloud2):

    x1,y1,z1 = cloud1[3], cloud1[4], cloud1[5]
    x2,y2,z2 = cloud2[3], cloud2[4], cloud2[5]

    if x1.size < x2.size:
        minSize = x1.size
        x2 = x2[0:minSize]
        y2 = y2[0:minSize]
        z2 = z2[0:minSize]
    else:
        minSize = x2.size
        x1 = x1[0:minSize]
        y1 = y1[0:minSize]
        z1 = z1[0:minSize]

    dist_array = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    dist_sum = np.sum(dist_array)
    dist_average = np.average(dist_array)

    return dist_sum, dist_average


def compareCloudsIdeal(cloud1, cloud2):

    x1,y1,z1 = cloud1[3], cloud1[4], cloud1[5]
    x2,y2,z2 = cloud2[3], cloud2[4], cloud2[5]

    if x1.size < x2.size:
        minSize = x1.size
        x2 = x2[0:minSize]
        y2 = y2[0:minSize]
        z2 = z2[0:minSize]
    else:
        minSize = x2.size
        x1 = x1[0:minSize]
        y1 = y1[0:minSize]
        z1 = z1[0:minSize]

    x1v = np.vstack(x1)
    x1m = np.tile(x1v,(1,minSize))
    y1v = np.vstack(y1)
    y1m = np.tile(y1v,(1,minSize))
    z1v = np.vstack(z1)
    z1m = np.tile(z1v,(1,minSize))

    x2m = np.tile(x2,(minSize,1))
    y2m = np.tile(y2,(minSize,1))
    z2m = np.tile(z2,(minSize,1))

    x = x1m - x2m
    y = y1m - y2m
    z = z1m - z2m

    dist_array = np.sqrt(x**2 + y**2 + z**2)

    dist_array_ideal = np.min(dist_array, axis=0)

    dist_sum = np.sum(dist_array_ideal)
    dist_average = np.average(dist_array_ideal)

    return dist_sum, dist_average


def plotKPI(npValues,title="NO_TITLE_GIVEN"):

    combos, positions, time = npValues.shape

    x = [i for i in range(0,positions)]

    for c in range(0,combos):
        for t in range(0,time):

            y = [npValues[c,i,t] for i in range(0,positions)]

            plt.plot(x,y)
            plt.title(title)
            plt.xlabel("Positons")
            plt.ylabel("KPI Value")

            # plt.show()
            plt_dir = os.path.join(os.getcwd(),"results", f"{title}_comb_{c}_time_{t}.png")
            plt.savefig(plt_dir)
            plt.close()


def compareDivergence(cloud1, cloud2):
    x1,y1,z1 = cloud1[3], cloud1[4], cloud1[5]
    x2,y2,z2 = cloud2[3], cloud2[4], cloud2[5]

    if x1.size < x2.size:
        minSize = x1.size
        x2 = x2[0:minSize]
        y2 = y2[0:minSize]
        z2 = z2[0:minSize]
    else:
        minSize = x2.size
        x1 = x1[0:minSize]
        y1 = y1[0:minSize]
        z1 = z1[0:minSize]

    x = x1 - x2
    y = y1 - y2
    z = z1 - z2

    field = np.stack((x,y,z),axis=1)

    gradient = np.abs(np.gradient(field,axis=0))
    divergence = np.sum(gradient)

    return divergence


def compareSingleVector(cloud1, cloud2):
    x1,y1,z1 = cloud1[3], cloud1[4], cloud1[5]
    x2,y2,z2 = cloud2[3], cloud2[4], cloud2[5]

    if x1.size < x2.size:
        minSize = x1.size
        x2 = x2[0:minSize]
        y2 = y2[0:minSize]
        z2 = z2[0:minSize]
    else:
        minSize = x2.size
        x1 = x1[0:minSize]
        y1 = y1[0:minSize]
        z1 = z1[0:minSize]

    x = x1 - x2
    y = y1 - y2
    z = z1 - z2

    xSingle = np.sum(x)
    ySingle = np.sum(y)
    zSingle = np.sum(z)

    magnitude = np.sqrt(xSingle**2 + ySingle**2 + zSingle**2)

    return magnitude


def compareConvexHull(cloud1, cloud2):
    x1,y1,z1 = cloud1[3], cloud1[4], cloud1[5]
    x2,y2,z2 = cloud2[3], cloud2[4], cloud2[5]

    if x1.size < x2.size:
        minSize = x1.size
        x2 = x2[0:minSize]
        y2 = y2[0:minSize]
        z2 = z2[0:minSize]
    else:
        minSize = x2.size
        x1 = x1[0:minSize]
        y1 = y1[0:minSize]
        z1 = z1[0:minSize]

    hullPoints = np.stack((x1,y1,z1),axis=1)
    checkPoints = np.stack((x2,y2,z2),axis=1)

    hull = Delaunay(hullPoints)

    insidePoints = sum(hull.find_simplex(checkPoints)>=0)

    return insidePoints


raw_output_dir = os.path.join(os.getcwd(), "swarm_raw_output")
detected_output_dir = os.path.join(os.getcwd(), "swarm_detected")

dronesID = os.listdir(detected_output_dir)
wayPointsID = os.listdir(os.path.join(detected_output_dir, dronesID[0]))
wayPointsSize = len(wayPointsID)
timeStepsID = os.listdir(os.path.join(detected_output_dir, dronesID[0],wayPointsID[0]))
timeStepsSize = len(timeStepsID)
combinations = list(itertools.combinations(dronesID,2))
combinationsSize = len(combinations)

goodPointsAggregated = np.zeros((combinationsSize,wayPointsSize,timeStepsSize))
distSumAggregated = np.zeros((combinationsSize,wayPointsSize,timeStepsSize))
distAvgAggregated = np.zeros((combinationsSize,wayPointsSize,timeStepsSize))
distSumIdealAggregated = np.zeros((combinationsSize,wayPointsSize,timeStepsSize))
distAvgIdealAggregated = np.zeros((combinationsSize,wayPointsSize,timeStepsSize))
divergenceAggregated =  np.zeros((combinationsSize,wayPointsSize,timeStepsSize))
singleVectorAggregated =  np.zeros((combinationsSize,wayPointsSize,timeStepsSize))
convexHullAggregated =  np.zeros((combinationsSize,wayPointsSize,timeStepsSize))


report_file = open(os.path.join(os.getcwd(),"results", "report_similarity.txt"),"w+")

for positionIdx, position in enumerate(tqdm(wayPointsID, desc="Postion")):
    print(f"[POSITION]: position_{positionIdx}", file=report_file)

    for imageIdx,imageID in enumerate(timeStepsID):
        print(f"{2*' '}[TIME]: {imageIdx} -- [IMAGE]: scene_time_{imageIdx}", file=report_file)

        # dronesComb -> [("drone1,drone2"),("drones1","drones3") ...]
        for comboIdx,dronesComb in enumerate(itertools.combinations(dronesID,2)):

            image1 = cv2.imread(os.path.join(raw_output_dir, dronesComb[0],
                                f"position_{positionIdx}", f"scene_time_{imageIdx}.png"))
            image2 = cv2.imread(os.path.join(raw_output_dir, dronesComb[1],
                                f"position_{positionIdx}", f"scene_time_{imageIdx}.png"))

            file_1 = os.path.join(raw_output_dir, dronesComb[0],
                                  f"position_{positionIdx}", "coordinates3D.pickle")
            pointCloud1 = pickle.load(open(file_1,"rb"))
            file_2 = os.path.join(raw_output_dir, dronesComb[1],
                                  f"position_{positionIdx}", "coordinates3D.pickle")
            pointCloud2 = pickle.load(open(file_2,"rb"))

            print(f"{4*' '}Combination: {dronesComb[0]} - {dronesComb[1]}", file=report_file)

            if not ("--noSift" in sys.argv):
                start_time = time.time()
                good_points = compareImages(image1, image2)
                timeSift = time.time() - start_time
                goodPointsAggregated[comboIdx,positionIdx,imageIdx] = good_points
                print(f"{4*' '}  {8*' '} similarity {good_points}[#], {timeSift:.4f}[sec]", file=report_file)

            if not ("--noDistRandom" in sys.argv):
                start_time = time.time()
                clouds_dist_sum, clouds_dist_average = compareCloudsDist(pointCloud1, pointCloud2)
                timeDistRandom = time.time() - start_time
                print(f"{4*' '} {8*' '} cloud's dist aggregated {clouds_dist_sum:.2f}, {timeDistRandom:.4f}[sec]", file=report_file)
                print(f"{4*' '} {8*' '} cloud's dist average {clouds_dist_average:.2f}, {timeDistRandom:.4f}[sec]", file=report_file)
                distSumAggregated[comboIdx,positionIdx,imageIdx] = clouds_dist_sum
                distAvgAggregated[comboIdx,positionIdx,imageIdx] = clouds_dist_average

            if not ("--noDistIdeal" in sys.argv):
                start_time = time.time()
                clouds_dist_sum_ideal, clouds_dist_average_ideal = compareCloudsIdeal(pointCloud1, pointCloud2)
                timeDistIdeal = time.time() - start_time
                print(f"{4*' '} {8*' '} cloud's dist aggregated ideal {clouds_dist_sum_ideal:.2f}, {timeDistIdeal:.4f}[sec]", file=report_file)
                print(f"{4*' '} {8*' '} cloud's dist average ideal {clouds_dist_average_ideal:.2f}, {timeDistIdeal:.4f}[sec]", file=report_file)
                distSumIdealAggregated[comboIdx,positionIdx,imageIdx] = clouds_dist_sum_ideal
                distAvgIdealAggregated[comboIdx,positionIdx,imageIdx] = clouds_dist_average_ideal

            if not ("--noDivergence" in sys.argv):
                start_time = time.time()
                divergenceValue = compareDivergence(pointCloud1, pointCloud2)
                timeDivergence = time.time() - start_time
                print(f"{4*' '} {8*' '} cloud's divergence {divergenceValue:.2f}, {timeDivergence:.4f}[sec]", file=report_file)
                divergenceAggregated[comboIdx,positionIdx,imageIdx] = divergenceValue

            if not ("--noSingleVector" in sys.argv):
                start_time = time.time()
                singleVector = compareSingleVector(pointCloud1, pointCloud2)
                timeSingleVector = time.time() - start_time
                print(f"{4*' '} {8*' '} cloud's divergence {singleVector:.2f}, {timeSingleVector:.4f}[sec]", file=report_file)
                singleVectorAggregated[comboIdx,positionIdx,imageIdx] = singleVector

            if not ("--noConvexHull" in sys.argv):
                start_time = time.time()
                pointsInHull = compareConvexHull(pointCloud1, pointCloud2)
                timeHull = time.time() - start_time
                print(f"{4*' '} {8*' '} points in hull {pointsInHull:.2f}, {timeHull:.4f}[sec]", file=report_file)
                convexHullAggregated[comboIdx,positionIdx,imageIdx] = pointsInHull


report_file.close()

if not ("--noSift" in sys.argv):
    np.save(os.path.join(os.getcwd(),"results", "good_points.npy"),goodPointsAggregated)
    plotKPI(goodPointsAggregated,title="GoodPoints")

if not ("--noDistRandom" in sys.argv):
    plotKPI(distSumAggregated,title="DistSum")
    plotKPI(distAvgAggregated,title="DistAvg")
    np.save(os.path.join(os.getcwd(),"results", "dist_sum.npy"),distSumAggregated)
    np.save(os.path.join(os.getcwd(),"results", "dist_avg.npy"),distAvgAggregated)

if not ("--noDistIdeal" in sys.argv):
    plotKPI(distSumIdealAggregated,title="DistSumIdeal")
    plotKPI(distAvgIdealAggregated,title="DistAvgIdeal")
    np.save(os.path.join(os.getcwd(),"results", "dist_sum_ideal.npy"),distSumIdealAggregated)
    np.save(os.path.join(os.getcwd(),"results", "dist_avg_ideal.npy"),distAvgIdealAggregated)

if not ("--noDivergence" in sys.argv):
    plotKPI(divergenceAggregated,title="Divergence")
    np.save(os.path.join(os.getcwd(),"results", "divergenceValue.npy"),divergenceAggregated)

if not ("--noSingleVector" in sys.argv):
    plotKPI(singleVectorAggregated,title="SingleVector")
    np.save(os.path.join(os.getcwd(),"results", "singleVector.npy"),singleVectorAggregated)

if not ("--noConvexHull" in sys.argv):
    plotKPI(convexHullAggregated,title="ConvexHull")
    np.save(os.path.join(os.getcwd(),"results", "convexHull.npy"),convexHullAggregated)
