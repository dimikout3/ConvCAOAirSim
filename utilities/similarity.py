# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import os
import itertools
import pickle
import matplotlib.pyplot as plt
import time


THRESHOLD_DIST = 2.5

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


def compareExhaustiveDuplicates(detectionsData1, detectionsData2):

    # indices that will be excluded because they are the same object detected
    excluded1 = []
    excluded2 = []

    info1 = detectionsData1[1]
    info2 = detectionsData2[1]

    coordinates1 = detectionsData1[0]
    coordinates2 = detectionsData2[0]

    if (coordinates1.size == 0) or (coordinates2.size == 0) :
        return excluded1,excluded2

    x1,y1,z1 = coordinates1[:,0], coordinates1[:,1], coordinates1[:,2]
    x2,y2,z2 = coordinates2[:,0], coordinates2[:,1], coordinates2[:,2]

    x1v = np.vstack(x1)
    x1m = np.tile(x1v,(1,x2.size))
    y1v = np.vstack(y1)
    y1m = np.tile(y1v,(1,y2.size))
    z1v = np.vstack(z1)
    z1m = np.tile(z1v,(1,z2.size))

    x2m = np.tile(x2,(x1.size,1))
    y2m = np.tile(y2,(y1.size,1))
    z2m = np.tile(z2,(z1.size,1))

    x = x1m - x2m
    y = y1m - y2m
    z = z1m - z2m

    dist_array = np.sqrt(x**2 + y**2 + z**2)

    # while dist_array.size != 0:
    for _ in range(dist_array.size):

        x,y = np.unravel_index(dist_array.argmin(), dist_array.shape)
        # x-> index of drone1 detections, y-> index of drone2 detections

        if dist_array[x,y] < THRESHOLD_DIST:
            # check if they are the same time
            if info1[x][0].split("_")[0] == info2[y][0].split("_")[0]:
                if info1[x][1] > info2[y][1]:
                    excluded2.append(y)
                else:
                    excluded1.append(x)

        else:
            break

        # dist_array = np.delete(dist_array,(x),axis=0)
        # dist_array = np.delete(dist_array,(y),axis=1)
        # instead of deleting put big values
        dist_array[x,y] =  THRESHOLD_DIST*2

    return excluded1, excluded2


def compareRandomDist(cloud1, cloud2):

    x1,y1,z1 = cloud1[0], cloud1[1], cloud1[2]
    x2,y2,z2 = cloud2[0], cloud2[1], cloud2[2]

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


def similarityOut(dataPoints, similarityKPI = None, ip = 0):

    report_file = open(os.path.join(os.getcwd(),f"results_{ip}", "report_similarity.txt"),"w+")

    droneIDs = list(dataPoints.keys())
    combos = list(itertools.combinations(droneIDs,2))

    excludedDict = {id:set() for id in droneIDs}

    # dronesComb -> [("drone1,drone2"),("drones1","drones3") ...]
    simSum = []
    simAvg = []

    for comboIdx,dronesComb in enumerate(itertools.combinations(droneIDs,2)):

        print(f"{4*' '}Combination: {dronesComb[0]} - {dronesComb[1]}", file=report_file)

        if similarityKPI == "DistExhaustive":

            detectionsData1 = dataPoints[dronesComb[0]]
            detectionsData2 = dataPoints[dronesComb[1]]

            start_time = time.time()
            excluded1, excluded2 = compareExhaustiveDuplicates(detectionsData1, detectionsData2)
            timeDistIdeal = time.time() - start_time

            excludedDict[dronesComb[0]].update(excluded1)
            excludedDict[dronesComb[1]].update(excluded2)

            # simSum.append(clouds_dist_sum_ideal)
            # simAvg.append(clouds_dist_average_ideal)

            # print(f"{4*' '} {8*' '} objects dist aggregated exhuastive {clouds_dist_sum_ideal:.2f}, {timeDistIdeal:.4f}[sec]", file=report_file)
            # print(f"{4*' '} {8*' '} objects dist average exhuastive {clouds_dist_average_ideal:.2f}, {timeDistIdeal:.4f}[sec]", file=report_file)

        if similarityKPI == "DistRandom":

            pointCloud1 = dataPoints[dronesComb[0]]
            pointCloud2 = dataPoints[dronesComb[1]]

            start_time = time.time()
            clouds_dist_sum_ideal, clouds_dist_average_ideal = compareRandomDist(pointCloud1, pointCloud2)
            timeDistIdeal = time.time() - start_time

            simSum.append(clouds_dist_sum_ideal)
            simAvg.append(clouds_dist_average_ideal)

            print(f"{4*' '} {8*' '} objects dist aggregated exhuastive {clouds_dist_sum_ideal:.2f}, {timeDistIdeal:.4f}[sec]", file=report_file)
            print(f"{4*' '} {8*' '} objects dist average exhuastive {clouds_dist_average_ideal:.2f}, {timeDistIdeal:.4f}[sec]", file=report_file)

    # return np.sum(simSum), np.average(simAvg)
    return excludedDict
