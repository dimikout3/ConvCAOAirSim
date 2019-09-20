# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import cv2
import os
import itertools

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

raw_output_dir = os.path.join(os.getcwd(), "swarm_raw_output")
detected_output_dir = os.path.join(os.getcwd(), "swarm_detected")

dronesID = os.listdir(raw_output_dir)
wayPointsID = os.listdir(os.path.join(raw_output_dir, dronesID[0]))
wayPointsSize = len(wayPointsID)
timeStepsID = os.listdir(os.path.join(raw_output_dir, dronesID[0],wayPointsID[0]))
timeStepsSize = len(timeStepsID)

report_file = open(os.path.join(detected_output_dir, "report_similarity.txt"),"w+")


for positionInx, position in enumerate(wayPointsID):
    print(f"[POSITION]: {position}", file=report_file)

    for imageIdx,imageID in enumerate(timeStepsID):
        print(f"{2*' '}[TIME]: {imageIdx} -- [IMAGE]: {imageID}", file=report_file)

        # dronesComb -> [("drone1,drone2"),("drones1","drones3") ...]
        for dronesComb in itertools.combinations(dronesID,2):

            image1 = cv2.imread(os.path.join(raw_output_dir, dronesComb[0], position, imageID))
            image2 = cv2.imread(os.path.join(raw_output_dir, dronesComb[1], position, imageID))

            good_points = compareImages(image1, image2)

            print(f"{4*' '} {dronesComb[0]} - {dronesComb[1]} have {good_points} good points (similarity)", file=report_file)

report_file.close()
