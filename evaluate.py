# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


raw_output_dir = os.path.join(os.getcwd(), "swarm_raw_output")
detected_output_dir = os.path.join(os.getcwd(), "swarm_detected")
try:
    os.makedirs(detected_output_dir)
except OSError:
    if not os.path.isdir(detected_output_dir):
        raise

dronesID = os.listdir(raw_output_dir)
wayPointsID = os.listdir(os.path.join(raw_output_dir, dronesID[0]))
wayPointsSize = len(wayPointsID)
timeStepsID = os.listdir(os.path.join(raw_output_dir, dronesID[0],wayPointsID[0]))
timeStepsSize = len(timeStepsID)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.join(os.getcwd(),"yolo-coco", "coco.names")
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.join(os.getcwd(),"yolo-coco", "yolov3.weights")
configPath = os.path.join(os.getcwd(),"yolo-coco", "yolov3.cfg")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

for droneInx, drone in enumerate(dronesID):
    print(f"\n[DRONE]: {drone}")
    for position in range(0,wayPointsSize):
        print(f"{4*' '}[POSITION]: {position}")
        # path to all images
        images_raw_dir = os.path.join(raw_output_dir,drone, f"position_{position}")

        #list with all the images ["type_0_time_0.png", "type_1_time_3.png" ...]
        imagesList = os.listdir(images_raw_dir)

        images_output_dir = os.path.join(detected_output_dir,drone, f"position_{position}")
        try:
            os.makedirs(images_output_dir)
        except OSError:
            if not os.path.isdir(images_output_dir):
                raise

        for imageIdx,imageID in tqdm(enumerate(imagesList), desc="Image"):

            n_cars = np.zeros((len(dronesID),wayPointsSize,timeStepsSize))
            conf_cars = np.zeros((len(dronesID),wayPointsSize,timeStepsSize))
            n_persons = np.zeros((len(dronesID),wayPointsSize,timeStepsSize))
            conf_persons = np.zeros((len(dronesID),wayPointsSize,timeStepsSize))
            n_trafficLights = np.zeros((len(dronesID),wayPointsSize,timeStepsSize))
            conf_trafficLights = np.zeros((len(dronesID),wayPointsSize,timeStepsSize))

            image = cv2.imread(os.path.join(images_raw_dir,imageID))
            (H, W) = image.shape[:2]

            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            	swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            # show timing information on YOLO
            # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
            	# loop over each of the detections
            	for detection in output:
            		# extract the class ID and confidence (i.e., probability) of
            		# the current object detection
            		scores = detection[5:]
            		classID = np.argmax(scores)
            		confidence = scores[classID]

            		# filter out weak predictions by ensuring the detected
            		# probability is greater than the minimum probability
            		if confidence > args["confidence"]:
            			# scale the bounding box coordinates back relative to the
            			# size of the image, keeping in mind that YOLO actually
            			# returns the center (x, y)-coordinates of the bounding
            			# box followed by the boxes' width and height
            			box = detection[0:4] * np.array([W, H, W, H])
            			(centerX, centerY, width, height) = box.astype("int")

            			# use the center (x, y)-coordinates to derive the top and
            			# and left corner of the bounding box
            			x = int(centerX - (width / 2))
            			y = int(centerY - (height / 2))

            			# update our list of bounding box coordinates, confidences,
            			# and class IDs
            			boxes.append([x, y, int(width), int(height)])
            			confidences.append(float(confidence))
            			classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            	args["threshold"])

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

                    if LABELS[classIDs[i]] == "car":
                        n_cars[droneInx][position][imageIdx] += 1
                    elif LABELS[classIDs[i]] == "person":
                        n_persons[droneInx][position][imageIdx] += 1
                    elif LABELS[classIDs[i]] == "traffic light":
                        n_trafficLights[droneInx][position][imageIdx] += 1


            print(f"Drone: {drone} - Position: {position} - Time: {imageIdx}")
            print(f"{2*' '}Number of cars detected: {n_cars[droneInx][position][imageIdx]}")
            print(f"{2*' '}Number of persons detected: {n_persons[droneInx][position][imageIdx]}")
            print(f"{2*' '}Number of traffic lights detected: {n_trafficLights[droneInx][position][imageIdx]}")

            # show the output image
            # cv2.imshow("Image", image)
            plt.imshow(image)
            # plt.show()
            plt.savefig(os.path.join(images_output_dir,imageID))
            plt.close()
            # cv2.waitKey(0)
