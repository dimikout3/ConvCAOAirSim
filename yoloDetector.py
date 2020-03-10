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

class yoloDetector:

    def __init__(self, path):

        self.args = {"confidence": 0.5, "threshold": 0.3 }

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.join(path,"yolo-coco", "coco.names")
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
        	dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.join(path,"yolo-coco", "yolov3.weights")
        configPath = os.path.join(path,"yolo-coco", "yolov3.cfg")

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    def detect(self, imageInput, display=False, save=None):

        image = imageInput
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        	swapRB=True, crop=False)

        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        detections = {'cars':[], 'persons':[], 'trafficLights':[]}

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
                if confidence > self.args["confidence"]:
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
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args["confidence"],
            self.args["threshold"])

        # ensure at least one detection exists
        # Loop every detection
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # text = f"{self.LABELS[classIDs[i]]}: {confidences[i]:.4f} - X: {x+w/2} Y: {y+h/2}"
                text = f"{self.LABELS[classIDs[i]]}_{i}: {confidences[i]:.4f} "
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

                car = ("car" in self.LABELS[classIDs[i]])
                bus = ("bus" in self.LABELS[classIDs[i]])
                truck = ("truck" in self.LABELS[classIDs[i]])

                if car or bus or truck:
                    detections['cars'].append( (x+w/2,y+h/2, f"car_{i}",confidences[i],w,h) )
                elif "person" in self.LABELS[classIDs[i]]:
                    detections['persons'].append( (x+w/2,y+h/2, f"person_{i}",confidences[i],w,h) )
                elif "traffic light" in self.LABELS[classIDs[i]]:
                    detections['trafficLights'].append( (x+w/2,y+h/2, f"traffic_{i}",confidences[i],w,h) )

        if display:
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            imS = cv2.resize(image, (960, 540))
            cv2.imshow("output", imS)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save!=None:
            cv2.imwrite(save, image) # write to png

        return detections
