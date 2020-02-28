import sys
import os
import numpy as np
import pickle
import cv2

"""  """

def generate(path="", time=60):

    combinedPCD_dir = os.path.join(path, f"combinedPCD")

    combinedPCD_images = os.listdir(combinedPCD_dir)

    combinedPCD_reference = os.path.join(path, f"combinedPCD", f"CombinedPCD_0.png")
    image_reference= cv2.imread(combinedPCD_reference)
    height, width, _ = image_reference.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = os.path.join(combinedPCD_dir, f"combinedPCD.avi")
    out = cv2.VideoWriter(video_out, fourcc, len(combinedPCD_images)/time, (width,height))

    for posIndex in range(len(combinedPCD_images)):

        combinedPCD_image = os.path.join(combinedPCD_dir, f"CombinedPCD_{posIndex}.png")
        frame = cv2.imread(combinedPCD_image)
        out.write(frame)

    out.release()
