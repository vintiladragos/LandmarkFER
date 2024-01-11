import cv2
import numpy as np
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import os
import joblib
from classes.ImageLandmarks import ImageLandmarks
path = "/pics_KDEF"
all_image_landmarks = {}
dataset = 'wflw'
processor = SPIGAFramework(ModelConfig(dataset))
# image_landmarks = ImageLandmarks
image_landmarks = joblib.load("landmarks_KDEF.joblib")

landmarks_for_one_image = image_landmarks.landmarks["1011-neutral"]
img = cv2.imread(os.path.join(path, "1011-neutral.jpg"))

if __name__ == "__main__":
    # draw landmarks
    for index, (x, y) in enumerate(landmarks_for_one_image):
        # we take the top of the nose and right corner of the left eye as positional references
        if index == 51:
            cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)
        else:
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    smile_landmarks = image_landmarks.landmarks["1011-afraid"]

    # draw lines between neutral and smile landmarks
    for index, (x, y) in enumerate(landmarks_for_one_image):
        if index == 51:
            continue
        else:
            cv2.line(img, (int(x), int(y)), (int(smile_landmarks[index][0]), int(smile_landmarks[index][1])), (255, 0, 0), 1)

    # smile_image = cv2.imread(os.path.join(path, "m-005-2.jpg"))

    # draw smile landmarks
    for index, (x, y) in enumerate(smile_landmarks):
        # we take the top of the nose and right corner of the left eye as positional references
        if index == 51:
            cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), -1)
        else:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)


    print("Showing image")
    cv2.imshow("image", img)
    cv2.waitKey(0)

