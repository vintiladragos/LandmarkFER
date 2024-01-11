import cv2
import numpy as np
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import os
import joblib
from classes.ImageLandmarks import ImageLandmarks
path = "./pics_ck+"
all_image_landmarks = {}
dataset = 'wflw'
processor = SPIGAFramework(ModelConfig(dataset))
# image_landmarks = ImageLandmarks
image_landmarks = joblib.load("landmarks_ck+.joblib")

landmarks_for_one_image = image_landmarks.landmarks["11-neutral"]
img = cv2.imread(os.path.join(path, "11-neutral.png"))

if __name__ == "__main__":
    # draw landmarks
    for index, (x, y) in enumerate(landmarks_for_one_image):
        # we take the top of the nose and right corner of the left eye as positional references
        if index == 51:
            cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)
        else:
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    emotion_landmarks = image_landmarks.landmarks["11-disgust"]

    # draw lines between neutral and smile landmarks
    for index, (x, y) in enumerate(landmarks_for_one_image):
        if index == 51:
            continue
        else:
            cv2.line(img, (int(x), int(y)), (int(emotion_landmarks[index][0]), int(emotion_landmarks[index][1])), (255, 0, 0), 1)

    # draw smile landmarks
    for index, (x, y) in enumerate(emotion_landmarks):
        # we take the top of the nose and right corner of the left eye as positional references
        if index == 51:
            cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), -1)
        else:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

    print("Showing image")
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # legend
    # red: neutral landmarks
    # blue: disgust landmarks
    # cv2.imwrite("11-disgust.png", img)

