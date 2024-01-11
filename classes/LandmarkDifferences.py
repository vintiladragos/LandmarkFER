import joblib
from .ImageLandmarks import ImageLandmarks
import numpy as np


class LandmarkDifferences:
    def __init__(self, image_landmarks: ImageLandmarks):
        self.image_landmarks = image_landmarks
        self.differences = {}

        for image_landmarks_name, landmarks in image_landmarks.landmarks.items():
            subject_id, emotion = image_landmarks_name.split("-")

            if emotion == "neutral":
                continue
            else:  # "neutral", "anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"

                neutral_landmarks = image_landmarks.landmarks[f"{subject_id}-neutral"]
                # neutral_nose = neutral_landmarks[51]
                # image_landmarks.landmarks[image_landmarks_name] = np.array(neutral_landmarks) - np.array(landmarks)
                self.differences[image_landmarks_name] = np.array(neutral_landmarks) - np.array(landmarks)

        joblib.dump(self.differences, "differences.joblib")

    def get_differences(self):
        return self.differences
