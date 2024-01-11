
import cv2
from classes.ImageLandmarks import ImageLandmarks
from classes.LandmarkDifferences import LandmarkDifferences
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import os
import numpy as np
import joblib


def get_landmarks(img_path, processor, face_boundaries):
    img = cv2.imread(img_path)
    # we need to change the structure of the bounding box to be compatible with the SPIGA framework
    # instead of[x1, y1, x2, y2] we need [x1, y1, w, h]
    x1, y1, x2, y2 = face_boundaries
    bounding_box = [x1, y1, x2-x1, y2-y1]
    features = processor.inference(img, [bounding_box])
    landmarks = features['landmarks'][0]

    return landmarks


def populate_landmarks_dictionary_emotion(landmarks_dictionary: ImageLandmarks, emotion: str, face_boundaries_dict, id: int):

    emotion_for_id_path = os.path.join("pics_KDEF", f"{id}-{emotion}.jpg")
    if emotion == "neutral":
        if not os.path.exists(emotion_for_id_path):
            raise ValueError("Neutral image does not exist")
    else:
        if not os.path.exists(emotion_for_id_path):
            return None

    emotion_boundaries = face_boundaries_dict[f"{id}-{emotion}"]

    emotion_landmarks = get_landmarks(emotion_for_id_path, processor, emotion_boundaries)

    landmarks_dictionary.landmarks[f"{id}-{emotion}"] = emotion_landmarks
    print(f"Done for subject {id} and emotion {emotion}")


def align_landmarks(landmarks_dictionary: ImageLandmarks, emotion:str, subject_id: int):
    if emotion == "neutral":
        return None

    neutral_landmarks = landmarks_dictionary.landmarks[f"{subject_id}-neutral"]
    # check if key exists in dictionary
    if f"{subject_id}-{emotion}" not in landmarks_dictionary.landmarks:
        return None
    to_align_landmarks = landmarks_dictionary.landmarks[f"{subject_id}-{emotion}"]

    nasal_root = neutral_landmarks[51]
    to_align_nose = to_align_landmarks[51]
    to_align_nose_difference = np.array(nasal_root) - np.array(to_align_nose)
    aligned_landmarks = (np.array(to_align_landmarks) + to_align_nose_difference).tolist()
    landmarks_dictionary.landmarks[f"{subject_id}-{emotion}"] = aligned_landmarks


def populate_landmarks_dictionary(landmarks_dictionary: ImageLandmarks, emotions: list):
    id_list = []
    for filename in os.listdir("pics_KDEF"):
        id, emotion = filename[:-4].split("-")
        id_list.append(id) if id not in id_list else None

    print(id_list)
    face_boundaries_dict = joblib.load("face_boundaries_dict_KDEF.pkl")
    for id in id_list:

        for emotion in emotions:
            populate_landmarks_dictionary_emotion(landmarks_dictionary, emotion, face_boundaries_dict, id)

        neutral_landmarks = landmarks_dictionary.landmarks[f"{id}-neutral"]

        for emotion in emotions:
            align_landmarks(landmarks_dictionary, emotion, id)

        print(f"Done for subject {id}")


if __name__ == '__main__':
    path = "D://dataspellprojects//spigabasedck+//picsKDEF"
    dataset = 'wflw'
    processor = SPIGAFramework(ModelConfig(dataset))
    emotions = ["neutral", "afraid", "angry", "disgusted", "happy", "sad", "surprised"]
    landmarks_dictionary = ImageLandmarks()
    # populate_landmarks_dictionary(landmarks_dictionary, emotions)
    # joblib.dump(landmarks_dictionary, "landmarks_KDEF.joblib")
    landmarks_dictionary = joblib.load("landmarks_KDEF.joblib")



    landmarks = LandmarkDifferences(landmarks_dictionary)
    landmark_dictionary = landmarks.get_differences()

    keys = list(landmark_dictionary.keys())
    labels = [key.split("-")[1] for key in keys]

    landmark_list = list(landmark_dictionary.values())
    landmark_list = np.array(landmark_list)
    landmark_list = landmark_list.reshape(-1, 196)

    # random forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(landmark_list, labels, test_size=0.2, random_state=41)

    params = {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'max_depth': [10, 15, 20, 25],
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 3, 4, 5]
    }

    # gridsearch = GridSearchCV(RandomForestClassifier(), params, cv=5, verbose=1, n_jobs=-1)
    # gridsearch.fit(X_train, y_train)
    # print(gridsearch.best_params_)
    bootstrap = True
    max_depth = 10
    max_features = 'sqrt'
    min_samples_leaf = 1
    min_samples_split = 4
    n_estimators = 300
    # fit the model
    # clf = RandomForestClassifier(n_estimators=200, max_depth=20, bootstrap=True, max_features='sqrt', min_samples_leaf=1, min_samples_split=2)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)

    # predict
    y_pred = clf.predict(X_test)

    # accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    # classification report
    print(classification_report(y_test, y_pred))

    probabilities = clf.predict_proba(X_test)

    for index, probability in enumerate(probabilities):
        true_label = y_test[index]
        if true_label == "angry" and np.argmax(probability) != 0: # 0 is anger's index
            print(probability)
            classes = clf.classes_
            print(classes)
            str = ""
            for a, b in zip(classes, probability):
                str += f"{a}: {b}  "
            print(str)

