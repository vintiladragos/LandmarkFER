import os
import cv2

path = ""  # original CK+ emotion-labelled images


def preprocess_files(emotion: str):
    emotion_path = path + "//" + emotion
    if emotion == "neutral":
        for file in os.listdir(emotion_path):
            if not file.endswith(".png"):
                raise ValueError("Non-png file in directory")
            img_name = file[:-4]
            subject_id, emotion_th, _ = img_name.split("_")
            subject_id = int(subject_id[1:])
            if subject_id not in max_for_subject:
                max_for_subject[subject_id] = int(emotion_th)
            else:
                if int(emotion_th) > max_for_subject[subject_id]:
                    max_for_subject[subject_id] = int(emotion_th)

    for file in os.listdir(emotion_path):
        if not file.endswith(".png"):
            raise ValueError("Non-png file in directory")

        img_path = os.path.join(emotion_path, file)
        img = cv2.imread(img_path)

        img_name = file[:-4]
        subject_id, emotion_th, _ = img_name.split("_")
        subject_id = int(subject_id[1:])
        if emotion == "neutral":
            if int(emotion_th) != max_for_subject[subject_id]:
                continue

        landmark_key = f"{subject_id}-{emotion}"
        cv2.imwrite("../../pics_ck+" + "//" + landmark_key + ".png", img)


if __name__ == '__main__':
    preprocess_files("happiness")
    preprocess_files("anger")
    preprocess_files("surprise")
    preprocess_files("fear")
    preprocess_files("disgust")
    preprocess_files("sadness")
    preprocess_files("contempt")
    max_for_subject = {}
    preprocess_files("neutral")
