import os
import cv2
path = "" # original KDEF folder
new_folder_path = "../../pics_KDEF" # reformatted KDEF folder
index = 0
for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        if not file.endswith(".JPG"):
            raise ValueError("Non-jpg file in directory")
        image = cv2.imread(os.path.join(path, folder, file))

        image_name = file[:-4]
        if not image_name.endswith("S"):
            continue

        session = image_name[0]
        gender = image_name[1]
        subject_id = image_name[2:4]
        emotion = image_name[4:6]
        emotions = {"AF": "afraid", "AN": "angry", "DI": "disgusted", "HA": "happy", "NE": "neutral", "SA": "sad", "SU": "surprised"}

        if gender == "M":
            subject_id = "1" + subject_id
        elif gender == "F":
            subject_id = "2" + subject_id
        else:
            asdf = subject_id + "-" + emotions[emotion]
            raise ValueError(f"{gender}")

        if session == "A":
            subject_id = subject_id + "1"
        elif session == "B":
            subject_id = subject_id + "2"
        else:
            raise ValueError(f"{session}")

        emotion = emotions[emotion]
        image_name = subject_id + "-" + emotion

        cv2.imwrite(os.path.join(new_folder_path, image_name + ".jpg"), image)
        index += 1
        print(index)
