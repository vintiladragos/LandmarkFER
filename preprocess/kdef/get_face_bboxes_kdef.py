from retinaface import RetinaFace
import os
import cv2
import joblib

path = "../../pics_KDEF" # reformatted KDEF folder


if __name__ == '__main__':
    face_boundaries_dict = {}

    for file in os.listdir(path):
        if not file.endswith(".jpg"):
            raise ValueError("Non-png file in directory")

        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img_name = file[:-4]

        coord = RetinaFace.detect_faces(img_path=img_path)
        if len(coord) != 1:
            raise ValueError("Why did you detect something else than one face?")

        face_info = coord["face_1"]
        face_boundaries_dict[img_name] = face_info["facial_area"]
        print("detected face: " + str(img_name))

    joblib.dump(face_boundaries_dict, "face_boundaries_dict_KDEF.pkl")

    print(face_boundaries_dict)
