import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_id = 0

for name in sorted(os.listdir("dataset")):
    person_path = f"dataset/{name}"
    for img in os.listdir(person_path):
        img_path = f"{person_path}/{img}"
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(gray)
        labels.append(label_id)
    label_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

print("✅ Model trained successfully")
