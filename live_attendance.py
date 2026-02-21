import cv2
import csv
import os
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

labels = {}
for i, name in enumerate(sorted(os.listdir("dataset"))):
    labels[i] = name

cam = cv2.VideoCapture(0)
marked = []
cv2.namedWindow("SMART ATTENDANCE SYSTEM", cv2.WINDOW_NORMAL)
with open("attendance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Time"])

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Frame not captured properly, exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face_id, confidence = recognizer.predict(
                gray[y:y+h, x:x+w]
            )

            if confidence < 85:
                name = labels[face_id]
                if name not in marked:
                    marked.append(name)
                    writer.writerow(
                        [name, datetime.now().strftime("%H:%M:%S")]
                    )

                cv2.putText(frame, name, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,255,0), 2)
            else:
                cv2.putText(frame, "Unknown", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,0,255), 2)

            cv2.rectangle(frame,(x,y),(x+w,y+h),
                          (255,0,0),2)
        

        cv2.imshow("SMART ATTENDANCE SYSTEM", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

   

cam.release()
cv2.destroyAllWindows()
