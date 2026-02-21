import cv2
import os

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)
name = input("Enter person name: ")

path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        cv2.imwrite(f"{path}/{count}.jpg",
                    gray[y:y+h, x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),
                      (255,0,0),2)

    cv2.imshow("Dataset Creator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or count >= 30:
        break


cam.release()
cv2.destroyAllWindows()
