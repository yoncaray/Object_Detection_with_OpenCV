import cv2
import matplotlib.pyplot as plt

einstein = cv2.imread("images/einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")

# Sınıflandırıcı
face_cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y), (x+w,y+h), (255,255,255), 10)
plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")

#------------------------------------------------------

barcelona = cv2.imread("images/barcelona.jpg", 0)
plt.figure(), plt.imshow(barcelona, cmap="gray"), plt.axis("off")

face_rect = face_cascade.detectMultiScale(barcelona, minNeighbors = 6)

for (x,y,w,h) in face_rect:
    cv2.rectangle(barcelona, (x,y), (x+w,y+h), (255,255,255), 10)
plt.figure(), plt.imshow(barcelona, cmap="gray"), plt.axis("off")

# Video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == True:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 6)
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.imshow("Face Detect", frame)
    if cv2.waitKey(1) &0xFF == 13: break
cap.release()
cv2.destroyAllWindows()
