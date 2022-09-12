import cv2
import os

path = "images"
imgWidth = 180
imgHight = 120

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 180)

global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + "/" + str(countFolder)):
        countFolder += 1
    os.makedirs(path + "/" + str(countFolder))
saveDataFunc()

count = 0
i = 0
while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (imgWidth, imgHight))
        if count % 5 == 0:
            cv2.imwrite(path + "/" + str(countFolder) + "/" + str(i) + "_.png", frame)
            i += 1
            print(i)
        count += 1
        
        cv2.imshow("Frame", frame) 
    if cv2.waitKey(1) &0xFF == 13: break 
cap.release()
cv2.destroyAllWindows()
