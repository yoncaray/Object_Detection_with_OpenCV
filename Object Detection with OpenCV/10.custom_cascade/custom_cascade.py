import cv2

frame_width = 280
frame_height = 360

cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

def empty(): pass
# Trackbar
cv2.namedWindow("Result")
cv2.resizeWindow("Result", frame_width, frame_height + 100)
cv2.createTrackbar("Scale", "Result", 400, 1000, empty)
cv2.createTrackbar("Neighbor", "Result", 4, 50, empty)

# Cascade Classifier
cascade = cv2.CascadeClassifier("images/cascade.xml")

while True:
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detection Parameters
        scaleVal = 1 + (cv2.getTrackbarPos("Neighbor", "Result")/1000)
        neighbor = cv2.getTrackbarPos("Neighbor", "Result")
        # Detection
        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)
    
    for (x,y,w,h) in rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.putText(frame, "Bottle", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) &0xFF == 13: break

cap.release()
cv2.destroyAllWindows()
