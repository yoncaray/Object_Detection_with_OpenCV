import cv2
import numpy as np
from collections import deque

# Nesne merkezini depolayacak veri tipi
buffer_size = 16
points = deque(maxlen = buffer_size)

# Mavi renk aralığı HSV
blueLower = (85,100,0)
blueUpper = (180,255,255)

# Capture
cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    success, imgOriginal = cap.read()
    if success:
        # Blur
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0)
        # HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)
        
        # Mavi için maske oluştur
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("Mask Image", mask)
        
        # Maskenin etrafında kalan gürültüleri sil
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)
        cv2.imshow("Mask + Erezyon ve Genisleme", mask)
        
        # Kontur
        (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        
        if len(contours) > 0:
            # En büyük konturu al
            c = max(contours, key = cv2.contourArea)
            # Dikdörtgene çevir
            rect = cv2.minAreaRect(c)
            ((x,y), (width,height), rotation) = rect
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x), np.round(y), np.round(width), 
                                                                           np.round(height), np.round(rotation))
            # Kutucuk oluştur
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            # Moment
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), (int(M["m01"]/M["m00"])))
            
            # Konturu çizdir: sarı
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255), 2)
            # Merkeze bir nokta çiz
            cv2.circle(imgOriginal, center, 5, (255,0,255), -1)
            # Ekrana bilgileri yazdır
            cv2.putText(imgOriginal, s, (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
        # deque
        points.append(center)
        for i in range(1, len(points)):
            if points[i-1] is None or points[i] is None: continue
            cv2.line(imgOriginal, points[i-1], points[i], (0,255,255), 3)
            
        cv2.imshow("Original Tespit", imgOriginal)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()
