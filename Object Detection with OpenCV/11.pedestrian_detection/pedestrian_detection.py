import cv2
import os

files = os.listdir("images")
print(files)

# Hog Tanımlayıcısı
hog = cv2.HOGDescriptor()

# Tanımlayıcıya SVM Ekle
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for i in files:
    img = cv2.imread("images/" + i)
    
    (rects, weights) = hog.detectMultiScale(img, padding = (8,8), scale = 1.05)
    for (x,y,w,h) in rects:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
    cv2.imshow("Pedestrian", img)
    if cv2.waitKey(0) &0xFF == 13: continue
    if cv2.waitKey(0) &0xFF == ord("q"): break 

cv2.destroyAllWindows()
