import cv2
import os

files = os.listdir()
print(files)

img_path_list = []
for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)  
print(img_path_list)

for i in img_path_list:
    print(i)
    image = cv2.imread(i)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.045, minNeighbors = 6)
    
    for (j, (x,y,w,h)) in enumerate(rects):
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(image, "Kedi {}".format(j+1), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow(i, image)   
    
    if cv2.waitKey(0) &0xFF == 13: continue
