# opencv kütüphanesini içe aktaralım
import cv2 

# resmi siyah beyaz olarak içe aktaralım resmi çizdirelim
img = cv2.imread("images/exam.jpg", 0)
cv2.imshow("Original", img)

# resim üzerinde bulunan kenarları tespit edelim ve görselleştirelim edge detection
edges =cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)
cv2.imshow("Edge Detection", edges)

# yüz tespiti için gerekli haar cascade'i içe aktaralım
face_cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")

# yüz tespiti yapıp sonuçları görselleştirelim
face_rects = face_cascade.detectMultiScale(img)
for (x,y,w,h) in face_rects:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 3)
cv2.imshow("Face Detection", img)

# HOG ilklendirelim insan tespiti algoritmamızı çağıralım ve svm'i set edelim
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# resme insan tespiti algoritmamızı uygulayalım ve görselleştirelim
(face_rects, weights) = hog.detectMultiScale(img, padding=(38,8), scale=1.05)
for (x,y,w,h) in face_rects:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 3)
cv2.imshow("Face Detection with HOG", img)

cv2.waitKey(0)
