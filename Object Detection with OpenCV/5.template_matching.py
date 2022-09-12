import cv2
import matplotlib.pyplot as plt

# Template Matching: Şablon Eşleme
img = cv2.imread("images/cat.jpg", 0)
print(img.shape)
template = cv2.imread("images/cat_face.jpg", 0)
print(template.shape)
h, w = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    method = eval(meth) # eval stringleri metota dönüştürüyor.
    res = cv2.matchTemplate(img, template, method)  
    print(res.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if methods in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray") 
    # 1 satır 2 sütun olsun ve 1.satırı alıyor.
    plt.title("Eslesen Sonuc"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap = "gray")
    # 1 satır 2 sütun olsun ve 2.satırı alıyor.
    plt.title("Tespit edilen Sonuc"), plt.axis("off")
    plt.suptitle(meth)
    plt.show()
           