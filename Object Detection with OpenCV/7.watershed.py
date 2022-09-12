import cv2
import matplotlib.pyplot as plt
import numpy as np

coins = cv2.imread("images/coins.jpg")
plt.figure(), plt.imshow(coins), plt.axis("off")

# lpf: blurring
coins_blur = cv2.medianBlur(coins, 13)
plt.figure(), plt.imshow(coins_blur), plt.axis("off")

# grayscale
coins_gray = cv2.cvtColor(coins_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coins_gray, cmap="gray"), plt.axis("off")

# binary threshold
ret, coins_thresh = cv2.threshold(coins_gray, 65, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coins_thresh, cmap="gray"), plt.axis("off")

# kontur
contours, hierarchy = cv2.findContours(coins_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coins, contours, i, (0,255,0), 10)
plt.figure(), plt.imshow(coins), plt.axis("off")

# açılma
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(coins_thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
plt.figure(), plt.imshow(opening, cmap="gray"), plt.axis("off")

# nesneler arası distance bulma
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.figure(), plt.imshow(dist_transform, cmap="gray"), plt.axis("off")

# nesneleri küçült 
ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform), 255, 0)
plt.figure(), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off")

# arka plan için nesneleri büyüt
sure_background = cv2.dilate(opening, kernel, iterations = 1)
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background, sure_foreground)
plt.figure(), plt.imshow(unknown, cmap="gray"), plt.axis("off")

# bağlantı
ret, marker = cv2.connectedComponents(sure_foreground)
marker = marker + 1
marker[unknown == 255] = 0
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

# havza
marker = cv2.watershed(coins, marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

# kontur
contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coins, contours, i, (0,255,0), 2)
plt.figure(), plt.imshow(coins), plt.axis("off")
