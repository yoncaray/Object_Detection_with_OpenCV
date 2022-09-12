import cv2
import matplotlib.pyplot as plt

chos = cv2.imread("images/chocolates.jpg", 0)
plt.figure(), plt.imshow(chos, cmap = "gray"), plt.axis("off")

cho = cv2.imread("images/nestle.jpg", 0)
plt.figure(), plt.imshow(cho, cmap = "gray"), plt.axis("off")

# Orb tanımlayıcı
# Köşe-kenar gibi nesneye ait özellikler
orb = cv2.ORB_create()

# Anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

# Bruce-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# Noktaları eşleştir
matches = bf.match(des1, des2)
# Mesafeye göre sırala
mathes = sorted(matches, key = lambda x:x.distance)
# Eşleşen resimleri görselteştirme
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 2)
plt.imshow(img_match), plt.axis("off"), plt.title("ORB")

# ------------------------------------

# Sift
sift = cv2.SIFT.create()

# Brute-force
bf = cv2.BFMatcher()

# Anahtar nokta tespiti
kp1, des1 = sift.detectAndCompute(cho, None)
k2, des2 = sift.detectAndCompute(chos, None)

matches = bf.knnMatch(des1, des2, k = 2)

iyi_eslesme = []

for match1, match2 in matches:
    if match1.distance < 0.75*match2.distance:
        iyi_eslesme.append([match1])

plt.figure()
sift_matches = cv2.drawMatchesKnn(cho, kp1, chos, kp2, iyi_eslesme, None, flags = 2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("SIFT")
