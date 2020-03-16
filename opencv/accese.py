# -*- coding: utf-8 -*-

import cv2

img = cv2.imread('lenna.png', 1)
img_copy = img.copy()

h, w, ch = img.shape[:3]
print(h, w, ch)

for i in range(int(h / 2) - 50, int(h / 2) + 50):
    for j in range(int(w / 2) - 50, int(w / 2) + 50):
        #img_copy[i, j, 0] = 0
        img_copy[i, j, 1] = 0
        img_copy[i, j, 2] = 0
        

color = img_copy[int(h / 2), int(w / 2)]
print(color)

cv2.imshow('image',img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()