# -*- coding: utf-8 -*-

import cv2

# Load an color image in grayscale
img_color = cv2.imread('lenna.png', 1)
img_gray = cv2.imread('lenna.png', 0)

cv2.imshow('image',img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('lenna_gray.png',img_gray)