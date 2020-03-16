# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread("lenna.png", 0)
img2 = img.copy()
template = cv2.imread('lenna_mini.png',0)
w, h = template.shape[::-1]


img = img2.copy()
method = eval("cv2.TM_SQDIFF")

# Apply template Matching
res = cv2.matchTemplate(img,template,method)
res=cv2.normalize(res,0,255,cv2.NORM_L1)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
top_left = min_loc

bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img,top_left, bottom_right, 255, 2)

cv2.imshow('SSD',res)
cv2.imshow('result',img)
cv2.waitKey(1)
cv2.destroyAllWindows()