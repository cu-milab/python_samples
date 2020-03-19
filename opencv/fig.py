import numpy as np
import cv2 as cv

# 512 x 512の画像領域を確保
img = np.zeros((512, 512, 3), np.uint8)

# 線，矩形，円，円弧を描画
cv.line(img, (0,0), (511,511), (255,0,0),5)
cv.rectangle(img, (384,0), (510,128), (0,255,0),3)
cv.circle(img, (447,63), 63, (0,0,255),  -1)
cv.ellipse(img, (256,256), (100,50), 0, 0, 180, 255, -1)

#折れ線を描画
pts = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img, [pts], True, (0,255,255))

#文字列を描画
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10,500), font, 4, (255,255,255), 2, cv.LINE_AA)

cv2.imshow('image',img)
cv2.waitKey(0)