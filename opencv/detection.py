import cv2

face_cascade_path = "/Users/yuzi5/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"


face_cascade = cv2.CascadeClassifier(face_cascade_path)


img = cv2.imread('people.jpg')
src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(src_gray, 1.1, 3, True, (20, 20))

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = img[y: y + h, x: x + w]

cv2.imwrite('opencv_face_detect_rectangle.png', img)

cv2.imshow('image',img)
cv2.waitKey(0)