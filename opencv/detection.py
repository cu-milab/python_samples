import cv2

#学習モデルの読み込み。下記のNAMEにはwiondowsのユーザー名に変更する必要あり。
face_cascade_path = "/Users/NAME/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"


face_cascade = cv2.CascadeClassifier(face_cascade_path)


img = cv2.imread('lenna.png')
src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#OpenCV4.xより仕様が変わったため修正。
faces = face_cascade.detectMultiScale(src_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = img[y: y + h, x: x + w]

cv2.imwrite('opencv_face_detect_rectangle.png', img)

cv2.imshow('image',img)
cv2.waitKey(0)
