
import cv2
import numpy as np

def gamma(image, gamma):
    img = image / 255.0
    img = cv2.pow(img, gamma)
    img = np.uint8(img*255)
    return img

classifier_path = "haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(classifier_path)
video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img,"Rostos: "+ `len(faces)`, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    img2 = gamma(img,0.7)

    cv2.imshow('Video',np.hstack((img,img2)))
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
video_capture.release()
cv2.destroyAllWindows()
