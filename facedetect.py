import numpy as np
import cv2 as cv
import pickle

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {'dr-satish-chandra': 0, 'aman': 1, 'rohan': 2, 'anisha': 3, 'goodnaina': 4}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}
cap = cv.VideoCapture(0)
while True:
    ret, img=cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        id_,conf = recognizer.predict(roi_gray)
        if conf>=45 and conf <=85:
            print(id_)
            print(labels.get(id_))
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels.get(id_)
            color = (255,255,255)
            stroke=2
            cv.putText(img, name,(x,y), font, 1, color, stroke, cv.LINE_AA)
        img_item="my-image.png"
        cv.imwrite(img_item,roi_color)
        color=(255,0,0)
        stroke=2
        end_cord_x=x+h
        end_cord_y=y+w
        cv.rectangle(img,(x,y),(end_cord_x,end_cord_y),color,stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv.imshow('img',img)
    k=cv.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
cv.destroyAllWindows()
