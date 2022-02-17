import numpy as np
import cv2 as cv
import os

id = int(input('Enter your id: '))
l = []
m = []
label = 0
confidence = 0

people = []
for i in os.listdir(r'D:\sem8_stuff\project_code\face\faces\train'):
    people.append(i)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap = cv.VideoCapture(0)

des_face = 0
while(True):
    ret, frame = cap.read()
    gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    act_frame = 0
    act_frame = act_frame + 1
    t = ()
    if faces_rect == t:
        print('no face')
    else:
        des_face = des_face +1
    

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)

        l.append(label)
        m.append(confidence)
        
        if label != id:
            print('recognized person is not you')
        if confidence > 100:
            print('face is not clearly visible')

        #print(f'Label = {label} with a confidence of {confidence}')  #confidence - the distance to the closest item in the database
                                                                     #i.e smaller the better

        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        cv.putText(frame, str(people[label]) , (x+w, y+h), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.putText(frame, str(confidence) , (x+w, y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)


    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#score calculation
des_label = 0
des_confidence_count = 0
score = 0

des_label = l.count(id)
if des_label > 0.9*(len(l)):
    score = 1
    #print("label ", score)
for des_confidence in m:
    if des_confidence < 100:
        des_confidence_count = des_confidence_count + 1
if des_confidence_count > 0.6*(len(m)):
    score = score + 1
    #print("confidence", score)
if des_face != 0:
    score = score + 1
    #print("face", score)

print("Score = ", score)
