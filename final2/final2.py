
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import argparse
import cv2
import dlib
import os
from gaze_tracking import GazeTracking
from datetime import datetime

#-----------------------------------------------------------------------------------------------------------#

f = open("report.txt", "w")

#-----------------------------------------------------------------------------------------------------------#

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar

MOUTH_AR_THRESH = 0.72

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
    help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
    help="index of webcam on system")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
#grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

#---------------------------------------------------------------------------------------------------------------#

#id = int(input('Enter your id: '))
id = 2
l = []
m = []
label = 0
confidence = 0

people = []
for i in os.listdir(r'D:\sem8_stuff\project_code\face\faces\train'):
    people.append(i)

haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
des_face = 0

#---------------------------------------------------------------------------------------------------------------#

gaze = GazeTracking()

#---------------------------------------------------------------------------------------------------------------#

vs = VideoStream(src=0).start()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 5, (640, 480))

temp = 0
t1 = 0
t2 = 0
t3 = 0
t4 = 0
t5 = 0

while True:
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    f.write(date_time)
    f.write("\n")

    #-------------------------------------------------------------------------------------#
    frame = vs.read()
    temp = temp + 1
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #--------------------------------------------------------------------------------------#
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    act_frame = 0
    act_frame = act_frame + 1
    t = ()
    if faces_rect == t:
        print('no face')
        f.write('no face\n')
    else:
        print('face is detected')
        f.write('face is detected\n')
        des_face = des_face +1
    
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)

        l.append(label)
        m.append(confidence)
        
        if label != id:
            print('recognized person is not you')
            f.write('recognized person is not you\n')
        else:
            print('face is recognized')
            f.write('face is recognized\n')

        if confidence > 100:
            print('face is not clearly visible')
            f.write('face is not clearly visible\n')
        else:
            print('face is clearly visible')
            f.write('face is clearly visible\n')

        print(f'Label = {label} with a confidence of {confidence}')  #confidence - the distance to the closest item in the database
                                                                    #i.e smaller the better
        f.write(f'Label = {label} with a confidence of {confidence}\n')

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        cv2.putText(frame, str(people[label]) , (x+w, y+h), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), thickness=2)
        cv2.putText(frame, "CON: {:.2f}".format(confidence),  (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), thickness=2)
    #--------------------------------------------------------------------------------------#
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #print(shape)

        # extract the mouth coordinates, then use the
        # coordinates to compute the mouth aspect ratio
        mouth = shape[mStart:mEnd]

        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        # compute the convex hull for the mouth, then
        # visualize the mouth
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw text if mouth is open
        if mar > MOUTH_AR_THRESH:
            print(mar, "Mouth is open")
            f.write(f'{mar} - Mouth is open\n')
            cv2.putText(frame, "Mouth is Open!", (30,60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        else:
            print(mar, "Mouth is closed")
            f.write(f'{mar} - Mouth is closed\n')
            t2 = t2 + 1
    #--------------------------------------------------------------------------------------#

    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Looking down"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
        t3 = t3 + 1
    elif gaze.is_center():
        text = "Looking center"
        t3 = t3 + 1

    cv2.putText(frame, text, (90, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
    print(text)
    f.write(f'{text}\n')
    
   

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 2)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 2)

    #--------------------------------------------------------------------------------------#
    #print(temp)
    cv2.imshow("Frame", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
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

# print(temp, score, t2, t3)


# f.write("\n")
# f.write(f'total frames analysed: {temp}\n')
