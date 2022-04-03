
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def main_mouth():
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

	# define one constants, for mouth aspect ratio to indicate open mouth
	MOUTH_AR_THRESH = 0.70

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
		help="path to facial landmark predictor")
	ap.add_argument("-w", "--webcam", type=int, default=0,
		help="index of webcam on system")
	args = vars(ap.parse_args())

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor....")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# grab the indexes of the facial landmarks for the mouth
	(mStart, mEnd) = (49, 68)

	# start the video stream thread
	print("[INFO] starting video stream thread....")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

	frame_width = 640
	frame_height = 360

	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
	time.sleep(1.0)

	# loop over frames from the video stream
	while True:
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=640)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
			
			cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
			cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# Draw text if mouth is open
			if mar > MOUTH_AR_THRESH:
				print(mar, "Mouth is open")
				cv2.putText(frame, "Mouth is Open!", (30,60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
			else:
				print(mar, "Mouth is closed")
		# Write the frame into the file 'output.avi'
		out.write(frame)
		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
main_mouth()