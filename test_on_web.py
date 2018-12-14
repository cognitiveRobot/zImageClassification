# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# initialize the total number of frames that *consecutively* contain
# santa along with threshold required to trigger the santa alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 1
 
# initialize is the santa alarm has been triggered
HAND = False


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# prepare the image to be classified by our deep learning network
	image = cv2.resize(frame, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)




	
 
# classify the input image
# (notSanta, santa) = model.predict(image)[0]

# # build the label
# label = "Santa" if santa > notSanta else "Not Santa"
# proba = santa if santa > notSanta else notSanta
# label = "{}: {:.2f}%".format(label, proba * 100)

	(notHand, fist, handOpen) = model.predict(image)[0]



	if fist > handOpen and fist > notHand:
		label = "fist" 
		proba = fist
		TOTAL_CONSEC += 1	
	elif handOpen > fist and handOpen > notHand:
		label = "handOpen"
		proba = handOpen
		TOTAL_CONSEC += 1		
	else:
		label = "notHand"
		proba = notHand	
		HAND = False
		TOTAL_CONSEC = 0
	if not HAND and TOTAL_CONSEC >= TOTAL_THRESH:
			# indicate that santa has been found
		HAND= True
		label = "Hand"
			
	# else:
	# 	label = "notHand"
	# build the label and draw it on the frame
	label = "{}: {:.2f}%".format(label, proba * 100)
	frame = cv2.putText(frame, label, (10, 25),
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	
		

	

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()

