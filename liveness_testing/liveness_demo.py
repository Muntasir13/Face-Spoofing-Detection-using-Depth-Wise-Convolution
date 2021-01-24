# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
import torchvision.transforms as transforms
from model_arch import Model
from PIL import Image
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2
import os


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.join(os.getcwd(), "deploy.prototxt")
modelPath = os.path.join(os.getcwd(), "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = Model()
model.load_state_dict(torch.load(os.path.join(os.getcwd(), "model.pth")))
model.cuda()
model.eval()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(-1)
time.sleep(3.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	_, frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = Image.fromarray(face)
			face = face.transpose(Image.ROTATE_270)
			transform = transforms.Compose(
				[
					transforms.Resize((112, 112)),
					transforms.ToTensor(),
					transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				]
			)

			# pass the face ROI through the trained liveness detector
			# model to determine if the face is "real" or "fake"
			face = transform(face)
			output = model(face.unsqueeze(0).cuda())
			print(output)
			output = torch.where(output < 0.25, torch.zeros_like(output), torch.ones_like(output))
			if output.item() == 0:
				label = "{}".format("Live")
				# draw the label and bounding box on the frame
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
			else:
				label = "{}".format("Spoof")
				# draw the label and bounding box on the frame
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
