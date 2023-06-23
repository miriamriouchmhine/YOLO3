from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule
import numpy as np
import cv2
import argparse, imutils
import time, dlib, datetime
from datetime import datetime
from itertools import zip_longest
import mysql.connector
from mylib.config import prototxt, model, frame_size, downIsEntry, line_color, line_position, line_thickness, pixel_end_height, pixel_end_width, pixel_start_height, pixel_start_width, confidence_config, skip_frames, media, factor_escala 

t0 = time.time()
#Declaramos la variable global ocupación anterior al principio del código, 
global ocu_anterior
ocu_anterior = -1

def run():

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to detect
	#Contiene el nombre de las clases que se utilizan en el modelo de detección de objetos
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromDarknet(prototxt, model)
	if not args.get("input", False):
		# grab a reference to the ip camera
		print("[INFO] Starting the live stream..")
		vs = VideoStream(config.url).start()
		time.sleep(2.0)

	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])

	# Obtener número total de fotogramas
	#total_frames = int(vs.stream.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_count = 0
	#total_frames = vs.count_frames()

	# Imprimir número total de fotogramas
	#print('Número total de fotogramas: ', total_frames)

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]

	# start the frames per second throughput estimator
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)
		

#-------------------------GUARDAR EN BASE DATOS-----------------------------------
	#Crear conexion a la base de datos
	conn = mysql.connector.connect(
		host="localhost",
		user="root",
		password="12345678"
		# password="admin"
	)
	#Crear un cursor para ejecutar comandos SQL
	cur = conn.cursor()
	#Crear una base de datos llamada base si no existe
	cur.execute("CREATE DATABASE IF NOT EXISTS db")
	#Usar la base de datos creada
	cur.execute("USE db")
	#Crear una tabla llamada registros con tres columnas, fecha, hora, ocupacion
	cur.execute("CREATE TABLE IF NOT EXISTS biblioteca (fecha_hora TEXT, ocupacion INT)")
	#Guardamos cambios en la base de datos
	conn.commit()

	# Definir una funcion que inserte la ocupación en la tabla
	def guardar_x(x):
		#Obtener la fecha y hora actuales y convertirlas en cadenas de texto
		fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		# hora = datetime.datetime.now().strftime("%H:%M:%S")
		#Insertar el registro en la tabla con los valores de fecha, hora  y ocupacion
		cur.execute("INSERT INTO biblioteca (fecha_hora, ocupacion) VALUES (%s,%s)", (fecha_hora, x))
		#Guardar los cambios en la base de datos
		conn.commit()

		#comprobacion
		print(f"Se ha insertadp el registro: {fecha_hora}, {x}")
	
	# Get the names of the output layers
	def getOutputsNames(net):
		# Get the names of all the layers in the network
		layersNames = net.getLayerNames()
		# Get the names of the output layers, i.e. the layers with unconnected outputs
		return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

	
	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if args["input"] is not None and frame is None:
			break

		frame_count += 1
		#print('Número total de fotogramas: ', total_frames)
		#print('Frame contados: ', frame_count)

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width = frame_size)
		frame = frame[pixel_start_height:pixel_end_height, pixel_start_width:pixel_end_width]
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % skip_frames == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			# blob = cv2.dnn.blobFromImage(frame, 0.0117647059, (W, H), 85)
			blob = cv2.dnn.blobFromImage(frame, 1/255, (W, H), [0,0,0], 1, crop=False)
			# blob = cv2.dnn.blobFromImage(frame, 0.01, (W, H), 100) 			
			net.setInput(blob)
			detections = net.forward(getOutputsNames(net))

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > confidence_config:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue
					
					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		cv2.line(frame, (0, line_position), (W, line_position), line_color, line_thickness)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)
		
		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:
					if downIsEntry == 1:
						# if the direction is negative (indicating the object
						# is moving up) AND the centroid is above the center
						# line, count the object
						if direction < 0 and centroid[1] < line_position:
							totalUp += 1
							empty.append(totalUp)
							to.counted = True

						# if the direction is positive (indicating the object
						# is moving down) AND the centroid is below the
						# center line, count the object
						elif direction > 0 and centroid[1] > line_position:
							totalDown += 1
							empty1.append(totalDown)
							#print(empty1[-1])
							# # if the people limit exceeds over threshold, send an email alert
							# if sum(x) >= config.Threshold:
							# 	cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
							# 		cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							# 	if config.ALERT:
							# 		print("[INFO] Sending email alert..")
							# 		Mailer().send(config.MAIL)
							# 		print("[INFO] Alert sent")

							to.counted = True
					else:
						if direction < 0 and centroid[1] < line_position:
							totalDown += 1
							empty.append(totalDown)
							to.counted = True

						
						elif direction > 0 and centroid[1] > line_position:
							totalUp += 1
							empty1.append(totalUp)
							# if sum(x) >= config.Threshold:
							# 	cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
							# 		cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							# 	if config.ALERT:
							# 		print("[INFO] Sending email alert..")
							# 		Mailer().send(config.MAIL)
							# 		print("[INFO] Alert sent")

							to.counted = True
						
					x = []
					# compute the sum of total people inside
					x.append(len(empty)-len(empty1))

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to
			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# construct a tuple of information we will be displaying on the
		info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info2 = [
		("Total people inside", x),
		]
		ocu = totalDown - totalUp
		#Obtenemos el valor de la variable ocupacion anterior
		global ocu_anterior
		# Comparar el valor actual con el anterior, si es diferente guardamos ese valor en la tabla, si no pasamos
		if ocu !=ocu_anterior:
			guardar_x(ocu)
			ocu_anterior = ocu
		else:
			pass

              # Display the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

				
		#------------------Conteo mostrando imagen en pantalla--------------------------------------------		
		# show the output frame
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF
		
		
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		
		#------------------Conteo sin mostrar imagen en pantalla------------------------------------------	
		
		# if the `q` key was pressed, break from the loop
		# if cv2.waitKey(1) & 0xFF == ord("q"):
		# 	break

		#------------------------------------------------------------	


		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

		if config.Timer:
			# Automatic timer to stop the live stream. Set to 8 hours (28800s).
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	#Fin de bucle while true

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


	
	# issue 15
	if config.Thread:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()
	

##learn more about different schedules here: https://pypi.org/project/schedule/
if config.Scheduler:
	##Runs for every 1 second
	#schedule.every(1).seconds.do(run)
	##Runs at every day (09:00 am). You can change it.
	schedule.every().day.at("09:00").do(run)

	while 1:
		schedule.run_pending()

else:
	run()
