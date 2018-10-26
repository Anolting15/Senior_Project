import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import face_recognition
from tkinter import *


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        crop_img = frame[y: y+h, x: x+w]
        cv2.imwrite('unknown_image.jpg', crop_img)
        break

#once the image is captured, turn off the camera 
video_capture.release()
cv2.destroyAllWindows()

#now we load the files into numpy arrays
known_image = face_recognition.load_image_file("known_image.jpg")
unknown_image = face_recognition.load_image_file("unknown_image.jpg")

#now we will get the face encoding for each face in each image file
#i set these index to 0 because there is only one face in each picture
#if there were more than one face we would have to use a different method
try:
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]

#make an exception if the program doesn't find any faces within the image
except IndexError:
        print("I wasn't able to locate any faces in one or more of the images. Check the image files. TERMINATING PROGRAM :(")

#create an array to compare the unknown faces to
known_faces = [
    known_image_encoding
    ]

#The result of this is going to return in a true/false type way telling us if the unknown face matched the face of someone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_image_encoding)

print("The person who was just in the camera is the person that is in the picture. {}".format(results[0]))

#display the two images



