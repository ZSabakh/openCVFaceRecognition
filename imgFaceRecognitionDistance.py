# -*- coding: utf-8 -*-
""" 
@author: zurab
"""

import cv2
import dlib
import face_recognition

image_detect_path = "images/misha.jpg"

starting_image = cv2.imread(image_detect_path)

misha_image = face_recognition.load_image_file('images/misha.jpg')
misha_face_encodings = face_recognition.face_encodings(misha_image)[0]

bidzina_image = face_recognition.load_image_file('images/bidzina.jpg')
bidzina_face_encodings = face_recognition.face_encodings(bidzina_image)[0]


all_face_encodings = [misha_face_encodings, bidzina_face_encodings]
all_face_names = ["Mikheil Saakashvili", "Bidzina Ivanishvili"]

image_to_detect = face_recognition.load_image_file(image_detect_path)
image_to_detect_encodings = face_recognition.face_encodings(image_to_detect)[0]
face_distances = face_recognition.face_distance(all_face_encodings, image_to_detect_encodings)

for i,face_distance in enumerate(face_distances):
    print("Face distance is {} from {}".format(face_distance,all_face_names[i]))
    print("Matches {}% to {}".format(round(((1-float(face_distance))*100),2),all_face_names[i]))
