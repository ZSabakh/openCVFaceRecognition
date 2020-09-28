# -*- coding: utf-8 -*-
""" 
@author: zurab
"""

import cv2
import face_recognition
from numpy import load

import time

start = time.time()
all_file_names = load('dataset/names.npy')
all_face_encodings = load('dataset/encodings.npy')

def roundToPercentage(x):
    x = round(((1-float(x))*100),2)
    return x

image_to_scan = "images/mishabidz.jpg"

starting_image = cv2.imread(image_to_scan)


image_to_detect = face_recognition.load_image_file(image_to_scan)
faces_locations = face_recognition.face_locations(image_to_detect, model="hog", number_of_times_to_upsample=2)

for x in range(len(faces_locations)):
    faces_encodings = face_recognition.face_encodings(image_to_detect, faces_locations, num_jitters=3, model="small")[x]
    face_distances = face_recognition.face_distance(all_face_encodings, faces_encodings)



name_index = {}

for i,face_distance in enumerate(face_distances):
    name_index.update( {all_file_names[i] : face_distance} )


name_index = {k: v for k, v in sorted(name_index.items(), key=lambda item: item[1])}



for one_face_location, one_face_encoding in zip(faces_locations, faces_encodings):
    top_pos, right_pos, bottom_pos, left_pos = one_face_location
    cv2.rectangle(starting_image, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)

font = cv2.FONT_HERSHEY_DUPLEX

for x in range(len(faces_locations)):
    print('Probability: {}% Name: {}'.format(roundToPercentage(list(name_index.values())[x]),list(name_index.keys())[x]))

name = list(name_index.keys())[0]
end = time.time()
print('Took {} seconds to compute'.format(end - start))

cv2.putText(starting_image, name, (left_pos, bottom_pos), font, 1, (0,0,0), 1)
cv2.imshow("Detekt", starting_image)
cv2.waitKey(0)