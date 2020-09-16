# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:33:07 2020

@author: zurab
"""

import cv2
import dlib
import face_recognition

print(cv2.__version__)
print(dlib.__version__)
print(face_recognition.__version__)

detectimage = cv2.imread('images/mishabidz.jpg')

cv2.imshow("Test", detectimage)

face_locations = face_recognition.face_locations(detectimage, model='hog')

print('Found {} faces'.format(len(face_locations))) 

for index,face_location in enumerate(face_locations):
    top_pos, right_pos, bottom_pos, left_pos = face_location
    print('Found face {}, top:{}, right:{}, bottom: {}, left: {}'.format(index, top_pos, right_pos, bottom_pos, left_pos))
    one_face_image = detectimage[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow("Face: " + str(index), one_face_image)