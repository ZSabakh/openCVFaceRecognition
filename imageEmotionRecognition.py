# -*- coding: utf-8 -*-
"""
@author: zurab
"""

import cv2
import face_recognition
import numpy as np
from keras.preprocessing import image 
from keras.models import model_from_json

detectimage = cv2.imread('images/mishabidz.jpg')

facial_expression_model = model_from_json(open("dataset/facial_expression_model_structure.json", "r").read())
facial_expression_model.load_weights('dataset/facial_expression_model_weights.h5')
emotions_names = ('angr!', 'ew', 'fear', 'happ', 'big sad', 'surpriise', 'neutral')

face_locations = face_recognition.face_locations(detectimage, model='hog')

print('Found {} faces'.format(len(face_locations))) 

for index,face_location in enumerate(face_locations):
    top_pos, right_pos, bottom_pos, left_pos = face_location
    print('Found face {}, top:{}, right:{}, bottom: {}, left: {}'.format(index, top_pos, right_pos, bottom_pos, left_pos))
    one_face_image = detectimage[top_pos:bottom_pos, left_pos:right_pos]

    cv2.rectangle(detectimage, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)
        
    one_face_image = cv2.cvtColor(one_face_image, cv2.COLOR_BGR2GRAY)
    
    one_face_image = cv2.resize(one_face_image, (48,48))
    "Converting face into something dataset is trained for"
    
    face_pixels = image.img_to_array(one_face_image)
    face_pixels = np.expand_dims(face_pixels, axis = 0)
    face_pixels /= 255
    
    emotion_predictions = facial_expression_model.predict(face_pixels)
    
    max_value = np.argmax(emotion_predictions[0])
    
    emotion_name = emotions_names[max_value]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(detectimage, emotion_name, (left_pos, bottom_pos), font, 0.5, (255,255,255), 1)
    
    
cv2.imshow("Face from picture", detectimage)