# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:44:18 2020

@author: zurab
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:54:51 2020

@author: zurab
"""
"""
!!!!!!!!!!!!!!!!!!!!
Applies facial emotion detection to other files.
!!!!!!!!!!!!!!!!!!!!
"""

import cv2
import face_recognition
import numpy as np
from keras.preprocessing import image 
from keras.models import model_from_json

camera_stream = cv2.VideoCapture(0)
" captures video from camera with index 0"
facial_expression_model = model_from_json(open("dataset/facial_expression_model_structure.json", "r").read())
facial_expression_model.load_weights('dataset/facial_expression_model_weights.h5')

emotions_names = ('angr!', 'ew', 'fear', 'happ', 'big sad', 'surpriise', 'neutral')

face_locations = []

while True: 
    brk, current_frame = camera_stream.read()
    current_frame_crop = cv2.resize(current_frame,(0,0), fx=0.25, fy=0.25)
    
    face_locations = face_recognition.face_locations(current_frame_crop, number_of_times_to_upsample=2, model='hog')

    for index,face_location in enumerate(face_locations):
        top_pos, right_pos, bottom_pos, left_pos = face_location
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4
        left_pos *= 4
        
        print('Found face {}, top:{}, right:{}, bottom: {}, left: {}'.format(index, top_pos, right_pos, bottom_pos, left_pos))
        one_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
        
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)
        
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
        cv2.putText(current_frame, emotion_name, (left_pos, bottom_pos), font, 0.5, (255,255,255), 1)
        
        
    cv2.imshow("Face from feed", current_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('f'):
        "Exit key is F !!!"
        break
camera_stream.release()
cv2.destroyAllWindows
