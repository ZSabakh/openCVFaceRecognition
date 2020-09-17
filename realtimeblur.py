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
only applies gaussian blur effect to the realtime face detection.
!!!!!!!!!!!!!!!!!!!!
"""

import cv2
import face_recognition

camera_stream = cv2.VideoCapture(0)
" captures video from camera with index 0"

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
        one_face_image = cv2.GaussianBlur(one_face_image, (99,99), 35)
        current_frame[top_pos:bottom_pos, left_pos:right_pos] = one_face_image
        
        
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)
        
    cv2.imshow("Face from feed", current_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('f'):
        "Exit key is F !!!"
        break
camera_stream.release()
cv2.destroyAllWindows
