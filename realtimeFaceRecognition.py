# -*- coding: utf-8 -*-
"""
@author: zurab
"""

import cv2
import face_recognition
import os

all_file_names = []
all_face_encodings = []
for filename in os.listdir('images/onlyFace'):
    if filename.endswith(".jpg"): 
        temp_image = face_recognition.load_image_file("images/onlyFace/" + filename)
        print(filename)
        temp_encoding = face_recognition.face_encodings(temp_image)[0]
        all_face_encodings.append(temp_encoding)
        all_file_names.append(filename)
    continue
print(all_file_names)

camera_stream = cv2.VideoCapture(0)
" captures video from camera with index 0"

face_locations = []

while True: 
    brk, current_frame = camera_stream.read()
    current_frame_crop = cv2.resize(current_frame,(0,0), fx=0.25, fy=0.25)
    
    face_locations = face_recognition.face_locations(current_frame_crop, number_of_times_to_upsample=2, model='hog')

    face_encodings = face_recognition.face_encodings(current_frame_crop, face_locations)
    
    for one_face_location, one_face_encoding in zip(face_locations, face_encodings):
        top_pos, right_pos, bottom_pos, left_pos = one_face_location
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4
        left_pos *= 4
        
        matches = face_recognition.compare_faces(all_face_encodings, one_face_encoding)
        name = "Not found"
    
        if True in matches:
            match_index = matches.index(True)
            name = all_file_names[match_index]
            
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        print('Name:  ' + name)
        cv2.putText(current_frame, name, (left_pos, bottom_pos), font, 1, (0,0,0), 1)
    cv2.imshow("Detekt", current_frame)



        
    if cv2.waitKey(1) & 0xFF == ord('f'):
        "Exit key is F !!!"
        break
camera_stream.release()
cv2.destroyAllWindows