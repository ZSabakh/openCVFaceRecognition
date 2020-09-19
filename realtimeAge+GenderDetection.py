# -*- coding: utf-8 -*-
"""
Real time age and gender detection based on pre-trained model
@author: zurab
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
        
        MEAN_VALUES_AGE = (78.4263377603, 87.7689143744, 114.895847746)
        one_face_image_blob = cv2.dnn.blobFromImage(one_face_image, 1, (227, 227), MEAN_VALUES_AGE, swapRB=False)
        
        "Gender prediction code !"
        gender_labels = ['Male', 'Female']
        gender_prototxt ="dataset/gender_deploy.prototxt"
        gender_caffemodel = "dataset/gender_net.caffemodel"
        
        "Creating model from prototxt and caffemodel"
        gender_mdl = cv2.dnn.readNet(gender_caffemodel, gender_prototxt)
        gender_mdl.setInput(one_face_image_blob)
        
        "Pushing input through neural network to get predictions"
        gender_prediction = gender_mdl.forward()
        gender = gender_labels[gender_prediction[0].argmax()]
        "End of gender prediction code !"
        
        "Age prediction code !"
        age_labels = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
        age_prototxt = "dataset/age_deploy.prototxt"
        age_caffemodel = "dataset/age_net.caffemodel"
        
        age_mdl = cv2.dnn.readNet(age_caffemodel, age_prototxt)
        age_mdl.setInput(one_face_image_blob)
        
        age_prediction = age_mdl.forward()
        age = age_labels[age_prediction[0].argmax()]
        "End of Age prediction code !"
        
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(current_frame, age+"years, " +gender, (left_pos, bottom_pos), font, 2, (0,0,0), 1)
        
    cv2.imshow("Face from feed", current_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('f'):
        "Exit key is F !!!"
        break
camera_stream.release()
cv2.destroyAllWindows
