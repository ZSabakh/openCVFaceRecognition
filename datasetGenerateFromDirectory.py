# -*- coding: utf-8 -*-
"""

@author: zurab
"""
import face_recognition
import os
from numpy import asarray
from numpy import save
import time

start = time.time()

all_file_names = []
all_face_encodings = []

for filename in os.listdir('images/onlyFace'):
    try:
        temp_image = face_recognition.load_image_file("images/onlyFace/" + filename)
        if(face_recognition.face_encodings(temp_image)):
            all_file_names.append(filename)
            print(filename)
            temp_encoding = face_recognition.face_encodings(temp_image)[0]
            all_face_encodings.append(temp_encoding)
            continue
    except:
        pass
    
end = time.time()
print(end - start)

data = asarray(all_face_encodings)
save('dataset/encodings.npy', data)
data1 = asarray(all_file_names)
save('dataset/names.npy', data1)
