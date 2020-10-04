# -*- coding: utf-8 -*-
"""

@author: zurab
"""
import os
import face_recognition
from numpy import load
from numpy import save
from numpy import asarray
import numpy as np
import time
start = time.time()

first_file_names = []
first_file_names = load('dataset/names.npy')
first_face_encodings = []
first_face_encodings = load('dataset/encodings.npy')

second_file_names = []
second_file_names = load('dataset/names1.npy')
second_face_encodings = []
second_face_encodings = load('dataset/encodings1.npy')

all_file_names = np.concatenate((first_file_names, second_file_names))
all_face_encodings = np.concatenate((first_face_encodings, second_face_encodings))


    
print(all_file_names[len(all_file_names) - 1])  

end = time.time()
print(end - start)
data = asarray(all_face_encodings)
save('dataset/encodings.npy', data)
data1 = asarray(all_file_names)
save('dataset/names.npy', data1)