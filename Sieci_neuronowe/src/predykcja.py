import csv
import numpy as np
import os
import time
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

#AMD GPU LOAD for Nvidia comment this part
import plaidml.keras
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
plaidml.keras.install_backend()

data = []
guess = []
accuracy = 0.0
correct_predictions = 0

accurate_classes = []
cur_path = os.getcwd()

flag = 0
with open('Test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if flag == 0:
            flag = 1
        else:
            accurate_classes.append(int(row[6]))
            

path = os.path.join(cur_path,'Test')     
images = os.listdir(path)
images.sort()

for a in images:
    try:             
        image = Image.open(path +'//' + a)             
        image = image.resize((30,30))                                              
        image = np.array(image)             
        data.append(image)
    except Exception as e:
        print(e)
            
model = load_model('traffic_signs_recognition_model.h5')

data = np.array(data)
data_len = len(accurate_classes)

predictions = model.predict(x=data, batch_size=16, verbose=2)

i = 0
for i in range(data_len):
    prediction = predictions[i]
    index_min = max(range(len(prediction)), key=prediction.__getitem__)
    guess.append(index_min)
    if accurate_classes[i] == guess[i]:
        correct_predictions += 1
        
        
accuracy = 100.0*(correct_predictions/data_len)
print("Model accuracy is: ", accuracy,"%\n")
    
