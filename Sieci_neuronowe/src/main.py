import numpy as np
import os
import time
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model


epochs = 1
data =[]
labels = []
classes =43
warunek = 1
val_acc_p = 0.0
cur_path = os.getcwd()
print(cur_path)
for i in range(classes):     
    path = os.path.join(cur_path,'Train',str(i))     
    images = os.listdir(path)
    for a in images:
        try:             
            image = Image.open(path +'//' + a)             
            image = image.resize((30,30))                                              
            image = np.array(image)             
            data.append(image)             
            labels.append(i)
        except Exception as e:
            print(e)

data = np.array(data) 
labels = np.array(labels)
print(data.shape, labels.shape) 


X_train, X_test, y_train, y_test =train_test_split(data, labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 

y_train = to_categorical(y_train,43) 
y_test = to_categorical(y_test,43)

start = time.time()
while warunek:
  if os.path.exists('traffic_signs_recognition_model.h5'):
    model = load_model('traffic_signs_recognition_model.h5')
    history = model.fit(X_train, y_train, batch_size=16, epochs=epochs, validation_data=(X_test, y_test), verbose=2)
    print('training done nr ' , i, '\n')
    acc = history.history['val_accuracy']
    val_acc = acc[-1]
    if (val_acc+0.02) > val_acc_p:
      if val_acc > val_acc_p:
        model.save('traffic_signs_recognition_model.h5')
        print('saving model\n')
        val_acc_p = val_acc
    else:
      warunek = 0
  else:
    model =Sequential() 
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:])) 
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu')) 
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu')) 
    model.add(MaxPool2D(pool_size=(2,2))) 
    model.add(Dropout(rate=0.25)) 
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu')) 
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu')) 
    model.add(MaxPool2D(pool_size=(2,2))) 
    model.add(Dropout(rate=0.25)) 
    model.add(Flatten()) 
    model.add(Dense(256, activation='relu')) 
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=16, epochs=epochs, validation_data=(X_test, y_test))
    model.save('traffic_signs_recognition_model.h5')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    i = epochs

  i += epochs

end = time.time()
elapsed_time = end - start
print("time of model compiling and fitting", elapsed_time, "\n")