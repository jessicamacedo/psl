import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Actions that we try to detect
actions = np.array(['ola', 'obrigada', 'bomdia', 'boanoite'])

label_map = {label:num for num, label in enumerate(actions)}

print(label_map)

# Thirty videos worth of data
no_sequences = 30
# Videos are going to be 30 frames in length
sequence_length = 30

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

count = np.array(sequences).shape
print("sequencias shape: ", count)

print("labels shape:", np.array(labels).shape)

X = np.array(sequences)
print("X shape:" , X.shape)

#converter o index da label (0,1,2,3) num array "binario"
# [1 0 0 0] = ola (first position)
# [0 0 0 1] = boanoite (last position)
Y = to_categorical(labels).astype(int)
print ("Y shape:" , Y)

#perform a testing partition size 5 from sklearn library
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

## x = frames 120
## y = labels 4
print("X_train:" , X_train.shape) ##120-5 = 114?
print("X_test:" , X_test.shape) ## 6 5??
print("y_train:" , Y_train.shape)
print("y_test:" , Y_test.shape)


## 7. Build and Train LSTM Neural Network KERAS AND TENSORFLOW ##

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# instanciar o modelo keras "Sequential" API (ver video)
## 30 frames
## 1662 points

model = Sequential()
## LAYER LSTM (units (neurons), return sequences, activation relu ou outros, input shape)
## LAYER Dense 
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) ##brincar com os valores
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
## LAYER Dense actions 4
model.add(Dense(actions.shape[0], activation='softmax')) # softmax goes 0 to 1 prediction
print("Actions shape:" , actions.shape[0]) ## 4 labels/actions

#example result
res = [.7, 0.5, 0.2, 0.1] ## the position predicted successfully is 0.7 (0->1) so its position 0 of array so its "ola"

actions[np.argmax(res)] ## array posicao 0

print("position predicted: " , np.argmax(res))
print("label predicted:" , actions[np.argmax(res)])

