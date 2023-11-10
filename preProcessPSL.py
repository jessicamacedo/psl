import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['ola', 'obrigada', 'bom dia', 'boa noite'])

# Thirty videos worth of data
no_sequences = 30
# Videos are going to be 30 frames in length
sequence_length = 30
label_map = {label:num for num, label in enumerate(actions)}

print(label_map)

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

####example result
#exampleRes = [.7, 0.5, 0.2, 0.1] ## the position predicted successfully is 0.7 (0->1) so its position 0 of array so its "ola"
#actions[np.argmax(exampleRes)] ## array posicao 0
#print("position predicted: " , np.argmax(exampleRes))
#print("label predicted:" , actions[np.argmax(exampleRes)])

#compile model
##loss must be categorical_crossentropy for multiple class binary classification models
##metrics is optional
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) ##change optimizer to another for testing

#fit and train model
model.fit(X_train, Y_train, epochs=2000, callbacks=[tb_callback]) ##avaliar epochs se esta muito alto para o numero de dados e avalisar loss + accuracy



### lOGS TENSORFLOW ######

## cd C:\Users\jessi\Desktop\GIT\Sign Language recognition\lstm\Logs\train>
## RUN  tensorboard --logdir=.
## open browser

model.summary()

## ele diz que com modelo CNN precisamos de milhoes de registos e neste os parametros sao 600k

## Layer (type)                Output Shape              Param #
## =================================================================
## lstm (LSTM)                 (None, 30, 64)            442112

## lstm_1 (LSTM)               (None, 30, 128)           98816

## lstm_2 (LSTM)               (None, 64)                49408

## dense (Dense)               (None, 64)                4160

## dense_1 (Dense)             (None, 32)                2080

## dense_2 (Dense)             (None, 4)                 132
## =================================================================
## Total params: 596,708
## Trainable params: 596,708
## Non-trainable params: 0
## _________________________________________________________________


#### 8. Make Predictions -> TEST!!! ####

res = model.predict(X_test)

print("teste 0:" , actions[np.argmax(res[0])])
print("teste 0:" , actions[np.argmax(Y_test[0])])

print("teste 1:" , actions[np.argmax(res[1])])
print("teste 1:" , actions[np.argmax(Y_test[1])])

print("teste 2:" , actions[np.argmax(res[2])])
print("teste 2:" , actions[np.argmax(Y_test[2])])

print("teste 3:" , actions[np.argmax(res[3])])
print("teste 3:" , actions[np.argmax(Y_test[3])])

print("teste 4:" , actions[np.argmax(res[4])])
print("teste 4:" , actions[np.argmax(Y_test[4])])


#### 9. Save Model ####

model.save('keras.h5')

## del model
## model.load_weights('handSignPSL.h5')