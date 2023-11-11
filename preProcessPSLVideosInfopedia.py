import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard
from moviepy.editor import VideoFileClip


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data_Infopedia') 

# Diretório que contém os folders com vídeos
diretorio_total = 'C:/Users/jessi/Desktop/TESE/MODELS-PSL/GIT/lgp-lstm/infopediavideos'

# Obter uma lista de nomes de pastas no diretório
nomes_pasta = [nome for nome in os.listdir(diretorio_total) if os.path.isdir(os.path.join(diretorio_total, nome))]

# Criar o array 'actions'
actions =  np.asarray(nomes_pasta)

print("Nomes das pastas:")
print(actions)

label_map = {label:num for num, label in enumerate(actions)}
print('label_map: ' , label_map)


sequences, labels = [], []
for action in actions:
    print('action: ', action)

    # Retornar o diretorio da action em questao
    diretorio_video = os.path.join('C:/Users/jessi/Desktop/TESE/MODELS-PSL/GIT/lgp-lstm/infopediavideos/', action)
    print('diretorio_video: ' , diretorio_video)

    # Listar os arquivos na pasta
    videos_na_pasta = [f for f in os.listdir(diretorio_video) if f.endswith('.mp4')]
    print('videos_na_pasta: ' , videos_na_pasta)

    for video_nome in videos_na_pasta:
        print('video_nome: ', video_nome)
        # Caminho completo para o vídeo atual
        video_path = os.path.join(diretorio_video, video_nome)
        print('video_path: ' , video_path)

        n_video = videos_na_pasta.index(video_nome)
        print('n_video: ' , n_video)

        clip = VideoFileClip(video_path)
       
        # Videos are going to be dynamic calculated frames in length with 30 FPS
        n_video_sequence_length = 30 #int(clip.duration) * 30
        print('n_video_sequence_length: ' , n_video_sequence_length)
        clip.close()

        window = []
        try:
            for frame_num in range(n_video_sequence_length):
                npyPath = os.path.join(DATA_PATH, action, str(n_video), "{}.npy".format(frame_num))
                #print ('npyPath: ', npyPath)
                res = np.load(npyPath)
                window.append(res)
            sequences.append(window)
          ##  print('sequences: ', sequences)
            labels.append(label_map[action])
        except FileNotFoundError:
            # Se o arquivo não for encontrado
            print(f"Arquivo não encontrado")


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

##
##
##
##sequencias shape:  (3,)
##labels shape: (3,)
##X shape: (3,)
##Y shape: [[1 0]
## [0 1]
## [0 1]]
##X_train: (2,)
##X_test: (1,)
##y_train: (2, 2)
##y_test: (1, 2)
##Actions shape: 2
##


## 7. Build and Train LSTM Neural Network KERAS AND TENSORFLOW ##

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# instanciar o modelo keras "Sequential" API (ver video)
## 30 frames
## 1662 points

model = Sequential()
## LAYER LSTM (units (neurons), return sequences, activation relu ou outros, input shape)
## LAYER Dense 
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) ##adicionar camada com 30 frames
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60,1662))) ##adicionar camada com 60 frames
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(90,1662))) ##adicionar camada com 90 frames
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


#### 9. Save Model ####
model.save('infopedia.h5')