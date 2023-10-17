## 1. Import and Install Dependencies  ##

##  pip install tensorflow opencv-python mediapipe scikit-learn matplotlib

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


### 2. Keypoints using MP Holistic #####

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results



def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2)
                             )
    
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                             )


### 3. Extract Keypoint Values ###

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose, face, lh, rh])


### 4. Setup Folders for Collection ###

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
# Ações que tentamos detectar
actions = np.array([
  #  {"texto": "cartão"},
    {"texto": "vermelho" , "inicio": 0, "fim": 6},
  #  {"texto": "cinco" }
])

# Thirty videos worth of data
no_sequences = 1

# Videos are going to be 30 frames in length
sequence_length = 30


# Define a taxa de quadros do vídeo
fps = 70

# Função para calcular os frames correspondentes a um intervalo de tempo
def calcular_frames_intervalo(inicio, fim):
    inicio_frame = int(inicio * fps)
    fim_frame = int(fim * fps)
    return inicio_frame, fim_frame


### 5. Collect Keypoint Values for Training and Testing  ####
## Colect 30 frames of video motion on respective folder MP_DATA

# Caminho para o arquivo de vídeo que você deseja carregar
video_path = 'C:/Users/jessi/Desktop/TESE/WHISPER/vermelho/vermelho.mp4'

# Inicialize a captura de vídeo usando o caminho do arquivo
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_MSEC, 2000) ##iniciar o video sem os primeiros 2 segundos de delay com a legenda
# Verifique se a captura de vídeo foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
   # NEW LOOP
    # Loop through actions
    for index, action in enumerate(actions):
        print("action: " , action )
     
        # Videos are going to be dinamic frames in length
        sequence_length = (action['fim'] - action['inicio']) * fps
        print("sequence_length: " , sequence_length)
        
        # Calcular os frames correspondentes ao intervalo de tempo da ação
        inicio_frame, fim_frame = calcular_frames_intervalo(action['inicio'], action['fim'])
        print("inicio_frame: " , inicio_frame)
        print("fim_frame: " , fim_frame)

        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):

            # Read feed
            ret, frame = cap.read()
            # Verifique se a captura foi bem-sucedida
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # print(results)
            keypoints = extract_keypoints(results)

            # Verificar se o frame atual está dentro do intervalo da ação
            if inicio_frame <= frame_num + inicio_frame <= fim_frame:
                try:
                    os.makedirs(os.path.join(DATA_PATH, "texto " + str( index), str(1), "keypoints" ))
                    os.makedirs(os.path.join(DATA_PATH, "texto " + str( index), str(1), "images" ))
                except:
                    pass

                # Salvar keypoints em arquivo .npy
                npy_path = os.path.join(DATA_PATH, "texto " + str( index), str(1), "keypoints", f'keypoints{frame_num}.npy')
                np.save(npy_path, keypoints)

                 # Exibir informações na tela
                print(f"Salvar keypoints para {action['texto']} - Frame {frame_num}")

                try:
                    
                    # Redimensionar e salvar frame como imagem (PNG)
                    resized_frame = cv2.resize(frame, (500, 500))  # Redimensionar o frame
                    frame_path = os.path.join(DATA_PATH, "texto " + str( index), str(1), "images", f'image{frame_num}.png')
                    cv2.imwrite(frame_path, resized_frame) 
                    # Exibir informações na tela
                  #  print(f"Salvar imagem para {frame_path}")

                except Exception as e:
                    print("Erro:", e)
                    
                
            # Mostrar o quadro
            cv2.imshow('Video Upload', image)
            cv2.waitKey(1)


            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
    cap.release()
    cv2.destroyAllWindows()
    
