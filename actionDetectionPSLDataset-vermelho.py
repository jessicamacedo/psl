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
actions = np.array(['vermelho'])

# 1 video worth of data
no_sequences = 1

# Videos are going to be 30 frames in length
sequence_length = 30


### 5. Collect Keypoint Values for Training and Testing  ####
## Colect 30 frames of video motion on respective folder MP_DATA

# Diretório que contém os vídeos
diretorio_videos = 'C:/Users/jessi/Desktop/TESE/WHISPER/vermelho'

# Listar os arquivos na pasta
videos_na_pasta = [f for f in os.listdir(diretorio_videos) if f.endswith('.mp4')]
print(videos_na_pasta)


for action in actions: 
    for video_nome in videos_na_pasta:    
        #dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        for sequence in range(no_sequences):
            print(sequence)
        #for sequence in range(1,no_sequences+1):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence),video_nome.replace(".mp4", ""), "keypoints"))
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence),video_nome.replace(".mp4", ""), "images"))
                print('!!!!')
            # print(os.path.join(DATA_PATH, action, str(sequence)))
                #os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
            except:
                pass




for video_nome in videos_na_pasta:
    print(video_nome)
     # Caminho completo para o vídeo atual
    video_path = os.path.join(diretorio_videos, video_nome)
    print(video_path)
    # Inicialize a captura de vídeo usando o caminho do arquivo
    cap = cv2.VideoCapture(video_path)


    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # NEW LOOP
        # Loop through actions
        for action in actions:

            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()
                print(ret)

                if not ret:
                    # Se não foi possível ler o frame, saia do loop ou trate o erro de alguma forma
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
            # print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image,'Starting Collection',(120,200),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                    cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), video_nome.replace(".mp4", ""), "keypoints", str(frame_num))
                np.save(npy_path, keypoints)
                #Show to Screen


                try:
                    # Redimensionar e salvar frame como imagem (PNG)
                    resized_frame = cv2.resize(frame, (500, 500))  # Redimensionar o frame
                    frame_path = os.path.join(DATA_PATH, action, str(sequence), video_nome.replace(".mp4", ""),  "images", f'image{frame_num}.png')
                    cv2.imwrite(frame_path, resized_frame) 
                    # Exibir informações na tela
                #  print(f"Salvar imagem para {frame_path}")

                except Exception as e:
                    print("Erro:", e)


                cv2.imshow('Dataset collection', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
                        
        cap.release()
        cv2.destroyAllWindows()
        
