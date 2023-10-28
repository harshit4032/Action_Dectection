import os 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
# import cv2
# import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
import seaborn as sn
import holistics
# def mediapipe_detection(img,model) : 
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img.flags.writeable = 0
#     results = model.process(img)
#     img.flags.writeable = 1
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     return img, results
# def draw_landmarks(img,results):
#     mp_drawing.draw_landmarks(img,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
#     mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
#     mp_drawing.draw_landmarks(img,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(img,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
# def draw_styled_landmarks(img,results):
#     # Draw face connections 
#     mp_drawing.draw_landmarks(img,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,
#                               # dots color
#                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
#                               # connection lines color
#                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)
                              
#                              )
#     # Draw Body land marks 
#     mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),
#                              mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2)
                              
#                              )
#     # Draw left hand landmarks
#     mp_drawing.draw_landmarks(img,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
#                              mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2)
#                               )
#     # Draw right hand landmarks
#     mp_drawing.draw_landmarks(img,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
#                              mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
#                               )

# def extract_keypoints(results):
#     pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
#     lh = np.array([[res.x,res.y,res.z,] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)  
#     rh = np.array([[res.x,res.y,res.z,] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)  
#     return np.concatenate([pose,face,lh,rh])

DATA_PATH = os.path.join('MP_DATA')

#  actions that we try to detect
actions  = np.array(['HELLO',"THANKS","VICTORY"])
# 30 videos worth of data
no_sequences= 30

#  videos are going to be 30 frames in length
sequence_lenght = 30 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num ,label in enumerate(actions)}

sequencs, labels = [],[]
for action in actions:
    for sequence in range(no_sequences):
        window=[]
        for frame_num in range(sequence_lenght):
            res= np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
        sequencs.append(window)
        labels.append(label_map[action])

X = np.array(sequencs)
y = to_categorical(labels).astype(int)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.05)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = tf.keras.models.load_model('Actionsv2.h5')
# print(model.summary())
res = np.argmax(model.predict(X_test))
y_hat = model.predict(X_test)
y_hat_labels = [np.argmax(i) for i in y_hat]
yhat = model.predict(X_train)
ytrue = np.argmax(y_train,axis=1).tolist()
yhat = np.argmax(yhat,axis=1).tolist()
cm = multilabel_confusion_matrix(ytrue,yhat)
print(cm)
plt.figure(figsize=(10,7))
sn.heatmap(cm[0],annot=True,fmt='g')
plt.title("HELLO")
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()
plt.figure(figsize=(10,7))
sn.heatmap(cm[1],annot=True,fmt='g')
plt.title("THANK YOU")
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()
plt.figure(figsize=(10,7))
sn.heatmap(cm[2],annot=True,fmt='g')
plt.title("VICTORY")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()