import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic  # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities
def swapVal(mp_holistic, mp_drawing):
    mp_holistic = mp.solutions.holistic  # holistic model
    mp_drawing = mp.solutions.drawing_utils # drawing utilities
    
    return mp_holistic, mp_drawing
    
def mediapipe_detection(img,model) : 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img.flags.writeable = 0
    results = model.process(img)
    img.flags.writeable = 1
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img, results
def draw_landmarks(img,results):
    mp_drawing.draw_landmarks(img,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(img,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
def draw_styled_landmarks(img,results):
    # Draw face connections 
    mp_drawing.draw_landmarks(img,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,
                              # dots color
                             mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
                              # connection lines color
                             mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)
                              
                             )
    # Draw Body land marks 
    mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2)
                              
                             )
    # Draw left hand landmarks
    mp_drawing.draw_landmarks(img,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2)
                              )
    # Draw right hand landmarks
    mp_drawing.draw_landmarks(img,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                              )

def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x,res.y,res.z,] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)  
    rh = np.array([[res.x,res.y,res.z,] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)  
    return np.concatenate([pose,face,lh,rh])
