import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# files
import glob
from os import listdir
from os.path import isfile, join

mp_pose = mp.solutions.pose # POSE model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Utils function --------------------------------------------------------------


def extract_keypoints(results):
    pose = np.array([
        [
            (results.pose_landmarks.landmark[0].x - res.x), 
            (results.pose_landmarks.landmark[0].y - res.y), 
            res.z, 
            res.visibility
        ] for res in results.pose_landmarks.landmark
        ]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    return np.concatenate([pose])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    
    return image, results
    
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                               mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )    

from tensorflow.keras.models import load_model

model = load_model('dense4-2x1000-40vid.h5')

classes = ['fall', 'normal']
classes.sort()

actions = np.array(classes)
actions

no_sequences = 80
sequence_length = 150


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

sequence = []
sentence = []
threshold = 0.8

buffer = []

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        ret, frame_original = cap.read()
        
        target_size = (240, 360)
        
        scale_percent = (target_size[0]/frame_original.shape[0], target_size[1]/frame_original.shape[1])
        width = int(frame_original.shape[1] * scale_percent[1])
        height = int(frame_original.shape[0] * scale_percent[0])
        dim = (width, height)
          
        frame = cv2.resize(frame_original, dim, interpolation = cv2.INTER_AREA)

        image, results = mediapipe_detection(frame, pose)
        
        draw_styled_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-150:]
        
        if len(sequence) == 150:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            if len(buffer) > 10:
                buffer.pop(0)
            buffer.append(actions[np.argmax(res)])
            
            most_common = max(set(buffer), key=buffer.count)
            cv2.putText(image, most_common, (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
