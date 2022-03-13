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

#------------------------------------------------------------------------------

# Setup folder ----------------------------------------------------------------

DATA_PATH = os.path.join('MP_Data') 

classes = [i.split(os.path.sep)[1] for i in glob.glob('./train_video/*')]
classes.sort()
dataset_root_path = './train_video/'

actions = np.array(classes)
actions

no_sequences = 40
sequence_length = 150

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

#------------------------------------------------------------------------------

# Collecting data -------------------------------------------------------------

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    for action in actions:
        files = [f for f in listdir(dataset_root_path + str(action)) if isfile(join(dataset_root_path + str(action), f))]
        files = files[:no_sequences]
        
        for sequence, file in enumerate(files):
            video_path = dataset_root_path + action + "/" + file
            cap = cv2.VideoCapture(video_path)
            print('Collecting frames for {} Video Number {}'.format(action, sequence))
            prev_frame = None
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()
                
                if ret is not True:
                    frame = prev_frame
                else:
                    prev_frame = frame

                image, results = mediapipe_detection(frame, pose)
                # print(results)

                draw_styled_landmarks(image, results)
                
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    #cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(10)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                #print(keypoints.shape)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

#------------------------------------------------------------------------------

# Preprocess data and create labels and features ------------------------------

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            except:
                continue
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

#------------------------------------------------------------------------------

# Build Model -----------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(16, return_sequences=False, activation='relu', input_shape=(sequence_length,132)))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

res = [.7, 0.2, 0.1]

actions[np.argmax(res)]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

#------------------------------------------------------------------------------

# Train -----------------------------------------------------------------------

history = model.fit(X_train, y_train, validation_split = 0.25, epochs=1000, callbacks=[tb_callback])

#------------------------------------------------------------------------------

# Plot ------------------------------------------------------------------------

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#------------------------------------------------------------------------------

model.save('dense4-2x1000-40vid.h5')

# Predict ---------------------------------------------------------------------

res = model.predict(X_test)
actions[np.argmax(res[0])]
actions[np.argmax(y_test[0])]

#------------------------------------------------------------------------------

del model

# Real time test --------------------------------------------------------------

from tensorflow.keras.models import load_model

model = load_model('dense4-2x1000-40vid.h5')

classes = ['fall', 'normal']
classes.sort()
dataset_root_path = './test_video/'

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
    files = [f for f in listdir(dataset_root_path) if isfile(join(dataset_root_path, f))]
    files = files[:no_sequences]
    
    for _, file in enumerate(files):
        video_path = join(dataset_root_path, file)
        #cap = cv2.VideoCapture(video_path)

        while cap.isOpened():

            ret, frame_original = cap.read()
            
            if ret is not True:
                frame_original = prev_frame
            else:
                prev_frame = frame_original
            
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

# -----------------------------------------------------------------------------

