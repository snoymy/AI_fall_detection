import cv2
import numpy as np
from os.path import join

def writeVideo(frames:list, outdir:str, filename:str):
    size = (np.array(frames[0]).shape[1], np.array(frames[0]).shape[0])
    
    filename += ".mp4"
    outpath = join(outdir, filename)
    writer = cv2.VideoWriter(
            outpath, 
            cv2.VideoWriter_fourcc(*'MJPG'),
            24, size
        )
    for frame in frames:
        writer.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    
    writer.release()
    

actions = ["fall", "normal"]
no_sequences = 40

# Videos are going to be 30 frames in length
sequence_length = 150


cap = cv2.VideoCapture(0)
for action in actions:
    # Loop through sequences aka videos
    for sequence in range(no_sequences):
        frames = []
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image = frame
#                 print(results)

            # Draw landmarks
            
            # NEW Apply wait logic
            if frame_num == 0: 
                cv2.putText(image, 'STARTING COLLECTION ' + str(action.upper()), (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(5000)
            else: 
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                
            frames.append(frame)
            
            # NEW Export keypoints

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        writeVideo(frames, action, action+"_"+str(sequence))
                
cap.release()
cv2.destroyAllWindows()