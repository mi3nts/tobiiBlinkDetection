# CODE AUTHORED BY ARJUN SRIDHAR
# PARTS OF CODE ADAPTED FROM: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# CODE THAT CLASSIFIES USING EYE-ASPECT RATIO (EAR) METHOD USING STATIC THRESHOLD AND CONSECUTIVE NUMBER OF FRAMES
# DEPENDENCIES - SHAPE PREDICTOR FILE

# import the necessary packages
import pandas as pd
from imutils import face_utils
from scipy.spatial import distance as dist
import argparse
import imutils
import time
import dlib
import cv2

# computes the eye aspect ratio (EAR) value for the given eye frame
# inputs - the eye frame 
# outputs - the EAR value
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

# runs the blink classification using the eye-aspect ration method
# inputs - the command line arguments
# outputs - the csv file with the blink classifications
def predict(args): 
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 3
    # initialize the frame counters and the total number of blinks
    COUNTER = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = cv2.VideoCapture(args['video'])
    time.sleep(1.0)
    
    face = cv2.imread('face.jpg')
    
    ear_values = []
    blink_flags = []
    frames = []
    
    k = 0
    
    # loop over frames from the video stream
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        ret, frame = vs.read()
        
        if ret is None or frame is None:
            break
                
        frame1 = frame[250:460, 0:240] # right eye, modify values based on how big eye is
        h, w, layers = frame1.shape
        frame1 = cv2.resize(frame1, (int(w / 2), int(h / 2)))
        
        frame2 = frame[750:960, 0:240] # left eye
        h2, w2, l2 = frame2.shape
        frame2 = cv2.resize(frame2, (int(w2 / 2), int(h2 / 2)))
        
        # overwrite face image with right eye from stream, change dimensions to get eye detected
        face[220:325, 540:660] = frame1 
        
        # overwrite face image with left eye from stream, change dimensions to get eye detected
        face[235:340, 385:505] = frame2 # overwrite face image with left eye from stream
        
        frame = imutils.resize(face, width=450)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                blink_flags.append(0)
                # otherwise, the eye aspect ratio is not below the blink threshold
            else:
                # if the eyes were closed for consecutive number of blinks - classify as blink
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    blink_flags.append(1)
                else:
                    blink_flags.append(0)
                
                # reset the eye frame counter
                COUNTER = 0
                    
            ear_values.append(ear)
            frames.append('Frame%d' %(k))
            
            k = k + 1
    
    cnt = 0
    for flag in blink_flags:
        if flag == 1:
            cnt += 1
    print(cnt)
    # data for output to be stored
    data_dict = {'Frame': frames, 'EAR_Avg': ear_values, 'Blink_Flag': blink_flags}
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    
    # create dataframe and write output
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv('./ear_static.csv', index=False)
    

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
    args = vars(ap.parse_args())
    predict(args)
