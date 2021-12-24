# CODE AUTHORED BY ARJUN SRIDHAR
# PARTS OF CODE ADAPTED FROM: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# CODE THAT TRAINS AN SVM MODEL BASED ON EYE-ASPECT-RATIO (EAR) FEATURES AND RETURNS A CSV WITH BLINK CLASSIFICATIONS
# DEPENDENCIES - SHAPE PREDICTOR FILE, download here http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz
# DEPENDENCIES - EAR_OPEN.CSV AND EAR_CLOSED.CSV TRAINING DATASETS

# import the necessary packages
import pandas as pd
from sklearn.svm import SVC
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from sklearn.model_selection import GridSearchCV

# function that merges the training data from the open and closed eye training datasets
# inputs - None
# outputs - the dataframe with the merged data
def merge_training_data():
    ear_open = pd.read_csv('./ear_open.csv') # read in training data for open eyes
    ear_closed = pd.read_csv('./ear_closed.csv') # read in training data for closed eyes
    
    training_data = pd.concat([ear_open, ear_closed])
    
    training_data = training_data[['EAR_Avg', 'Classification']]
    
    return training_data

# trains the model using the 13 feature dimension
# inputs - the training dataset
# outputs - the model used for predicting
def train_model(training_data):
    X = training_data['EAR_Avg'].to_numpy()
    
    y = training_data['Classification'].to_numpy()
    
    y = y[6:] 
    
    X_train = np.zeros((len(X) - 6, 13)) # 13 feature dimension window for each training example
    j = 0
    
    for i in range(6, len(X) - 6):
        features = X[i - 6: i + 7]
        X_train[j] = features # get 13 features for each training example
        
        j += 1
    
    params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
          #'gamma': [0.0001, 0.001, 0.01, 0.1],
          'kernel':['linear']}
    
    svc = GridSearchCV(SVC(), params_grid) # train model using GridSearch to find best parameters
    
    print(np.shape(X_train))
    print(np.shape(y))
    svc.fit(X_train, y)
    print(svc.best_params_)
    return svc

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


def predict(args):
    model = train_model(merge_training_data())    
    
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
    
    face = cv2.imread('../resources/face.jpg')
    
    ear_test = []
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
            
            ear_test.append(ear)
            frames.append('Frame%d' %(k))
            
            k = k + 1
    
    data_dict = {'Frame':[], 'EAR_Avg': [], 'Classification': []}
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    #vs.stop()
    
    #prediction = []
    
    for i in range(6, len(ear_test) - 6):
        feature_test = np.array(ear_test[i - 6: i + 7])
        
        prediction = model.predict(feature_test.reshape((1, 13)))[0]
        
        data_dict['Frame'].append(frames[i])
        data_dict['EAR_Avg'].append(ear_test[i])
        data_dict['Classification'].append(prediction)
    
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv('./svm.csv', index=False)
    

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
    args = vars(ap.parse_args())
    predict(args)