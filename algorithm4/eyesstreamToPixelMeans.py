# FUNCTION TO GET AVERAGE PIXEL VALUE FROM EACH EYESSTREAM.MP4 VIDEO FRAME

# CODE AUTHORED BY: SHAWHIN TALEBI
# PROJECT: tobiiBlinkDetection
# GitHub: https://github.com/mi3nts/tobiiBlinkDetection

# INPUTS
# - filename = string. path and filename of eyesstream.mp4 video

# OUTPUTS
# - pixel_means = 1D numpy array of pixel means for each frame in specified
# eyesstream.mp4 video

# PROJECT DEPENDENCIES
# - none

# PROJECT DEPENDERS
# - blinkDetector_1DEmbed()

# ==============================================================================

# import modules
import cv2
import numpy as np

def eyesstreamToPixelMeans(filename):

    # initialize numpy array to store mean pixel values for each frame
    pixel_means = np.empty([0,0])

    # read video
    cap = cv2.VideoCapture(filename)

    # Read until video is completed
    while(cap.isOpened()):

        # read next frame
        ret, frame = cap.read()

        # if frame is returned convert it to grayscale
        if ret == True:

            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # compute avergae grayscale pixel value
            cur_frame_mean = np.mean(gray)

            # append current frame mean to pixel means array
            pixel_means = np.append(pixel_means, cur_frame_mean)

        # if frame is not returned break out of loop
        else:
            break

            # When everything done, release the video capture object

    # release video capture object
    cap.release()


    return pixel_means
