# FUNCTION TO PRODUCE BLINK FLAGS BASED ON MEAN PIXEL VALUES FROM EYESSTREAM.MP4
# VIDEO FRAMES

# CODE AUTHORED BY: SHAWHIN TALEBI
# PROJECT: tobiiBlinkDetection
# GitHub: https://github.com/mi3nts/tobiiBlinkDetection

# INPUTS
# - pixel_means = 1D numpy array of pixel means for each frame in specified
# eyesstream.mp4 video

# OUTPUTS
# - blink_flags = 1D numpy array indicating blinks, where 0 means no blink and
# 1 means blink

# PROJECT DEPENDENCIES
# - none

# PROJECT DEPENDERS
# - blinkDetector_1DEmbed()

# ==============================================================================

# import modules
import numpy as np
from scipy.signal import find_peaks

def pixelMeansToBlinkFlags(pixel_means):
    # perform transformations
    transformation1 = np.gradient(pixel_means);
    transformation2 = np.power(transformation1, 2)
    transformation2_median = np.median(transformation2)
    transformation3 = transformation2 - transformation2_median
    transformation4 = np.power(transformation3, 2)

    # find peaks in transformation 4 signal
    pks, __ = find_peaks(transformation4, distance= 10, prominence= 0.5)

    # create binary blink flag array
    blink_flag = np.zeros(transformation4.shape)
    blink_flag[pks] = 1

    # output blink_flag
    return blink_flag
