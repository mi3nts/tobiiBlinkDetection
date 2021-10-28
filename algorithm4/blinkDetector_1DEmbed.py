# FUNCTION TO PERFORM 1-D EMBEDDING OF PIXEL SPACE BASED ON EYESSTREAM.MP4 VIDEO
# CAPTURED BY THE TOBII PRO GLASSES 2

# CODE AUTHORED BY: SHAWHIN TALEBI
# PROJECT: tobiiBlinkDetection
# GitHub: https://github.com/mi3nts/tobiiBlinkDetection

# INPUTS
# - filename = string. path and filename of eyesstream.mp4 video

# OUTPUTS
# - blink_flags = 1D numpy array indicating blinks, where 0 means no blink and
# 1 means blink

# PROJECT DEPENDENCIES
# - eyesstreamToPixelMeans()
# - pixelMeansToBlinkFlags()

# PROJECT DEPENDERS

# ==============================================================================

from eyesstreamToPixelMeans import *
from pixelMeansToBlinkFlags import *

def blinkDetector_1DEmbed(filename):

    # convert eyestream video into frame pixel means
    pixel_means = eyesstreamToPixelMeans(filename)

    # convert frame pixel means into blink flag
    blink_flag = pixelMeansToBlinkFlags(pixel_means)

    return blink_flag
