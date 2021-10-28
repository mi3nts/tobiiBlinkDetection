# EXAMPLE CODE FOR USING blinkDetector_1DEmbed() i.e. ALGORITHM 4

# CODE AUTHORED BY: SHAWHIN TALEBI
# PROJECT: tobiiBlinkDetection
# GitHub: https://github.com/mi3nts/tobiiBlinkDetection

# ==============================================================================

from blinkDetector_1DEmbed import *

# define filename i.e. relative path to eyesstream.mp4 video
# NOTE: you may need to change this line to reflect your input eyesstream video
filename = 'data/eyesstream.mp4'

blink_flag = blinkDetector_1DEmbed(filename)
print(blink_flag)
