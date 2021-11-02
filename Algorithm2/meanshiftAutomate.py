# FUNCTION TO AUTOMATE THRESHOLD OF EAR VALUES USING KMEANS MEANSHIFT ALGORITHM
# CAPTURED BY THE TOBII PRO GLASSES 2

# CODE AUTHORED BY: ASHEN FERNANDO
# PROJECT: tobiiBlinkDetection
# GitHub: https://github.com/mi3nts/tobiiBlinkDetection

# INPUTS
# - data = dataframe produced from ear_static.py in Algorithm1

# OUTPUTS
# - roll_grad_df = dataframe with processed EAR Values

# PROJECT DEPENDENCIES
# - ear_static.predict()

# PROJECT DEPENDERS

'''
NOTE: Pipeline is incomplete
'''

# ==============================================================================

import pandas as pd
from sklearn.cluster import estimate_bandwidth, MeanShift

def meanshift_threshold(data):

    # data

    bw = estimate_bandwidth(data['EAR_Avg'])

    meanshift_avg = MeanShift(bandwidth=bw).fit(data['EAR_Avg']).to_numpy().reshape(-1,1)

    threshold = meanshift_avg.cluster_centers_[1][0]

    return threshold



