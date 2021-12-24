# FUNCTION TO AUTOMATE THRESHOLD OF EAR VALUES USING A COMBINATION OF ROLLING MEAN AND STANDARD DEVIATION
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
import more_itertools as mit

def automate_threshold(data):
    
    df = pd.DataFrame(data)
    
    # extract Left, Right and Average EAR values
    df_EAR = df.iloc[:,3:]

    # implement a moving average of 5 records    
    roll_df = df_EAR.rolling(5).mean()

    # calculate the gradient     
#     roll_grad = np.gradient(roll_df.to_numpy())

    # moving average + gradient information in a dataframe for processing    
    roll_grad_df = pd.DataFrame(roll_df[0], columns=['Left_EAR_proc','Right_EAR_proc','EAR_Avg_proc'])
    
    # pks will be the locations where threshold value is 3 times the standard deviation (away from the mean) 
    pks = roll_grad_df.loc[(roll_grad_df['EAR_Avg_proc'] < roll_grad_df['EAR_Avg_proc'].mean()-3*roll_grad_df['EAR_Avg_proc'].std())].index.to_numpy().tolist()
    
    # creates a list of lists, grouping consecutive pks into one list
    blinks_list = [list(group) for group in mit.consecutive_groups(pks)]
    
    # blink_ind contains blink flags and is subsequently introduced as a column in the final dataframe
    blink_ind = []
    for i in blinks_list:
        if len(i) == 1:
            continue
        blink_ind.append(i[0])
    
    for i,j in enumerate(blink_ind):
        roll_grad_df.loc[roll_grad_df.index[j], 'Blink_Flag_EAR_proc'] = 1

    # for the blink flag column, set empty rows to zero        
    roll_grad_df['Blink_Flag_EAR_proc'] = roll_grad_df['Blink_Flag_EAR_proc'].fillna(0)

    # keep a running count of blinks, but this is only applied at every instance of a detected blink 
    blink_count = [0 for i in range(len(roll_grad_df))]

    k = 1
    for i in blink_ind:
        blink_count[i] = k
        k = k + 1
    
    roll_grad_df.insert(loc=len(roll_grad_df.columns),column="Cumulative_blink_proc",value=blink_count)
    
    # return final dataframe containing: Timestamp, Left EAR processed, Right EAR processed, EAR Average processed, blink flag, cumulative blinks

    roll_grad_df.insert(loc=0,column='Timestamp,value',value=df.iloc[:,0])
    roll_grad_df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], unit='ms')

    return roll_grad_df