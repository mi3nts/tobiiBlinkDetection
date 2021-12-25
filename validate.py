import more_itertools as mit
import numpy as np
import pandas as pd

def f_score(tp, fp, fn):
    return tp/(tp+(0.5*(fp+fn)))

def validateMethods(ground_truth, predictions):
    
    # get predictions of blinks
    blinks = predictions[predictions['Classification'] == 1]
    
    # get flat list of blinks and grouped list, where consecutive indices are recognized as one blink
    blinks_flat_list = list(blinks.index)
    blinks_grouped = [list(group) for group in mit.consecutive_groups(blinks_flat_list)]
#     for i in blinks_grouped:
#         print(i)
    
    # get ground truth blinks
    gt_close = ground_truth.loc[ground_truth['closed'] == 1]
#     print(gt_close)
    
    # set up true positive, false negative counters
    tp = 0
    fn = 0
    
    # num of ground truth blinks
    gt_num = len(gt_close)
    
    # predicted number of blinks
    pred_num = len(blinks_grouped)
    
    # perform validation: see if a blink is predicted +-window_size frames around the ground truth blink record
    window_size = 10
    for i in gt_close.index:

        # create an array of -window_size to +window_size around the ground_truth blink
        i_range = np.arange(i-window_size,i+window_size+1)

        # check if previous array has any overlap with list of predicted blinks
        if len(set(blinks_flat_list) & set(i_range)) > 0:
            tp += 1
        else:
            print(i, " is not predicted within given window size")
            fn += 1


    fp = pred_num - tp 
    tn = len(predictions) - fp - tp - fn
    f_sc = f_score(tp, fp, fn)
    print("Number of blinks predicted: ", pred_num)
    print(f"True positive:  {tp}, False negative: {fn}, False positive: {fp}, True negative: {tn}, F-score: {f_sc:.2f}")
    # fig, ax = plt.subplots(figsize=(25,5))
    # ax.plot(predictions['EAR_Avg'], zorder=0)
    # ax.scatter(gt_close.index, predictions['EAR_Avg'][gt_close.index], marker='x',c='r',label='Ground truth', s=65)
    # ax.scatter(blinks_flat_list, predictions['EAR_Avg'][blinks_flat_list], marker='o', edgecolors='black', facecolors='none', label='Method', s=130)
    # ax.set_xlabel('Frame Number')
    # ax.set_ylabel('EAR Value')

    # #     ax.set_xticks(twitter_labels.index[::1000])
    # ax.legend(loc='upper right')

if __name__ == '__main__':
    static = pd.read_csv("Baseline/ear_static.csv")
    svm = pd.read_csv("Support Vector Machines/svm.csv")
    isof = pd.read_csv("Isolation Forest/if.csv")
    gt = pd.read_csv("resources/twitter_blink_groundtruth_all.csv")

    print("\nNumber of ground truth blinks : ", len(gt.loc[gt['closed'] == 1]), "\n")

    methods = {"Baseline": static, "Support Vector Machines": svm, "Isolation Forest": isof}

    for i,j in methods.items():
        print("Method: ", i)
        validateMethods(gt,j)
        print("\n")