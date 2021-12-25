# Tobii Pro Glasses 2 Blink Detection and Validation 
Blink detection methods for Tobii Pro Glasses 2 data

This repo acts as both an implementation of methods proposed in the paper "Unsupervised Blink Detection Using Eye Aspect Ratio Values" and a validation of the results provided. 

To recreate the environment using conda:
```
conda env create -f environment.yml
```

The methods used are labeled as:

- Baseline 
- Support Vector Machines
- Isolation Forest

To reproduce the paper's results, one may clone the repo and simply run 
```
validate.py
```
 as each method's respective predictions are provided. 

To make new predictions, simply navigate to the desired method's directory (eg. SVM) and execute as follows:
```python
python ear_svm.py -p ../resources/shape_predictor_68_face_landmarks.dat -v ../resources/twitter_eyesstream.mp4
```

