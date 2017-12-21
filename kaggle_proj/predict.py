from xgboost import Booster, XGBClassifier

import numpy as np
from keras.utils import normalize
from sklearn.metrics import accuracy_score

import sys

model = XGBClassifier(seed=1, max_depth=5, objective='rank:pairwise')

booster = Booster()
booster.load_model('myModel')
model._Booster = booster

def generateOutput(inputFile, outputFile):

    data_t = np.genfromtxt(inputFile, delimiter=',', skip_header=1)
    
    prediction = model.predict_proba(data_t[:, 1:])
    prediction = normalize(prediction[:, 1])[0]
    
    myPrediction = np.zeros((len(data_t)))
    
    for i in range(len(data_t)):
        myPrediction[i] = prediction[i]
    
    ids = []
    
    for i, item in enumerate(data_t):
        ids.append((i, item[0]))
    
    def getValue(myId):
        return myPrediction[myId[0]]
    
    priority = sorted(ids, key=getValue, reverse=True)
    
    output = open(outputFile, 'w')
    output.write('RANK_ID')
    
    for p in priority:
        output.write('\n' + str(int(p[1])))
    
    output.close()

generateOutput(sys.argv[1], 'public.csv')
generateOutput(sys.argv[2], 'private.csv')

