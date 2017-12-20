from xgboost import XGBClassifier

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import normalize
from sklearn.metrics import accuracy_score
from keras.models import load_model

import sys

data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
np.random.seed(1)
np.random.shuffle(data)
data_x = data[:, 1:24]
data_y = data[:, 24]
data_y = data_y.astype(int)
# read in data
# model = XGBClassifier(max_depth=5, objective='rank:pairwise')
model = XGBClassifier(seed=1, max_depth=5, objective='rank:pairwise')

model.fit(data_x, data_y)
pred = model.predict(data_x)
print ('acc: ', accuracy_score(data_y, pred))

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

