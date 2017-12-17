from xgboost import XGBClassifier

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import normalize
from sklearn.metrics import accuracy_score
from keras.models import load_model

batch_size = 16000
epochs = 1000

data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
np.random.seed(1)
np.random.shuffle(data)
data_x = data[:, 1:24]
data_y = data[:, 24]
data_y = data_y.astype(int)
# read in data
model = XGBClassifier(max_depth=10, objective='rank:pairwise')

model.fit(data_x, data_y)
pred = model.predict(data_x)
print ('acc: ', accuracy_score(data_y, pred))

data_t = np.genfromtxt('Test_Public.csv', delimiter=',', skip_header=1)

prediction = model.predict_proba(data_t[:, 1:])
prediction = normalize(prediction[:, 1])

myPrediction = np.zeros((len(data_t)))

for i in range(len(data_t)):
    myPrediction[i] = prediction[0][i]

ids = []

for i, item in enumerate(data_t):
    ids.append((i, item[0]))

def getValue(myId):
    return myPrediction[myId[0]]

priority = sorted(ids, key=getValue, reverse=True)

output = open('output/public6.csv', 'w')
output.write('RANK_ID')

for p in priority:
    output.write('\n' + str(int(p[1])))

output.close()


