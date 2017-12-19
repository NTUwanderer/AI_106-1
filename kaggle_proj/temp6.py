from xgboost import XGBClassifier, XGBRegressor

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import normalize
from sklearn.metrics import accuracy_score
from keras.models import load_model
import math

batch_size = 16000
epochs = 1000

validate_ratio = 0.25

data = np.genfromtxt('Train.csv', delimiter=',', skip_header=1)
data = data[:, 1:]
np.random.seed(2)
def getValue(x):
    return x[-1]
data = np.array(sorted(data, key=getValue))
counts = int(sum(data[:, -1]))

zeros = data[:-counts]
ones = data[-counts:]

np.random.shuffle(zeros)
np.random.shuffle(ones)

vo_split = int(counts / 4)
vz_split = 5000 - vo_split
data = np.concatenate((zeros[:-vz_split], ones[:-vo_split]))
validation = np.concatenate((zeros[-vz_split:], ones[-vo_split:]))

np.random.shuffle(data)
np.random.shuffle(validation)

#np.random.shuffle(data)
data_x = data[:, :-1]
data_y = data[:, -1]
data_y = data_y.astype(int)

weight = np.array(data_y, copy=True)

for i in range(len(weight)):
    if weight[i] == 1:
        # weight[i] = math.sqrt(20000 - counts)
        weight[i] = 1
    else:
        # weight[i] = math.sqrt(counts)
        weight[i] = 2

vx = validation[:, :-1]
vy = validation[:, -1]
vy = vy.astype(int)


# read in data
model = XGBClassifier(max_depth=3, n_estimators=100, objective='binary:logistic')
# model = XGBRegressor(objective='rank:pairwise')

def myMap500(p, y):
    rank = np.argsort(1 - p)
    y = y.get_label()
    values = [0] * 500
    for i in range(500):
        values[i] = y[rank[i]]

    total = 0.0
    run_sum = 0.0

    for i in range(500):
        run_sum += values[i]
        total += run_sum / (i + 1)

    return 'mymap500', total / 500

model.fit(data_x, data_y, sample_weight=weight, eval_set=[(vx, vy)], eval_metric=myMap500, verbose=True)
pred = model.predict_proba(vx)
pred = pred[:, 1]

def myMap(p, y, k=500):
    rank = np.argsort(1 - p)
    values = [0] * k
    for i in range(k):
        values[i] = y[rank[i]]

    total = 0.0
    run_sum = 0.0

    for i in range(k):
        run_sum += values[i]
        total += run_sum / (i + 1)

    return total / k

print ('myMap 100, 500: ', myMap(pred, vy, 50), myMap(pred, vy))

data_t = np.genfromtxt('Test_Public.csv', delimiter=',', skip_header=1)

prediction = model.predict_proba(data_t[:, 1:])
# prediction = model.predict(data_t[:, 1:])
# prediction = normalize(prediction[:, 1])

myPrediction = np.zeros((len(data_t)))

for i in range(len(data_t)):
    myPrediction[i] = prediction[i][1]

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


