from xgboost import XGBClassifier

import numpy as np
from keras.utils import normalize
from sklearn.metrics import accuracy_score

import sys

data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
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

model._Booster.save_model('myModel')
