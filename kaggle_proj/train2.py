import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import normalize

batch_size = 20000
epochs = 100
hidden_size = 128
layers = 20
myAct = 'selu'

data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
data_x = data[:, 6:12]
#data_x = data[:, 6:18]
#for i in range(len(data)):
#    for j in range(6):
#        data_x[i][j + 6] = data[i][j + 12] - data[i][j + 18]
data_y = data[:, 24]
# np.random.shuffle(data_x)
data_y = data_y.astype(int)

model = Sequential()
model.add(Dense(hidden_size, activation=myAct, kernel_initializer='normal', input_shape=(data_x.shape[1],)))

for i in range(layers - 1):
    model.add(Dense(hidden_size, activation=myAct, kernel_initializer='normal', input_shape=(hidden_size,)))
# model.add(Dense(32, activation='relu', kernel_initializer='normal'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data_x, data_y, batch_size=batch_size, epochs=epochs, class_weight={0: sum(data_y), 1: len(data) - sum(data_y)}, verbose=2)

model.save('model/model2.hdf5')
