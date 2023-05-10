import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# make input 3x3 output 1 model
model = models.Sequential()
model.add(layers.Conv2D(64, (2, 2),strides=(1, 1), activation='relu', input_shape=(13,6, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='relu'))
model.compile(optimizer='adam',
              loss='MeanSquaredError',
              metrics=['accuracy'])
model.build()
model.summary()

# read dataset


def readDataset(path):
    datasetFile = open(path, "r")
    X = []
    Y = []
    datastring = datasetFile.readlines()
    for i in range(0, len(datastring), 2):
        Xstring = datastring[i]
        Ystring = datastring[i+1]
        tmp = []
        for j in range(13):
            tmp2=[]
            for k in range(6):
                tmp2.append(float(Xstring[j*6+k])/6.0);
            tmp.append(tmp2)
        X.append(tmp)
        Y.append(float(Ystring))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

X, Y = readDataset("train.txt")
X = X.reshape(len(X),13,6,1,);
# train model
model.fit(X, Y, epochs=100)

# test model
X, Y = readDataset("test.txt")
X = X.reshape(len(X),13,6,1,);
for i in range(len(X)):
    print("predict", model.predict(np.array([X[i]])), "real", Y[i])
