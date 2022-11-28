import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# make input 3x3 output 1 model
model = models.Sequential()
model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=(3, 4, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
datasetFile = open("3connect.txt", "r")

# read dataset
X = []
Y = []
datastring = datasetFile.readlines()
for i in range(0, len(datastring), 5):
    Ystring = datastring[i].replace("\n", "").replace(" ", "")
    Y.append(float(Ystring))
    tmp2 = []
    for j in (datastring[i+1:i+4]):
        tmp = []
        for k in j:
            if(k == "" or k == "\n"):
                continue
            tmp.append(float(k))
        tmp2.append(tmp)
    X.append(tmp2)
X = np.array(X)
Y = np.array(Y)

# train model
model.fit(X, Y, epochs=100)

# test model
