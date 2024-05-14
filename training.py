
import numpy as np
import pandas as pd
import os
from glob import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, Adagrad, Adadelta, RMSprop
from keras import regularizers
from dotenv import load_dotenv 
from array_manipulation import transform_data

load_dotenv()
X_train = np.load(file='D:\TDBRAIN_dataset\TDBRAIN_data\TDBRAIN_derivatives_csv\X_train.npy',mmap_mode='r')

X_val = np.load(file='D:\TDBRAIN_dataset\TDBRAIN_data\TDBRAIN_derivatives_csv\X_val.npy',mmap_mode='r')

X_test = np.load(file='D:\TDBRAIN_dataset\TDBRAIN_data\TDBRAIN_derivatives_csv\X_test.npy',mmap_mode='r')

y_train = np.load('D:\TDBRAIN_dataset\TDBRAIN_data\TDBRAIN_derivatives_csv\y_train.npy')

y_val = np.load('D:\TDBRAIN_dataset\TDBRAIN_data\TDBRAIN_derivatives_csv\y_val.npy')

y_test = np.load('D:\TDBRAIN_dataset\TDBRAIN_data\TDBRAIN_derivatives_csv\y_test.npy')

indices_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 20]



X_train = transform_data(X_train,indices_to_keep)
X_val = transform_data(X_val,indices_to_keep)
X_test = transform_data(X_test,indices_to_keep)

print ("*"*100)
print(f'Training set size: {len(y_train)}, validation set size: {len(y_val)}, testing set size: {len(y_test)}')
print ("*"*100)


model = Sequential()

model.add(Conv1D(filters=4, kernel_size=12, strides=1, activation="relu", input_shape=(5000, 12), kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=8, kernel_size=12, strides=1, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.2))


model.add(Conv1D(filters=16, kernel_size=12, strides=1, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=12, strides=1, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Flatten())


model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(np.array(X_train), np.array(y_train), batch_size=1, epochs=5, validation_data = (np.array(X_val), np.array(y_val)))

predictions = model.predict(np.array(X_test))


predicted_classes = [1 if prediction >= 0.5 else 0 for prediction in predictions]

sum = 0 

for i in range (len(predicted_classes)):
    if predicted_classes[i] == y_test[i]:
        sum+=1

print(f'Testing accuracy: {sum/len(predicted_classes)} ')

print('arrs loaded and mapped, freeing memory ') 
del X_train, X_val, X_test