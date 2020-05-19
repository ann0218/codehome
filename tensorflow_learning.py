# -*- coding: utf-8 -*-
import numpy as np

np.set_printoptions(threshold=np.inf)
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


x_train = np.arange(0,1,0.01)
y_train = np.sin(2 * np.pi * x_train)
plt.plot(x_train,y_train,linestyle='-', label='Groud truth')    
plt.ylabel('f(x)')
plt.xlabel('x')
#plt.show()

batch_size = 1 
epochs = 100


model = Sequential()
model.add(Dense(5, activation='tanh', input_shape=(1,)))
model.add(Dense(5, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])
#model.fit(x_train,y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1)


#test 
loss_arr = []
for i in range(epochs):
    history=model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=1,
          shuffle=True,
          verbose=1)
    loss_arr.append( history.history['loss'][0])
     
    x_test = np.copy(x_train)
    y_pred = model.predict(x_test)
    
    plt.subplot(211)
    plt.cla()
    plt.plot(x_train,y_train,linestyle='-', label='Groud truth')   
    plt.plot(x_test,y_pred, linestyle='--',label='Predict',color='red')
    plt.title('eopch: '+ str(i))
    plt.legend()
    
    plt.subplot(212)
    plt.cla()
    plt.plot(loss_arr, label='Loss volue')
    plt.title('Learning Curve')
    plt.legend()
    
    
    plt.show()
    plt.pause(0.01)


