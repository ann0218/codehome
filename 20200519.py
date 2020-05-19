# -*- coding: utf-8 -*-
import numpy as np 
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt 
from sklearn import datasets
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

data_num = 100
data_dim = 2
data = 0.5 + 0.25*np.random.randn(data_num, data_dim)
data[:,1] = data[:,1] - 1.0
temp = -0.5 + 0.2*np.random.randn(data_num, data_dim)
data = np.concatenate((data, temp), axis=0)
temp = 0.35 + 0.25*np.random.randn(data_num, data_dim)
data = np.concatenate((data, temp), axis=0)


label = np.zeros(data_num)
temp = np.ones(data_num)
label = np.concatenate((label, temp),axis=0)
temp =2*np.ones(data_num)
label = np.concatenate((label, temp),axis=0)
print(label)


num_classes = 3
label_one_hot = tf.keras.utils.to_categorical(label, num_classes)
print(label_one_hot)


data_num = data_num*3
for i in range(data_num):
    
    if (label[i]==0):
        plt.scatter(data[i,0],data[i,1], color='blue',s=50,alpha=0.1)
    elif (label[i]==1):
        plt.scatter(data[i,0],data[i,1], color='red',s=50,alpha=0.1)
    elif (label[i]==2):
        plt.scatter(data[i,0],data[i,1], color='green',s=50,alpha=0.1)

plt.grid()
plt.show()



