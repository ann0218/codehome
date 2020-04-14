import numpy as np

np.set_printoptions(threshold=np.inf)
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance


def neighbourhood_points(X, x_centroid, dist=3):
    eligible_X = []
    for x in X:
        distance_between = distance.euclidean(x, x_centroid)
        if distance_between <= dist:
            eligible_X.append(x)

    eligible_X = np.array(eligible_X)
    mean = np.mean(eligible_X, axis=0)

    return eligible_X, mean


data_num = 100
data_dim = 2
data = 0 + 2 * np.random.randn(data_num, data_dim)
temp = 10 + 3 * np.random.randn(data_num, data_dim)
data = np.concatenate((data, temp), axis=0)
temp = 0 + 2 * np.random.randn(data_num, data_dim)
temp[:, 0] = temp[:, 0] + 20
data = np.concatenate((data, temp), axis=0)
temp = 0 + 1.5 * np.random.randn(data_num, data_dim)
temp[:, 0] = temp[:, 0] + 30
temp[:, 1] = temp[:, 1] + 20
data = np.concatenate((data, temp), axis=0)

data_num = data_num * 4

x = np.copy(data)

for iteration in range(15):

    mean = np.zeros((data_num, data_dim))
    for i in range(data_num):
        eligible_X, mean[i] = neighbourhood_points(data, x[i], dist=5)
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], color='blue', s=50, alpha=0.5)
    plt.scatter(x[:, 0], x[:, 1], color='red', s=50, alpha=0.3)

    x = mean

    plt.title('iteration' + str(iteration))
    plt.grid()
    plt.show()
    plt.pause(0.2)

threshold = 1.0
center = x[0,:].reshape((1.2))
for i in range(data_num):
    found = False
    for j in range(center.shape[0]):
        dst = distance.euclidean(x[i], center[j])

        if dst < threshold:
            found = True
            break
    if not found:
        center = np.concatenate((center, x[i].reshape((1,2))), axis = 0)
print(center)

k = center.shape[0]
c_color = ['red','green','blue','purple','yellow','gray']
plt.clf()

for i in range(data_num):
    dst_array = []
    for k in range(k):
        dst = distance.euclidean(center[k,:], data[i,:])
        dst_array.append(dst)
    cluster = np.argmin(dst_array)
    plt.scatter(data[i,0],data[i,1], color = c_color[cluster],s =50 , alpha = 0.1 )
for k in range(k):
    plt.scatter(center[k,0],center[k,1], color =c_color[k], s=150 , alpha = 1.0 ,marker='*')
plt.grid()
plt.show()
