import numpy as np

np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance

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

k = 4

c_color = ['red', 'green', 'blue', 'purple']
choose_idx = np.random.randint(0, data_num, size=(k,))
center = data[choose_idx]

for iteration in range(10):
    cluster_arr = []
    cluster_num = np.array([0] * k)
    mean = np.array([[0.0, 0.0]] * k)
    plt.clf()
    for i in range(data_num):
        dst_0 = distance.euclidean(center[0, :], data[i, :])
        dst_1 = distance.euclidean(center[1, :], data[i, :])
        dst_2 = distance.euclidean(center[2, :], data[i, :])
        dst_3 = distance.euclidean(center[3, :], data[i, :])

        cluster = np.argmin([dst_0, dst_1, dst_2, dst_3])
        cluster_arr.append(cluster)

        cluster_num[cluster] += 1
        mean[cluster, :] += data[i, :]

        plt.scatter(data[i, 0], data[i, 1], color=c_color[cluster], s=50, alpha=0.3)

    for i in range(k):
        mean[i, :] /= cluster_num[i]
        plt.scatter(center[i, 0], center[i, 1], color=c_color[i], s=200, alpha=1, marker='+')
        plt.scatter(mean[i, 0], mean[i, 1], color=c_color[i], s=200, alpha=1, marker='*')
        #        print (mean[i,0])
        #        print ("\n")
        #        print (center[i,0])
        ans = mean[i, 0] - center[i, 0]
    # update
    center = mean
    if (ans == 0):
        plt.title('Iteration' + str(iteration) + "   " + "ans" + str(ans))
        break;
    else:
        plt.title('Iteration' + str(iteration) + "   " + "ans" + str(ans))
        center = mean
    # print (mean[1,0]-center[1,0])

    # plt.title('Iteration' + str(iteration))
    plt.grid()
    plt.show()
    plt.pause(0.5)
