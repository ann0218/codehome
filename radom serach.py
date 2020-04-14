import numpy as np
np.set_printoptions(threshold=np.inf)
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance


particle_dim = 2
particle_num = 10
goal = np.random.rand(particle_dim)


iteration_num = 50
min_dis = 1000
sigma = 0.05
dacay = 0.9
best_particle = np.array([0,0])
particles = np.random.rand(particle_num , particle_dim)

for iteration in range(iteration_num):
    dst_array = []

    for i in range(particle_num):
        dst = distance.euclidean(particles[i,:], goal)
        dst_array.append(dst)
    if np.min(dst_array) < min_dis:
        min_dis = np.min(dst_array)
        min_idx = np.argmin(dst_array)
        best_particle = particles[min_idx]     
    
    error = distance.euclidean(best_particle, goal)
    
    sigma *= dacay
    particles = sigma * np.random.randn(particle_num , particle_dim)
    particles[:,0] += best_particle[0]
    particles[:,1] += best_particle[1]
    
    plt.clf()
    plt.scatter(particles[:,0], particles[:,1], color = 'blue' , s= 50 ,alpha = 1.0)
    plt.scatter(best_particle[0], best_particle[1], color = 'green' , s= 250 ,alpha = 1.0, marker= '+')
    plt.scatter(goal[0], goal[1], color = 'red' , s= 250 ,alpha = 1.0, marker = '*')
    plt.title('iteration' + str(iteration) + 'error' + str(error))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.show()
    plt.pause(0.2)
