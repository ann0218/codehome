import numpy as np
np.set_printoptions(threshold=np.inf)
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance

gene_num = 2
chromosome_num = 10
iteration_num = 50
mutation_rate = 0.3
crossover_rate = 0.3
select_ratio = 0.3
goal = np.random.rand(gene_num)
population = np.random.rand(chromosome_num, gene_num)
best_fitness = 0
best_chromosome = np.array([0,0])
select_num = int(chromosome_num * select_ratio)
copy_num = int(chromosome_num - select_num)

fitness_array = []
for i in range(chromosome_num):
    fitness = 1.0 / (1.0 + distance.euclidean(population[1,:], goal))
    fitness_array.append((fitness))

for iteration in range(iteration_num):

#  selection
    temp_population = np.copy(population)
    selected_idx = np.argsort(fitness_array)[-select_num:]
    temp_population = np.copy(population[selected_idx])

#  copy
    for i in range(copy_num):
        sel_chromosome = np.random.randint(0, select_num)
        copy_chromosome = np.copy(temp_population[sel_chromosome,:].reshape((1,gene_num)))
        temp_population = np.concatenate((temp_population, copy_chromosome), axis=0)
    population = np.copy(temp_population)
#  crossover
    for i in range(chromosome_num):
        if np.random.rand(1) < crossover_rate:
            sel_chromosome = np.random.randint(0, chromosome_num)
            sel_gene = np.random.randint(0, gene_num)

#           swap
            temp = np.copy(population[i, sel_gene])
            population[i, sel_gene] = np.copy(population[sel_chromosome, sel_gene])
            population[sel_chromosome, sel_gene] = np.copy(temp)
#  mutation
    for i in range(chromosome_num):
        if np.random.rand(1) < mutation_rate:
            sel_gene = np.random.randint(0, gene_num)
            population[i, sel_gene] = np.random.rand(1)

    fitness_array =[]
    for i in range(chromosome_num):
        fitness = 1.0 / (1.0 + distance.euclidean(population[i,:],goal))
        fitness_array.append(fitness)
    if np.max(fitness_array) > best_fitness :
        best_fitness = np.max(fitness_array)
        best_idx = np.argmax(fitness_array)
        best_chromosome = np.copy(population[best_idx])
    error = distance.euclidean(best_chromosome, goal)
    plt.clf()
    plt.scatter(population[:,0], population[:,1], color = 'blue', s= 50 , alpha=0.3, marker='o')
    plt.scatter(best_chromosome[0],best_chromosome[1], color = 'green', s= 250, alpha=0.7, marker='+')
    plt.scatter(goal[0], goal[1], color = 'red' , s = 250 , alpha=1.0, marker='*')

    plt.title('iteration: ' + str(iteration) + ', error: ' + str(error))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.show()
    plt.pause(0.2)
