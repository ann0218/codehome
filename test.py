import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

fig = plt.figure(figsize=(6,8))
ax1 = fig.add_subplot(211,projection='3d')
ax2 = fig.add_subplot(212)
ax2.set_title("learing curve")
ax2.set_xlim(0,30)
ax2.set_ylim(0,1)

x_min = -5
x_max = 5
y_min = -5
y_max = 5
z_min = 0
z_max = 15

X = np.arange(x_min, x_max, 0.25)
Y = np.arange(y_min, y_max, 0.25)
X1, Y1 = np.meshgrid(X, Y,sparse=True)
def ackley_funtion(X,Y):
    Z =15 - (-20 * np.exp(-0.2 * (np.sqrt(0.5 * (X ** 2 + Y** 2)))) -
              np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20)
    return Z
z1 = ackley_funtion(X1,Y1)

def plot_ackley():
    ax1.set_xlim(x_min,x_max)
    ax1.set_ylim(y_min,y_max)
    ax1.set_zlim(z_min,z_max)
    ax1.plot_surface(X1, Y1, z1, rstride=1, cstride=1, cmap='terrain',alpha=0.2)


# Highest point on Reversed Ackley surface
max_idx = np.unravel_index(np.argmax(z1,axis=None),z1.shape)
ackley_max_z = z1[max_idx]
test_goal = np.array((X[max_idx[0]],Y[max_idx[1]]))

plot_ackley()
ax1.scatter(*test_goal,ackley_max_z,s=200,marker='*',alpha=0.2)

iteration = 0
iteration_num = 30

# GA
ch_num = 100
gene_num = 2
population_shape = (ch_num, gene_num)
population = 8*np.random.rand(ch_num,gene_num)-4
mutation_rate =  0.3
crossover_rate = 0.7
select_rate = 0.3
select_num =int ( ch_num * select_rate)
copy_num =int( ch_num -select_num)

best_goal = None
best_fitness = 0.0

old_fitness = 0.0
early_stop_fitness = 0.98
error = 0.0
# print(population)
# population = np.zeros((population_shape[0], population_shape[1] + 1))
# population[:,: -1] = rand_population
# population[:, 2] = ackley_funtion(rand_population[:,0],rand_population[:,1])

fitness_array = []
for i in range(ch_num):
     fitness = 1.0 / (1.0 + distance.euclidean(population[i, :], test_goal))
     fitness_array.append(fitness)
for iteration in range(iteration_num):

    #  selection
    temp_population = np.copy(population)
    selected_idx = np.argsort(fitness_array)[-select_num:]
    temp_population = np.copy(population[selected_idx])


    #  copy
    for i in range(copy_num):
        sel_chromosome = np.random.randint(0, select_num)
        copy_chromosome = np.copy(temp_population[sel_chromosome, :].reshape((1, gene_num)))
        temp_population = np.concatenate((temp_population, copy_chromosome), axis=0)

    population = np.copy(temp_population)

    #  crossover
    for i in range(ch_num):
        if np.random.rand(1) < crossover_rate:
            sel_crossover = np.random.randint(0, ch_num)
            sel_gene = np.random.randint(0, gene_num)

            temp = np.copy(population[i, sel_gene])
            population[i, sel_gene] = np.copy(population[sel_crossover, sel_gene])
            population[sel_crossover, sel_gene] = np.copy(temp)

    #  mutation
    for i in range(ch_num):
        if np.random.rand(1) < mutation_rate:
            sel_gene = np.random.randint(0, gene_num)
            population[i, sel_gene] = 8*np.random.rand(1) -4

    fitness_array = []
    for i in range(ch_num):
        fitness = 1.0 / (1.0 + distance.euclidean(population[i, :], test_goal))
        fitness_array.append(fitness)

    old_fitness = error

    if np.max(fitness_array) > best_fitness:
        best_fitness = np.max(fitness_array)
        best_idx = np.argmax(fitness_array)
        # print(best_idx)
        best_chromosome = np.copy(population[best_idx])

    error = distance.euclidean(best_chromosome, test_goal)

    ax1.cla()
    #plt.clf()
    plot_ackley()
    ax1.scatter(*test_goal, ackley_max_z, color='red',s=200, marker='*', alpha=0.2)
    ax1.scatter(population[:, 0], population[:, 1],ackley_funtion(population[:, 0],population[:, 1]), color='blue', s=50, alpha=0.3, marker='o')
    ax1.scatter(best_chromosome[0], best_chromosome[1],ackley_funtion(best_chromosome[0],best_chromosome[1]), color='green', s=250, alpha=0.7, marker='+')
    if iteration > 0:
        ax2.plot((iteration - 1, iteration), (1-old_fitness,1-error), color='C0')
    print(iteration)
    print(old_fitness)
    print(best_chromosome)
    plt.draw()
    plt.pause(0.5)
plt.show()


