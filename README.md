# Particle-Swarm-Optimization

#from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fitness_weierstrass import *
from fitness_ackley import *
from fitness_rastrigin import *
from fitness_bent_cigar import *
from fitness_HCE import *
from fitness_discus import *
from fitness_rosenbrock import *
from fitness_griewank import *


# Control parameters
                 # Intertial weight. In some variations, it is set to vary with iteration number.
c1 = 2.05                  # Weight of searching based on the optima found by a particle
c2 = 2.05                  # Weight of searching based on the optima found by the swarm
v_fct = 1                 # Velocity adjust factor. Set to 1 for standard PSO.


Np = 100                   # population size (number of particles)
D = 2                    # dimension (= no. of parameters in the fitness function)
max_iter = 22             # maximum number of iterations 
xL = np.zeros(D) - 10      # lower bound (does not need to be homogeneous)  
xU = np.zeros(D) + 10      # upper bound (does not need to be homogeneous)   
v_max = xU-xL
v_min = -v_max

# Defining and intializing variables

pbest_val = np.zeros(Np)            # Personal best fintess value. One pbest value per particle.
gbest_val = np.zeros(max_iter)      # Global best fintess value. One gbest value per iteration (stored).

pbest = np.zeros((D,Np))            # pbest solution
gbest = np.zeros(D)                 # gbest solution

gbest_store = np.zeros((D,max_iter))   # storing gbest solution at each iteration

pbest_val_avg_store = np.zeros(max_iter)
fitness_avg_store = np.zeros(max_iter)

for run in range(5):
    x = np.random.rand(D,Np)            # Define a two dimensional array to store values of the initial position of the particles
    swarm_initial_points=x

    v = np.zeros((D,Np))                # Define a two dimensional array to store values of the initial velocity of the particles

    # Setting the initial position of the particles over the given bounds [xL,xU]
    for m in range(D):    
        x[m,:] = xL[m] + (xU[m]-xL[m])*x[m,:]


    #plt.subplots_adjust(subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None))
    w=np.linspace(0.9,0.4,max_iter)
    # Loop over the generations

    for i in range(max_iter):
        w_i=w[i]
        if i > 0:                             # Do not update postion for 0th iteration
            r1 = np.random.rand(D,Np)            # random numbers [0,1], matrix D x Np
            r2 = np.random.rand(D,Np)            # random numbers [0,1], matrix D x Np   
            v_global = np.multiply(((x.transpose()-gbest).transpose()),r2)*c2*(-1.0)    # velocity towards global optima
            v_local = np.multiply((pbest- x),r1)*c1           # velocity towards local optima (pbest)

            v = w_i*v + (v_local + v_global)       # velocity update
            x = x + v*v_fct                      # position update

        def clamp(v, v_min, v_max):
            if v < v_min:
                return v_min
            elif v > v_max:
                return v_max
            else:
                return v

        fit = np.zeros(Np)
        for k in range(len(fit)):
            fit[k] = Weierstrass_func(x[:,k])     # fitness function call (once per iteration). Vector Np

        if i == 0:
            pbest_val = np.copy(fit)             # initial personal best = initial fitness values. Vector of size Np
            pbest = np.copy(x)                   # initial pbest solution = initial position. Matrix of size D x Np
        else:
            # pbest and pbest_val update
            ind = np.argwhere(fit > pbest_val)   # indices where current fitness value set is greater than pbset
            pbest_val[ind] = np.copy(fit[ind])   # update pbset_val at those particle indices where fit > pbest_val
            pbest[:,ind] = np.copy(x[:,ind])     # update pbest for those particle indices where fit > pbest_val

        # gbest and gbest_val update
        ind2 = np.argmax(pbest_val)                       # index where the fitness is maximum
        gbest_val[i] = np.copy(pbest_val[ind2])        # store gbest value at each iteration
        gbest = np.copy(pbest[:,ind2])                    # global best solution, gbest
        gbest_store[:,i] = np.copy(gbest) 
        # store gbest solution
        pbest_val_avg_store[i] = np.mean(pbest_val)
        fitness_avg_store[i] = np.mean(fit)
    #swarm_init=np.array(gbest)
        print( ". D =", D, ". w =", w_i, ". Iter. =", i, ". gbest_val = ", gbest_val[i])  # print iteration no. and best solution at each iteration
        print("*************************************************************")



    # saving results in pandas DataFrame

    indexo=[]
    for i in range(D):
        label="Compoenent::dim"
        label+=str(i)
        indexo.append(label)
    columno=[]
    for i in range(Np):
        label="point::"
        label+=str(i)
        columno.append(label)
    df_gbest=pd.DataFrame(gbest, index=indexo, columns=['gbest'])
    df_x=pd.DataFrame(swarm_initial_points, index=indexo, columns=columno)
    print("===============================================\n")  
    print(df_gbest)
    print("\n^^^^=^^^^=^^^^=^^^^=^^^^=^^^^=^^^^=^^^^=^^^^")
    print("\n\n\n\n\n===============================================\n")  
    #print(df_x)

    # write dataframe to excel sheet named 'results_for_each_run'
    with pd.ExcelWriter('Masterfile.xlsx', engine='xlsxwriter') as writer: 
        i=run
        sheet_name_gbest_each_run="gbest_"+str(i)
        sheet_name_swarm_each_run="swarm_"+str(i)
        df_gbest.to_excel(writer, sheet_name=sheet_name_gbest_each_run, encoding='utf8', index=False) 
        df_x.to_excel(writer, sheet_name=sheet_name_swarm_each_run, encoding='utf8', index=False)
        writer.save
        writer.close
print('DataFrame is written successfully to Excel Sheet.')

print("#################################################\n")  

# Plotting
plt.rcParams.update({'font.size': 11})
plt.figure(figsize=(10,10))
plt.plot(gbest_val,label = 'gbest_val')
plt.plot(pbest_val_avg_store, label = 'Avg. pbest')
plt.plot(fitness_avg_store, label = 'Avg. fitness')
text="Weirestrauss\n"+"w =[0.9 .. 0.4]"+"\nD ="+str(D)
plt.legend(title=text, fontsize=10)
plt.xlabel('iterations')
plt.ylabel('fitness, gbest_val')

print("#################################################\n")
plt.figure(figsize=(6,6))
plt.plot(gbest_store[m,:],label = 'D = ' + str(m+1))

plt.legend(title="Weirestrauss")
plt.xlabel('iterations')
plt.ylabel('Best solution, gbest[:,iter]')
print("**********************")
#print("swarm_init[i,:]",swarm_init[:,1])
