import sys
import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import csv

global df_train, df_test

def cost_function(features, df_train, df_test):

    x_train = df_train.iloc[:,features]
    y_train = df_train['label']
    x_test = df_test.iloc[:,features]
    y_test = df_test['label']

    poly = svm.SVC(kernel='poly', degree=3, C=1)

    poly.fit(x_train, y_train)
    y_pred = poly.predict(x_test)

    poly_f1 = f1_score(y_test, y_pred, average='weighted')

    return poly_f1

def set_pso(n_dimensions, n_particles):
    
    # numero total de features disponiveis
    max_feature_idx = len(df_train.columns) -1
    feature_idxs = list(range(0, max_feature_idx, 1))
  
    # initial particles position
    # since we can't use the same feature repeated,
    # the initial position of every particle is a n_dimensions array with a random and unique combination of features
    initial_pos = []

    i = 0
    while i < n_particles:
        list_sample = random.sample(feature_idxs, n_dimensions)

        equal = False
        
        for particle in initial_pos:
            if set(list_sample) == set(particle):
                equal = True

        if not equal:
            initial_pos.append(list_sample)
            i += 1

    # min and max values for the features
    # 0 is the id for the first feature,
    # and max_feature_id is the id for the last feature
    bounds = [(0, max_feature_idx-1)]*n_dimensions
    
    return initial_pos, bounds

class Particle:
    def __init__(self, initial_pos, i):

        self.position_i = []          # particle position, i.e. features
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.f1_best_i = -1          # best error individual
        self.f1_i = -1               # error individual
        self.df_train = []
        self.df_test = []

        # initialize position
        self.position_i = initial_pos[i]
        # initialize velocity as values between -1 and 1
        for i in range(0, n_dimensions):
            self.velocity_i.append(random.uniform(-1,1))

    # evaluate current fitness
    def evaluate(self, cost_func):
        self.f1_i = cost_function(self.position_i, df_train, df_test)

        # check to see if the current position is an individual best
        if self.f1_i > self.f1_best_i or self.f1_best_i == -1:
            self.pos_best_i = self.position_i.copy()
            self.f1_best_i = self.f1_i

    # update new particle velocity
    def update_velocity(self, pos_best_g, w, c1, c2, n_dimensions):

        # constant inertia weight (how much to weigh the previous velocity)
        # cognitive constant (influences pbest)
        # social constant (influences gbest)

        for i in range(0, n_dimensions):

            # non-deterministic values to prevent particles
            # from getting stuck in local optima
            r1 = random.random()
            r2 = random.random()

            # update cognitive and social
            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])

            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds, n_dimensions):
        for i in range(0, n_dimensions):

            # round value to get discrete position
            position = round(self.position_i[i] + self.velocity_i[i])

            # adjust maximum position if necessary
            if position > bounds[i][1]:
                position = bounds[i][1]

            # adjust minimum position if necessary
            if position < bounds[i][0]:
                position = bounds[i][0]

            # make sure the feature isn't already in the position array
            if position not in self.position_i:
                self.position_i[i] = position

def maximize(cost_function, initial_pos, bounds, n_particles,
             n_dimensions, maxiter, w, c1, c2, verbose=False):

    f1_best_g = -1                    # best f1 score for group
    pos_best_g = []                   # best position for group

    # establish the swarm
    swarm = []
    for i in range(0, n_particles):
        swarm.append(Particle(initial_pos, i))

    # begin optimization loop
    i = 0
    while i < maxiter:
        if verbose: print(f'iter: {i}, best f1-score: {f1_best_g:10.4f}')

        # cycle through particles in swarm and evaluate fitness
        for j in range(0, n_particles):
            swarm[j].evaluate(cost_function)

            # determine if current particle is the best (globally)
            if swarm[j].f1_i > f1_best_g or f1_best_g == -1:
                pos_best_g = swarm[j].position_i
                f1_best_g = float(swarm[j].f1_i)

        # cycle through swarm and update velocities and position
        for j in range(0, n_particles):
            swarm[j].update_velocity(pos_best_g, w, c1, c2, n_dimensions)
            swarm[j].update_position(bounds, n_dimensions)

        i += 1

    f1_best_g = round(f1_best_g, 6)
    pos_best_g = list(df_train.iloc[:,pos_best_g].columns)
    # print final results
    if verbose:
        print('\nFINAL SOLUTION:')
        print(f'Features: {pos_best_g}')
        print(f'Score: {f1_best_g}\n')

    return f1_best_g, pos_best_g

def run_tests(inertia, cognitive, social, max_iter):
    
    results = []

    for w in inertia:
        for c1 in cognitive:
            for c2 in social:
                f1_best_g, pos_best_g = maximize(cost_function, initial_pos, bounds,
                                                 n_particles, n_dimensions, max_iter,
                                                 w, c1, c2, verbose=True)

                results.append([w, c1, c2, f1_best_g, pos_best_g])
                
    return results

if __name__ == '__main__':
    
    # Check if at least one argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script_name.py df_train_path df_test_path")
        sys.exit(1)

    # Load dataframes
    df_train = pd.read_csv(sys.argv[1], index_col=0)
    df_test = pd.read_csv(sys.argv[2], index_col=0)
    
    # Initialize number of particles
    n_particles = 2
    # Initialize dimensions, i.e. number of features per particle
    n_dimensions = 2   
    
    # Get initial position and bounds for each position
    initial_pos, bounds = set_pso(n_dimensions, n_particles)        

    # Set PSO parameters
    inertia = [0.1, 0.5, 1]
    cognitive = [0, 2, 4]
    social = [0, 2, 4]
    max_iter = 50
    
    # Run tests for each parameter
    results = run_tests(inertia, cognitive, social, max_iter)

    # Save results
    columns_names = ["w", "c1", "c2", "f1_best_g", "pos_best_g"]
    results_df = pd.DataFrame(results, columns=columns_names)
    results_df.index.name = 'iteration'

    results_df.to_csv('results.csv', index=True, header=True)



