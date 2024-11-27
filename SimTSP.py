# Imports

##
import networkx

##
import tsplib95

## Elkhai
import elkai

## Numpy
import numpy as np

##
from math import floor, sqrt, ceil

##
from multiprocessing import Pool, shared_memory

##
import os
import random


# Helpers

"""
Given a TSP instace file name `instance_filename`,
    this function will return its cost matrix.
"""
def get_cost_matrix(instance_filename):

    # read TSP instance
    instance = tsplib95.load(instance_filename)

    # get cost matrix
    costs = networkx.to_numpy_array(instance.get_graph())

    #
    return costs

"""
This function, creates a new metric matrix cost, where the
    distances were shuffled according to a normal distribution.
"""
def create_scenario(shared_array_name, n, seed, std_rate = 0.1):

    # Reconstruct original cost matrix from a shared memory
    shared_array = shared_memory.SharedMemory(name=shared_array_name)
    orig_costs   = np.ndarray((n, n), dtype=np.float64, buffer=shared_array.buf)

    # init new matrix cost
    costs = np.zeros(shape=(n, n), dtype=np.float64)

    # init RNG
    rng = np.random.default_rng(seed)

    # shuffle distances
    for i in range(n):        
        for j in range(i + 1, n):

            # mean and standard deviation
            mu    = orig_costs[i, j]
            sigma = orig_costs[i, j] * std_rate

            # guarantees that the new cost will be an integer >= 1
            val = int(max(1, floor(rng.normal(mu, sigma) + 0.5)))
            costs[i, j] = costs[j, i] = val

    # floyd and warshall: for ensuring metricity
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i == j: continue
                costs[i, j] = min(costs[i, k] + costs[k, j], costs[i, j])

    #
    return costs

"""
This is a worker function, which is used later on as callback when
    spawning new parallel precessors.
"""
def worker(sim_id, std_rate, shared_array_name, n):

    #
    seed = 12345 + sim_id

    # attain the new matrix cost
    costs = create_scenario(shared_array_name, n, seed, std_rate)

    #
    tour = elkai.DistanceMatrix(costs).solve_tsp()

    # get tour cost
    tour_cost = elkai.utils.path_distance(tour, costs)

    #
    return tour_cost
    
"""
Function for spawning several simulations
"""
def simulate(orig_costs, std_rate = 0.1, n_simulations=100000, n_pools = os.cpu_count()):

    # create the shared memory array for the matrix, for avoiding making repetitive copies
    n = orig_costs.shape[0]

    shared_array        = shared_memory.SharedMemory(create=True, size=orig_costs.nbytes)
    shared_array_buf    = np.ndarray(orig_costs.shape, dtype=orig_costs.dtype, buffer=shared_array.buf)
    shared_array_buf[:] = orig_costs[:]
    shared_array_name   = shared_array.name

    # spawn processes
    with Pool(n_pools) as pool:

        # 
        tour_costs = pool.starmap(worker, [( \
                i, \
                std_rate,
                shared_array_name, \
                n \
                ) for i in range(n_simulations)])

    # close shared memory
    shared_array.close()
    shared_array.unlink()

    # create histogram
    histogram = {}
    for cost in tour_costs:
        if cost not in histogram:
            histogram[cost] = 0
        histogram[cost] += 1

    #
    return histogram


orig_costs = get_cost_matrix('data/ch150.tsp')
histogram  = simulate(orig_costs, 0.1, 1000)
observations = sorted(histogram.items())
n_obs = 0
max_obs = [-1, 0]
for k, v in observations:
    print(k, v)
    n_obs += v
    if v > max_obs[1]:
        max_obs = [k, v]

print("# observations", n_obs)
print(max_obs)
