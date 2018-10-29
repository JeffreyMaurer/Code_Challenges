import itertools as it
import numpy as np
import random

def random_walk(d, n):
    """Return the coordinates of a random walk after 'n' steps in 'd' dimensions"""
    # setup position
    if len(random_walk.position) != d:
        random_walk.position = np.array([0 for i in range(d)])
    start = np.array(random_walk.position)
    
    if len(random_walk.velocity) != 2*d:
        velocity = []
        velocity = [tuple(perm) for perm in it.permutations([1 if i == 0 else 0 for i in range(d)])]
        velocity += [tuple(perm) for perm in it.permutations([-1 if i == 0 else 0 for i in range(d)])]
        random_walk.velocity = np.array(list(set(velocity)))
        random_walk.r = np.array([i for i in range(d*2)])
        
    for i in range(n):
        direction = np.random.choice(random_walk.r)
        random_walk.position += random_walk.velocity[direction]
        if np.array_equal(random_walk.position, start):
            return True
    return False
random_walk.position = []
random_walk.velocity = []


for dim in range(1, 4):
    for walk_length in [10,100,1000,5000]:
        counter = 0
        for replicate in range(1000):
            if random_walk(dim, walk_length):
                counter += 1
        print("A walk of {0} steps in {1} dimension{3} returns to its starting position {2}% of the time".format(walk_length, dim, float(counter) / 10, 's' if dim > 1 else ''))
            
