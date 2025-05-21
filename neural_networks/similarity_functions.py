import numpy as np

def euclidean_distance(weights, x):
    basic_errors = weights - x
    squared_errors = basic_errors ** 2
    return np.sqrt(np.sum(squared_errors))
