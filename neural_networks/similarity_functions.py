import numpy as np

def euclidean_distance(weights, x):
    basic_errors = weights - x
    squared_errors = basic_errors ** 2
    return np.sqrt(np.sum(squared_errors))

def exponential_similarity(weights, x, alpha:float = 1.0):
    basic_errors = weights - x
    squared_errors = basic_errors ** 2
    return np.exp(-alpha * squared_errors)
