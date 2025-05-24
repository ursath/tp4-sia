from typing import List
import numpy as np

class KohonenNeuron:
    def __init__(self, weights_len:int, distance_function, initialize_random_weights:bool, dataset:List[any] = []):
        if initialize_random_weights:
            weights = np.random.rand(weights_len)
        else:
            weights = np.random.zeroes(weights_len)
            for column in range(weights_len):
              row = np.random.randint(len(dataset))
               # assuming id column is the first one
              weights[column] = dataset[row][column + 1]

        self.weights = weights    
        self.distance_function = distance_function

    def update_weights(self,entry:List[float], learning_rate:float):
        delta_w = learning_rate * (entry - self.weights)
        self.weights += delta_w
        return delta_w