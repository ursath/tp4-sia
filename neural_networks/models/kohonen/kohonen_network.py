import numpy as np
from typing import List
from bidimensional_layer import BidimensionalLayer

# check for 
seed = 43
np.random.seed(seed)

class KohonenNetwork:
    #input_size = n
    def __init__(self, dataset:List[any], entry_size:int, k:int, distance_function, initialize_random_weights:bool):
        self.output_layer = BidimensionalLayer(k, entry_size, distance_function, initialize_random_weights, dataset, seed)
        self.dataset = dataset

    # learning_rate, epochs, R
    def classify(self, R:float, epochs:int, learning_rate:float=None):
        distances = np.array([])
        #repeated_result = 0

        for epoch in range(epochs):
            entries_per_neuron:List[List[int]] = np.empty(self.k, self.k)
            for i in range(self.k):
                for j in range(self.k):
                    entries_per_neuron[i][j] = []
            for entry_index, entry in enumerate(self.dataset):
                for neuron in list(self.output_layer.neuron_matrix):
                    np.append(distances, neuron.distance_function(neuron.weights, entry))
                best_neuron_index = np.argmin(distances)
                best_row = best_neuron_index / self.k
                best_col = best_neuron_index % self.k 
                entries_per_neuron[best_row][best_col].append(entry_index)
                neighbours = self.output_layer.get_neuron_neighbours(best_neuron_index, R)
                for neuron in neighbours:
                    neuron.update_weights(entry, learning_rate)

            #TODO: convergence

        return entries_per_neuron, epoch
