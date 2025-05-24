import numpy as np
from typing import List
from neural_networks.models.kohonen.bidimensional_layer import BidimensionalLayer

# check for 
seed = 43
np.random.seed(seed)

class KohonenNetwork:
    #input_size = n
    def __init__(self, dataset:List[any], entry_size:int, k:int, distance_function, initialize_random_weights:bool):
        self.output_layer = BidimensionalLayer(k, entry_size, distance_function, initialize_random_weights, dataset)
        self.dataset = dataset

    # learning_rate, epochs, R
    def classify(self, R:float, epochs:int, learning_rate:float=None):
        repeated_result = 0
        last_entries_per_neuron =np.empty((self.output_layer.k, self.output_layer.k), dtype=object)

        for epoch in range(epochs):
            entries_per_neuron = np.empty((self.output_layer.k, self.output_layer.k), dtype=object)
            for i in range(self.output_layer.k):
                for j in range(self.output_layer.k):
                    entries_per_neuron[i][j] = []
            for entry_index, entry in enumerate(self.dataset):
                distances = []
                for (i,j), neuron in np.ndenumerate(self.output_layer.neuron_matrix):
                    distances.append(neuron.distance_function(neuron.weights, entry))
                best_neuron_index = np.argmin(np.array(distances))
                best_row = int(best_neuron_index / self.output_layer.k)
                best_col = best_neuron_index % self.output_layer.k 
                entries_per_neuron[best_row][best_col].append(entry_index)
                neighbours = self.output_layer.get_neuron_neighbours(best_row, best_col, R)
                for neuron in neighbours:
                    neuron.update_weights(entry, learning_rate)

            if np.array_equal(last_entries_per_neuron, entries_per_neuron):
                repeated_result += 1
            else:
                repeated_result = 0
            if repeated_result == 2:
                return entries_per_neuron, epoch
            last_entries_per_neuron = entries_per_neuron

        return entries_per_neuron, epoch
