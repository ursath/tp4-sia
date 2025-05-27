import numpy as np
from typing import List
from neural_networks.models.kohonen.bidimensional_layer import BidimensionalLayer

class KohonenNetwork:
    #input_size = n
    def __init__(self, dataset:List[any], entry_size:int, k:int, distance_function, initialize_random_weights:bool):
        self.output_layer = BidimensionalLayer(k, entry_size, distance_function, initialize_random_weights, dataset)
        self.dataset = dataset

    def classify(self, entry):
        distances = []
        for (i,j), neuron in np.ndenumerate(self.output_layer.neuron_matrix):
            distances.append(neuron.distance_function(neuron.weights, entry))
        best_neuron_index = np.argmin(np.array(distances))
        best_row = int(best_neuron_index / self.output_layer.k)
        best_col = best_neuron_index % self.output_layer.k 
        return best_row, best_col, self.output_layer.neuron_matrix[best_row][best_col].weights

    # learning_rate, epochs, R
    def train(self, R0:float, epochs:int, learning_rate:float=None, r_variation:bool=False, learning_rate_variation:bool=False):
        repeated_result = 0
        last_entries_per_neuron =np.empty((self.output_layer.k, self.output_layer.k), dtype=object)

        for epoch in range(epochs):
            entries_per_neuron = np.empty((self.output_layer.k, self.output_layer.k), dtype=object)
            for i in range(self.output_layer.k):
                for j in range(self.output_layer.k):
                    entries_per_neuron[i][j] = []

            if learning_rate_variation:
                learning_rate = 1 / (epoch + 1)

            if r_variation:
                #R = R * (1 - (epoch / epochs))
                R = self.variate_radius(R0, epoch, epochs)

            for entry_index, entry in enumerate(self.dataset):
                best_row, best_col, best_weights = self.classify(entry)
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
    
    def variate_radius(self,R0:float, epoch:int, total_epochs:int):
        tau = total_epochs / (np.log(R0) * 2)
        return 1 + (R0 - 1) * np.exp(-epoch / tau)
    
