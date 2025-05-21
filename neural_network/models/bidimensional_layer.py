from typing import List
import numpy as np
from kohonen_neuron import KohonenNeuron

class BidimensionalLayer:
    def __init__(self, k:int, weights_len:int, distance_function, initialize_random_weights:bool, dataset:List[any] = []):
        self.k = k
        self.neuron_matrix = []
        for i in range(k):
            self.neuron_matrix[i] = []
            for j in range(k):
                self.neuron_matrix[i][j] = KohonenNeuron(weights_len, distance_function, initialize_random_weights, dataset)

    def get_neuron_neighbours(self, best_neuron_index:int, R:float):
        best_row = best_neuron_index / self.k
        best_col = best_neuron_index % self.k
        rounded_R = int(np.ceil(R))
        neighbours = []

        for delta_row in range(-rounded_R, rounded_R+1):
            for delta_col in range(-rounded_R, rounded_R+1):
                if (delta_row == 0 and delta_col == 0):
                    continue
                new_row = best_row + delta_row
                new_col = best_col + delta_col

                distance = np.sqrt(delta_col ** 2 + delta_row ** 2)
                if distance <= R:
                    neighbours.append(self.neuron_matrix[new_row][new_col])
        
        return neighbours
