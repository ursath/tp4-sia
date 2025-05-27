from typing import List
import numpy as np
from neural_networks.models.kohonen.kohonen_neuron import KohonenNeuron

# Define una capa de neuronas organizada en una matriz 2D de tamaño k x k
class BidimensionalLayer:
    def __init__(self, k:int, weights_len:int, distance_function, initialize_random_weights:bool, dataset:List[any] = []):
        self.k = k
        self.neuron_matrix = []
        # Creo una matriz k x k de neuronas
        for i in range(k):
            self.neuron_matrix.append([])
            for j in range(k):
                self.neuron_matrix[i].append(KohonenNeuron(weights_len, distance_function, initialize_random_weights, dataset))

    # Dado el índice de una neurona ganadora (best_row, best_col) y un radio de vecindad R
    # Devuelve las neuronas vecinas dentro del radio R (sin incluir a la neurona ganadora)
    def get_neuron_neighbours(self, best_row:int, best_col:int, R:float):
        rounded_R = int(np.ceil(R))
        neighbours = []

        for delta_row in range(-rounded_R, rounded_R+1):
            for delta_col in range(-rounded_R, rounded_R+1):
                if (delta_row == 0 and delta_col == 0):
                    continue
                new_row = best_row + delta_row
                new_col = best_col + delta_col
                if (new_row < 0 or new_col < 0 or new_row > self.k-1 or new_col > self.k - 1):
                    continue

                distance = np.sqrt(delta_col ** 2 + delta_row ** 2)
                if distance <= R:
                    neighbours.append(self.neuron_matrix[new_row][new_col])
        
        return neighbours
