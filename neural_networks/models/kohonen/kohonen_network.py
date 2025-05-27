import numpy as np
from typing import List
from neural_networks.models.kohonen.bidimensional_layer import BidimensionalLayer

class KohonenNetwork:

    # Kohonen Network params:
    # - dataset: la matriz de entrada (ej. países en columnas normalizadas).
    # - entry_size: número de features (dimensión de cada vector de entrada).
    # - k: tamaño del mapa 2D (mapa de k x k neuronas).
    # - distance_function: métrica de distancia (Euclidean, etc.).
    # - initialize_random_weights: si se inicializan pesos al azar.

    def __init__(self, dataset:List[any], entry_size:int, k:int, distance_function, initialize_random_weights:bool):
        self.output_layer = BidimensionalLayer(k, entry_size, distance_function, initialize_random_weights, dataset)
        self.dataset = dataset

    # Método principal de entrenamiento:
    # learning_rate, epochs, R
    def classify(self, R:float, epochs:int, learning_rate:float=None, r_variation:bool=False, learning_rate_variation:bool=False):
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

                if learning_rate_variation:
                    learning_rate = 1 / (epoch + 1)

                if r_variation:
                    R = R * (1 - (epoch / epochs))

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
    
