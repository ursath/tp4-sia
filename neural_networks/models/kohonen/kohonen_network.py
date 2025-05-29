import numpy as np
from typing import List
from neural_networks.models.kohonen.bidimensional_layer import BidimensionalLayer
from neural_networks.similarity_functions import euclidean_distance, exponential_similarity


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
        self.distance_function = distance_function

    # Identifica la neurona ganadora
    def classify(self, entry):
        distances = []
        for (i,j), neuron in np.ndenumerate(self.output_layer.neuron_matrix):
            distances.append(neuron.distance_function(neuron.weights, entry))
        if self.distance_function == exponential_similarity:
            best_neuron_index = np.argmax(np.array(distances))
        else:
            best_neuron_index = np.argmin(np.array(distances))  
        best_row = int(best_neuron_index / self.output_layer.k)
        best_col = best_neuron_index % self.output_layer.k 
        return best_row, best_col, self.output_layer.neuron_matrix[best_row][best_col].weights

    # learning_rate, epochs, R
    def train(self, R0:float, epochs:int, learning_rate:float=None, r_variation:bool=False, learning_rate_variation:bool=False):
        repeated_result = 0
        last_entries_per_neuron =np.empty((self.output_layer.k, self.output_layer.k), dtype=object)
        R = R0

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
                # Se clasifica y se identifica la neurona ganadora
                best_row, best_col, best_weights = self.classify(entry)
                entries_per_neuron[best_row][best_col].append(entry_index)

                # Se obtienen los vecinos de la neurona ganadora
                neighbours = self.output_layer.get_neuron_neighbours(best_row, best_col, R)

                # Se actualizan los pesos
                for neuron in neighbours:
                    neuron.update_weights(entry, learning_rate)
            print(f"Epoch {epoch}, R: {R}, Learning Rate: {learning_rate:.4f}")

            # Condición de corte: si las entradas por neurona no cambian en 2 épocas consecutivas
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
    
