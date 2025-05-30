import numpy as np
import os

class HopfieldNetwork:
    def __init__(self, patterns):
        self.patterns = patterns
        self.len_patterns = len(patterns)
        self.weights_matrix = np.zeros((len(patterns[0]), len(patterns[0])))
        output = os.path.join(os.getcwd(), 'output')

        if not os.path.exists(output):
            os.makedirs(output)

        self.weights_path = f"{output}/hopfield_network_weights.txt"
        self.output_path = f"{output}/hopfield_network_output.txt"
        self.energy_path = f"{output}/hopfield_network_energy_evolution.txt"

    def initialize_weights(self):
        # Apilar los patrones como columnas de una matriz
        K = np.column_stack(self.patterns)
        # Calcular la matriz de pesos con la regla de Hebb: W = (1/N) * K * K.T
        self.weights_matrix = (1 / self.len_patterns) * np.dot(K, K.T)
        # Eliminar las auto-conexiones
        np.fill_diagonal(self.weights_matrix, 0)

        with open(self.weights_path, 'w') as f:
            for i in range(len(self.weights_matrix)):
                for j in range(len(self.weights_matrix)):
                    f.write(f"{self.weights_matrix[i][j]} ")
                f.write("\n")
        return self.weights_matrix
    
    def classify(self, input_pattern):
        state_vector = input_pattern  # Estado inicial es el patrón de entrada
        previous_energy = 0
        repeated = 0
        iteration = 1
        with open(self.output_path, 'w') as states_file:
            with open(self.energy_path, 'w') as energy_file:
                while True:
                    # Se actualiza el estado aplicando: sign(W · state)
                    new_state_vector = np.sign(np.dot(self.weights_matrix, state_vector))

                    # Si el producto es cero, se mantiene el valor anterior
                    zero_mask = (np.dot(self.weights_matrix, state_vector) == 0)
                    new_state_vector[zero_mask] = state_vector[zero_mask]

                    # Se calcula la energía del nuevo estado
                    current_energy = self.calculate_energy_function(self.weights_matrix, new_state_vector)
                    energy_file.write(f"{iteration} {current_energy}\n")

                    # Condición de parada: Si la energía se repitió dos veces seguidas
                    if current_energy == previous_energy:
                    #if np.array_equal(new_state_vector, state_vector):
                        repeated += 1

                    if repeated >= 2:
                        return iteration, new_state_vector

                    iteration+=1
                    state_vector = new_state_vector
                    previous_energy = current_energy

                    for i in range(0, len(state_vector), 5):
                        states_file.write(f"{state_vector[i:i+5]}\n")
                    states_file.write("\n")

    def calculate_energy_function(self, weights_matrix, state_vector):
        # Función de energía de Hopfield: E = -∑(i<j) w_ij * s_i * s_j
        energy = 0
        for i in range(weights_matrix.shape[0]):
            for j in range(i+1, weights_matrix.shape[0]):
               energy -= weights_matrix[i][j] * state_vector[i] * state_vector[j]
        return energy
