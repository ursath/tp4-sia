import os
from neural_networks.models.hopfield.hopfield_network import HopfieldNetwork
from utils import apply_noise, save_input_pattern
import numpy as np
import json
np.random.seed(43)
from visualization import plot_energy_vs_iteration

if __name__ == "__main__":

    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    hopfield_config = config["hopfield"]
    pattern_filenames = hopfield_config["patterns"]
    input_filename = hopfield_config["pattern_for_input"]
    noise_percentage = hopfield_config["noise_percentage"]
    
    patterns_dir = "input_data/hopfield_patterns"

    patterns = []

    # Crea una lista de patrones a partir de los archivos .txt
    for filename in pattern_filenames:
        file_path = patterns_dir + "/" + filename
        with open(file_path, "r") as f:
            p = [[int(x) for x in line.split()] for line in f.read().splitlines()]
        flat_pattern = [x for row in p for x in row]
        patterns.append(flat_pattern)

    #letter_file = "input_data/input.txt"
    #letter_pattern = []
    #with open(letter_file, "r") as f:
    #    letter_pattern.append([x.split() for x in f.read().splitlines()])
    #letter_pattern = [[int(x) for x in sublist] for sublist in letter_pattern[0]]
    #flat_letter_pattern = [line for sublist in letter_pattern for line in sublist]

    # Aplica ruido al patrón de entrada (el primer patrón en la lista)
    with open("input_data/hopfield_patterns/" + input_filename, "r") as f:
        pattern = [[int(x) for x in line.split()] for line in f.read().splitlines()]
    flat_pattern = [x for row in pattern for x in row]
    flat_letter_pattern = apply_noise(flat_pattern, noise_percentage)

    # Save the noisy input pattern to a file
    save_input_pattern(flat_letter_pattern, "input_data/hopfield_input.txt")

    # Convertimos los patrones a arrays de NumPy para hacer cálculos vectoriales
    np_patterns = []    
    for pattern in patterns:
        np_patterns.append(np.array(pattern))

    # Crea una matriz vacía para guardar productos escalares entre patrones
    ortogonality_matrix = np.zeros((len(patterns), len(patterns)))

    # Recorre cada par de patrones y calcula el producto escalar guardando el resultado en la matriz de ortogonalidad
    for i, first_pattern in enumerate(np_patterns):
        for j, second_pattern in enumerate(np_patterns):
            if not np.array_equal(first_pattern, second_pattern):
                ortogonality_matrix[i][j] = np.dot(first_pattern, second_pattern)

    print("ortogonality check:")
    for i in range(len(patterns)):
        print(ortogonality_matrix[i])

    # Crea la red de Hopfield con los patrones e inicializa los pesos
    hopfield_network = HopfieldNetwork(patterns)
    hopfield_network.initialize_weights()

    # check weights 
    weights = open("output/hopfield_network_weights.txt", "r").readlines()
    weight_matrix = np.zeros((len(patterns[0]), len(patterns[0])))
    for row, line in enumerate(weights):
        weights_row = line.strip().split(" ")
        for col, weight in enumerate(weights_row):
            weight_matrix[row][col] = float(weight)

    symetric = True
    for row in range(len(patterns[0])):
        for column in range(len(patterns[0])):
            if weight_matrix[row, col] != weight_matrix[col, row]:
                symetric = False

    #print("The weight matrix is ok") if symetric else print("The weight matrix is not symetric")

    # Llama a la red para clasificar el patrón de entrada ruidoso y devuelve el resultado final
    epoch, state_vector = hopfield_network.classify(np.array(flat_letter_pattern))
    energy_evolution_lines = open("output/hopfield_network_energy_evolution.txt", "r").readlines()
    energy_values = []
    iteration_values = []

    for line in energy_evolution_lines:
        values = line.split(" ") 
        iteration_values.append(float(values[0]))
        energy_values.append(float(values[1]))

    plot_energy_vs_iteration(energy_values, iteration_values, input_filename, ''.join(os.path.splitext(f)[0][0] for f in pattern_filenames))

    print("obtained output:")
    for i in range(0, len(state_vector), 5):
        print(state_vector[i:i+5])
