import os
from neural_networks.models.hopfield.hopfield_network import HopfieldNetwork
import numpy as np

np.random.seed(43)

if __name__ == "__main__":
    
    patterns_dir = "input_data/hopfield_patterns"

    patterns = []
    for i in os.listdir(patterns_dir):
        p = []
        if i.endswith(".txt"):
            with open(os.path.join(patterns_dir, i), "r") as f:
                p.append([x.split() for x in f.read().splitlines()])
            p = [[int(x) for x in sublist] for sublist in p[0]]
            flat_pattern = [line for sublist in p for line in sublist]
            patterns.append(flat_pattern)

    letter_file = "input_data/input.txt"
    letter_pattern = []
    with open(letter_file, "r") as f:
        letter_pattern.append([x.split() for x in f.read().splitlines()])
    letter_pattern = [[int(x) for x in sublist] for sublist in letter_pattern[0]]
    flat_letter_pattern = [line for sublist in letter_pattern for line in sublist]


    hopfield_network = HopfieldNetwork(patterns)
    hopfield_network.initialize_weights()

    state_vector = hopfield_network.classify(flat_letter_pattern)

    for i in range(0, len(state_vector), 5):
        print(state_vector[i:i+5])
