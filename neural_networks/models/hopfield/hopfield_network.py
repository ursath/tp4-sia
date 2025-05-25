import numpy as np
import os

class HopfieldNetwork:
    def __init__(self, patterns):
        self.patterns = patterns
        self.len_patterns = len(patterns)
        self.weights = np.zeros((len(patterns[0]), len(patterns[0])))
        output = os.path.join(os.getcwd(), 'output')

        if not os.path.exists(output):
            os.makedirs(output)

        self.weights_path = f"{output}/hopfield_network_weights.txt"
        self.output_path = f"{output}/hopfield_network_output.txt"

    def initialize_weights(self):
        K = np.column_stack(self.patterns)
        self.weights = (1 / self.len_patterns) * np.dot(K, K.T)

        with open(self.weights_path, 'w') as f:
            for i in range(len(self.weights)):
                for j in range(len(self.weights)):
                    f.write(f"{self.weights[i][j]} ")
                f.write("\n")

        return self.weights
    
    def classify(self, input_pattern):
        state_vector = input_pattern

        with open(self.output_path, 'w') as f:
            while True:
                new_state_vector = np.sign(np.dot(self.weights, state_vector))

                if np.array_equal(new_state_vector, state_vector):
                    return state_vector

                state_vector = new_state_vector

                for i in range(0, len(state_vector), 5):
                    f.write(f"{state_vector[i:i+5]}\n")
                f.write("\n")
