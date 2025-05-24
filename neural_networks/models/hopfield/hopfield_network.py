import numpy as np

class HopfieldNetwork:
    def __init__(self, patterns):
        self.patterns = patterns
        self.len_patterns = len(patterns)
        self.weights = np.zeros((len(patterns[0]), len(patterns[0])))

    def initialize_weights(self):
        K = np.column_stack(self.patterns)
        self.weights = (1 / self.len_patterns) * np.dot(K, K.T)
        return self.weights
    
    def classify(self, input_pattern):
        state_vector = input_pattern

        while True:
            new_state_vector = np.sign(np.dot(self.weights, state_vector))

            if np.array_equal(new_state_vector, state_vector):
                return state_vector

            state_vector = new_state_vector
