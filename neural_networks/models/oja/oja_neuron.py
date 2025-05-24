from typing import List
import numpy as np

class OjaNeuron:
    def __init__(self, features_len:int):
        self.weights = np.random.uniform(0, 1, features_len)   

    def update_weights(self, entry: np.ndarray, learning_rate: float):
        O = np.dot(self.weights, entry)
        delta_w = learning_rate * (O * entry - (O**2) * self.weights)
        self.weights += delta_w

        return delta_w