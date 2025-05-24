from .oja_neuron import OjaNeuron
from typing import List

class OjaNetwork:
    def __init__(self, dataset: List[any], features_len:int, learning_rate: float, epochs: int):
        self.dataset = dataset
        self.features_len = features_len
        self.learning_rate = learning_rate
        self.epochs = epochs

        input_dim = len(dataset[0])  
        self.neuron = OjaNeuron(input_dim)

    def classify(self):
        for _ in range(self.epochs):
            for entry in self.dataset:
                self.neuron.update_weights(entry, self.learning_rate)
                
        return self.neuron.weights

