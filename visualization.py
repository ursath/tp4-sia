import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def create_heatmap_for_kohonen_network(data:List[int], k:int, R:float, epochs:int):
    matrix_indexes = np.arange(k)
    fig, ax = plt.subplots()
    plt.imshow(data, cmap='plasma', interpolation='nearest')
    plt.yticks(matrix_indexes)
    plt.xticks(matrix_indexes)
    plt.colorbar()
    for i in range(k):
        for j in range(k):
            ax.text(j, i, "'1': " + str(data[i, j]), ha='center', va='center', color='black')
    plt.savefig(f"graphs/kohonen_heatmap_k_{k}_R_{R}_epochs_{epochs}.png")

def create_heatmap_with_country_labels_for_kohonen_network(data:List[int], countries_per_neuron:List[str],k:int, R:float, epochs:int):
    matrix_indexes = np.arange(k)
    fig, ax = plt.subplots(figsize=(8,5))
    plt.imshow(data, cmap='plasma', interpolation='nearest')
    plt.yticks(matrix_indexes)
    plt.xticks(matrix_indexes)
    plt.colorbar()
    for i in range(k):
        for j in range(k):
            ax.text(j, i, '\n'.join(countries_per_neuron[i, j]) if len(countries_per_neuron[i, j]) > 0 else "-", ha='center', va='center', color='black')
    plt.savefig(f"graphs/kohonen_heatmap_with_countries_k_{k}_R_{R}_epochs_{epochs}.png")
