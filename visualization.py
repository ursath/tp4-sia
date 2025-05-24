import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import seaborn as sns

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

from scipy.spatial.distance import euclidean # TODO sacar por la nuestra

def create_distance_map(neuron_matrix):
    rows = len(neuron_matrix)
    cols = len(neuron_matrix[0])
    dist_map = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            current_weights = neuron_matrix[i][j].weights
            neighbors = []

            if i > 0:
                neighbors.append(neuron_matrix[i - 1][j].weights)
            if i < rows - 1:
                neighbors.append(neuron_matrix[i + 1][j].weights)
            if j > 0:
                neighbors.append(neuron_matrix[i][j - 1].weights)
            if j < cols - 1:
                neighbors.append(neuron_matrix[i][j + 1].weights)

            if neighbors:
                distances = [euclidean(current_weights, n) for n in neighbors]
                dist_map[i, j] = np.mean(distances)

    plot_distance_map(dist_map)

def plot_distance_map(dist_map):
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_map, cmap="plasma", annot=False, square=True, linewidths=0.3)
    plt.title("Mapa de distancias promedio entre neuronas")
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.tight_layout()
    plt.savefig("graphs/kohonen_distance_map.png")
    plt.show()

