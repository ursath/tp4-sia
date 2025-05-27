import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import seaborn as sns
from neural_networks.similarity_functions import euclidean_distance

def create_heatmap_for_kohonen_network(data:List[int], k:int, R:float, epochs:int):
    matrix_indexes = np.arange(k)
    fig, ax = plt.subplots()
    plt.imshow(data, cmap='plasma', interpolation='nearest')
    plt.yticks(matrix_indexes)
    plt.xticks(matrix_indexes)
    plt.colorbar()
    for i in range(k):
        for j in range(k):
            ax.text(j, i,str(data[i, j]), ha='center', va='center', color='black')
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

def create_distance_map(neuron_matrix,k:int, R:float, epochs:int):
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
                distances = [euclidean_distance(current_weights, n) for n in neighbors]
                dist_map[i, j] = np.mean(distances)

    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_map, cmap="plasma", annot=False, square=True, linewidths=0.3)
    plt.title("Mapa de distancias promedio entre neuronas")
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.tight_layout()
    plt.savefig(f"graphs/kohonen_distance_map_k_{k}_R_{R}_epochs_{epochs}.png")

def plot_pca_comparison(features, pca_our, pca_lib):

    x = np.arange(len(features))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, pca_our, width, label='PCA (Nuestro Algoritmo)')
    ax.bar(x, pca_lib, width, label='PCA (Libreria)')

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_ylabel('Componente')
    ax.set_title('Comparacion de Componentes PCA')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("graphs/pca_comparison.png")