import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import seaborn as sns
from neural_networks.similarity_functions import euclidean_distance

def create_heatmap_for_kohonen_network(data:List[int], k:int, R:float, epochs:int, random_weights:bool, learning_rate:float, learning_rate_variation:bool, r_variation:bool):
    matrix_indexes = np.arange(k)
    fig, ax = plt.subplots()
    plt.imshow(data, cmap='plasma', interpolation='nearest', vmin=0, vmax=10)
    plt.yticks(matrix_indexes)
    plt.xticks(matrix_indexes)
    plt.colorbar()
    for i in range(k):
        for j in range(k):
            ax.text(j, i,str(data[i, j]), ha='center', va='center', color='black')
    save_path = f"graphs/kohonen/heatmap_k={k}_R={R}_epochs={epochs}_lr={learning_rate}"
    if(random_weights): 
        save_path += "_weights=random"
    if(r_variation):
        save_path += "_rVariation"
    if(learning_rate_variation):
        save_path += "_lrVariation"
    plt.savefig(f"{save_path}.png")


def create_heatmap_with_country_labels_for_kohonen_network(data:List[int], countries_per_neuron:List[str],k:int, R:float, epochs:int, random_weights:bool, learning_rate:float, learning_rate_variation:bool, r_variation:bool):
    matrix_indexes = np.arange(k)
    fig, ax = plt.subplots(figsize=(8,5))
    plt.imshow(data, cmap='plasma', interpolation='nearest', vmin=0, vmax=10)
    plt.yticks(matrix_indexes)
    plt.xticks(matrix_indexes)
    plt.colorbar()
    for i in range(k):
        for j in range(k):
            ax.text(j, i, '\n'.join(countries_per_neuron[i, j]) if len(countries_per_neuron[i, j]) > 0 else "-", ha='center', va='center', color='black')
    
    save_path = f"graphs/kohonen/heatmap_with_countries_k={k}_R={R}_epochs={epochs}_lr={learning_rate}"
    if(random_weights): 
        save_path += "_weights=random"
    if(r_variation):
        save_path += "_rVariation"
    if(learning_rate_variation):
        save_path += "_lrVariation"
    plt.savefig(f"{save_path}.png")


# Muestra qué tan diferentes son los pesos entre neuronas vecinas
# Ayuda a ver transiciones abruptas: zonas donde hay un cambio fuerte en las características representadas.
# Colores más oscuros → neuronas similares a sus vecinas.
# Colores más claros → neuronas muy distintas a sus vecinas 
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
    sns.heatmap(dist_map, cmap="plasma", vmin=0, vmax=1.5, annot=True, fmt=".2f", square=True, linewidths=0.3)
    plt.title("Mapa de distancias promedio entre neuronas")
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.tight_layout()
    plt.savefig(f"graphs/kohonen_distance_map_k_{k}_R_{R}_epochs_{epochs}.png")


def create_u_matrix(neuron_matrix, R=1.0, k=5, epochs=100, learning_rate=0.1, random_weights=False, r_variation=False, learning_rate_variation=False):
    rows = len(neuron_matrix)
    cols = len(neuron_matrix[0])
    dist_map = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            current_weights = neuron_matrix[i][j].weights
            neighbors = []

            # Buscar vecinos dentro del radio R
            for di in range(-int(np.ceil(R)), int(np.ceil(R)) + 1):
                for dj in range(-int(np.ceil(R)), int(np.ceil(R)) + 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        distance = np.sqrt(di**2 + dj**2)
                        if distance <= R:
                            neighbors.append(neuron_matrix[ni][nj].weights)

            if neighbors:
                distances = [euclidean_distance(current_weights, n) for n in neighbors]
                dist_map[i, j] = np.mean(distances)

    save_path = f"graphs/kohonen/u_matrix/u_matrix_k={k}_R={R}_epochs={epochs}_lr={learning_rate}"
    if(random_weights): 
        save_path += "_weights=random"
    if(r_variation):
        save_path += "_rVariation"
    if(learning_rate_variation):
        save_path += "_lrVariation"

    # Visualización en escala de grises
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_map, cmap="Greys", annot=False, square=True, linewidths=0.3, cbar=True)
    plt.title("Matriz U (Distancias promedio a vecinos)")
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.tight_layout()
    plt.savefig(f"{save_path}.png")


def visualize_single_variable(neuron_matrix, var_index: int):
    rows = len(neuron_matrix)
    cols = len(neuron_matrix[0])
    value_map = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # Obtenemos el valor de la variable 'var_index' del vector de pesos de cada neurona
            value_map[i, j] = neuron_matrix[i][j].weights[var_index]

    # Visualización en escala de grises
    plt.figure(figsize=(8, 6))
    sns.heatmap(value_map, cmap="plasma", annot=False, square=True, linewidths=0.3, cbar=True)
    plt.title(f"Distribución de la variable {var_index}")
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.tight_layout()
    plt.savefig(f"graphs/single_variable_{var_index}.png")
    plt.close()


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