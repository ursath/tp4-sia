import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
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


# Para cada neurona calcula el promedio fr la distancia euclídea entre:
# ● el vector de pesos de la neurona
# ● el vector de pesos de las neuronas vecinas.
def create_distance_map(neuron_matrix,k:int, R:float, epochs:int, learning_rate:float, random_weights:bool, r_variation:bool, learning_rate_variation:bool):
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

    save_path = f"graphs/kohonen/u_matrix/u_matrix_k={k}_R={R}_epochs={epochs}_lr={learning_rate}"
    if(random_weights): 
        save_path += "_weights=random"
    if(r_variation):
        save_path += "_rVariation"
    if(learning_rate_variation):
        save_path += "_lrVariation"

    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_map, cmap="Greys", vmin=0, annot=False, square=True, linewidths=0.3, cbar=True)
    plt.title("Mapa de distancias promedio entre neuronas")
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


def plot_pca_comparison_features(features, pca_our, pca_lib):

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


def plot_pca_comparison_countries(countries, pca_our, pca_lib):
    x = np.arange(len(countries))
    width = 0.25

    order = np.argsort(pca_our)
    countries = countries[order]
    pca_our = pca_our[order]
    pca_lib = pca_lib[order]


    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, pca_our, width, label='PCA (Nuestro Algoritmo)')
    ax.bar(x, pca_lib, width, label='PCA (Librería)')

    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.set_ylabel('Valor Componente Principal')
    ax.set_title('Comparación de Componentes PCA por País')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("graphs/pca_comparison_by_country.png")



###    Hopfield   ###

def load_matrix(file_path):
    matrix = []
    with open(file_path, "r") as f:
        block = []
        for line in f:
            line = line.strip()
            if not line:
                continue  
            clean_line = line.replace("[", "").replace("]", "").replace(",", " ")
            row = [float(x) for x in clean_line.split()]
            if len(row) != 5:
                raise ValueError(f"Row has {len(row)} elements, expected 5.")
            block.append(row)
            if len(block) == 5:
                matrix.append(np.array(block))
                block = []
    return matrix


def display_matrix(matrix, title="", save_as=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xticks(np.arange(5 + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(5 + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    for y in range(5):
        for x in range(5):
            if matrix[y][x] == 1:
                ax.text(x, y, "*", ha='center', va='center', fontsize=20)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    ax.set_title(title)

    if save_as:
        plt.savefig(save_as, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def visualize_matrix(file_path, output_folder="output/state_images", show_step=True):
    os.makedirs(output_folder, exist_ok=True)
    matrix = load_matrix(file_path)
    for idx, matrix in enumerate(matrix):
        file_name = os.path.join(output_folder, f"state_{idx + 1:02}.png")
        if show_step:
            display_matrix(matrix, title=f"Step {idx + 1}", save_as=file_name)
        else:
            display_matrix(matrix, save_as=file_name)
    print(f"{len(matrix)} images generated in folder '{output_folder}'.")

def plot_energy_vs_iteration(energy_values, iteration_values, consulted_pattern, stored_pattern, noise):
    x_values = np.arange(1, len(energy_values)+1)
    plt.scatter(iteration_values, energy_values)
    plt.xlabel("Número de iteración")
    plt.ylabel("Valor de energía asociada a la red de Hopfield")
    plt.xticks(x_values)
    plt.tight_layout()
    plt.savefig(f"graphs/hopfield/energy_vs_iteration_for_consulted_{consulted_pattern}_noise_{noise}_with_{stored_pattern}.png")

if __name__ == "__main__":

    ### Hopfield ###
    visualize_matrix("output/hopfield_network_output.txt")
    visualize_matrix("input_data/hopfield_input.txt", "output/image_input", False)
