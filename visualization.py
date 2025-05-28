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

###    Hopfield ###

def cargar_matrices_bloqueadas(ruta_archivo):
    matrices = []
    with open(ruta_archivo, "r") as f:
        bloque = []
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue  # Saltar líneas vacías
            # 🔽 Limpieza extra: remover corchetes y convertir a float
            linea_limpia = linea.replace("[", "").replace("]", "").replace(",", " ")
            fila = [float(x) for x in linea_limpia.split()]
            if len(fila) != 5:
                raise ValueError(f"Fila con {len(fila)} elementos, se esperaban 5.")
            bloque.append(fila)
            if len(bloque) == 5:
                matrices.append(np.array(bloque))
                bloque = []
    return matrices


def mostrar_matriz(matriz, titulo="", guardar_como=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xticks(np.arange(5 + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(5 + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    for y in range(5):
        for x in range(5):
            if matriz[y][x] == 1:
                ax.text(x, y, "*", ha='center', va='center', fontsize=20)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    ax.set_title(titulo)

    if guardar_como:
        plt.savefig(guardar_como, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def visualizar_matrices(ruta_archivo, carpeta_salida="imagenes_estados",print_step = True):
    os.makedirs(carpeta_salida, exist_ok=True)
    matrices = cargar_matrices_bloqueadas(ruta_archivo)
    for idx, matriz in enumerate(matrices):
        nombre_archivo = os.path.join(carpeta_salida, f"estado_{idx + 1:02}.png")
        if print_step:
            mostrar_matriz(matriz, titulo=f"Paso {idx + 1}", guardar_como=nombre_archivo)
        else:
            mostrar_matriz(matriz, guardar_como=nombre_archivo)
    print(f"Generadas {len(matrices)} imágenes en la carpeta '{carpeta_salida}'.")

if __name__ == "__main__":

    ### Hopfield ###
    visualizar_matrices("output/hopfield_network_output.txt")

    visualizar_matrices("input_data/hopfield_patterns/a.txt","imagen_A",False)
    visualizar_matrices("input_data/hopfield_patterns/f.txt","imagen_F",False)
    visualizar_matrices("input_data/hopfield_patterns/i.txt","imagen_I",False)
    visualizar_matrices("input_data/hopfield_patterns/j.txt","imagen_J",False)
    visualizar_matrices("input_data/hopfield_patterns/t.txt","imagen_T",False)
    visualizar_matrices("input_data/hopfield_patterns/l.txt","imagen_L",False)
    visualizar_matrices("input_data/hopfield_patterns/o.txt","imagen_O",False)
    visualizar_matrices("input_data/hopfield_patterns/x.txt","imagen_X",False)
    visualizar_matrices("input_data/hopfield_patterns/z.txt","imagen_Z",False)
