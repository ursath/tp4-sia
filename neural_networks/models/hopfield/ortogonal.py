from itertools import combinations
import numpy as np
import os

def find_best_and_worst_orthogonal_patterns(patterns):
    """
    Recibe: lista de diccionarios con claves: 'filename' y 'flat_pattern'
    Devuelve: los 4 patrones más ortogonales y los 4 menos ortogonales (con filenames)
    """
    num_patterns = len(patterns)
    dot_products = {}

    # Crea mapa de productos escalares absolutos entre todos los pares
    for i, j in combinations(range(num_patterns), 2):
        pi = patterns[i]["flat_pattern"]
        pj = patterns[j]["flat_pattern"]
        dot = abs(np.dot(pi, pj))
        dot_products[(i, j)] = dot

    best_combo = None
    worst_combo = None
    best_score = float('inf')
    worst_score = float('-inf')

    # Evalúa todas las combinaciones posibles de 4 patrones
    for combo in combinations(range(num_patterns), 4):
        score = sum(dot_products.get((min(i, j), max(i, j)), 0)
                    for i, j in combinations(combo, 2))  # 

        if score < best_score:
            best_score = score
            best_combo = combo

        if score > worst_score:
            worst_score = score
            worst_combo = combo

    return {
        "best_patterns": [patterns[i]["filename"] for i in best_combo],
        "best_score": best_score,
        "worst_patterns": [patterns[i]["filename"] for i in worst_combo],
        "worst_score": worst_score
    }


def sum_orthogonalities(patterns_subset):
    """
    Recibe una lista de 4 patrones (cada uno con 'flat_pattern' como np.array)
    Devuelve la suma de los productos escalares absolutos entre todos los pares
    """
    if len(patterns_subset) != 4:
        raise ValueError("Debe recibir exactamente 4 patrones")

    total = 0
    for p1, p2 in combinations(patterns_subset, 2):
        dot = np.dot(p1["flat_pattern"], p2["flat_pattern"])
        total += abs(dot)

    return total

# main
if __name__ == "__main__":
    patterns_dir = "input_data/hopfield_patterns"

    patterns = []
    file_names = []

    # Crea una lista de patrones a partir de los archivos .txt
    for filename in os.listdir(patterns_dir):
        p = []
        if filename.endswith(".txt"):
            with open(os.path.join(patterns_dir, filename), "r") as f:
                p.append([x.split() for x in f.read().splitlines()])
            p = [[int(x) for x in sublist] for sublist in p[0]]
            flat_pattern = [line for sublist in p for line in sublist]
            patterns.append({"filename": filename, "flat_pattern": flat_pattern})
    
    # Convertimos los patrones a arrays de NumPy para hacer cálculos vectoriales
    np_patterns = []    
    for pattern in patterns:
        np_patterns.append(np.array(pattern))

    result = find_best_and_worst_orthogonal_patterns(patterns)

    print("Más ortogonales:", result["best_patterns"], "-> score:", result["best_score"])
    print("Menos ortogonales:", result["worst_patterns"], "-> score:", result["worst_score"])