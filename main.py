import pandas as pd
from neural_networks.models.kohonen.kohonen_network import KohonenNetwork
from neural_networks.similarity_functions import euclidean_distance, exponential_similarity
from normalization import Normalization
import json
import numpy as np
from visualization import create_heatmap_for_kohonen_network, create_distance_map

np.random.seed(43)

if __name__ == '__main__':
    with open("config.json") as f:
        config = json.load(f)

    similarity_functions = {"euclidean_distance": euclidean_distance, "exponential": exponential_similarity}

    df = pd.read_csv("input_data/europe.csv")  
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    x = df[features]

    # kohonen
    kohonen_config = config['kohonen']

    # calculate best k
    k_values = kohonen_config['k']
    selected_similarity_function = similarity_functions[kohonen_config['similarity_function']]
    initialize_random_weights = bool(kohonen_config['initialize_random_weights'])

    R_values = kohonen_config['R']
    epoch_values = kohonen_config['epochs']
    learning_rates = kohonen_config['learning_rate']
    learning_rate_variation = kohonen_config['learning_rate_variation']
    r_variation = kohonen_config['r_variation']

    standarized_x = Normalization(x).standarize()
    entries = standarized_x.astype(float).to_numpy()

    for k in k_values:
        for R in R_values:
            for epochs in epoch_values:
                for learning_rate in learning_rates: 
                    kohonen_network = KohonenNetwork(entries, len(features), k, selected_similarity_function, initialize_random_weights)
                    entries_per_neuron, epoch = kohonen_network.classify(R, epochs, learning_rate, r_variation, learning_rate_variation)

                    heatmap_data = np.empty((k, k), dtype=int)

                    for (i,j), entry in np.ndenumerate(entries_per_neuron):
                        heatmap_data[i, j] = len(entry)
    
                    create_heatmap_for_kohonen_network(heatmap_data,k, R, epochs)
                    create_distance_map(kohonen_network.output_layer.neuron_matrix)

                    

