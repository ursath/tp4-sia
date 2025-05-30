import pandas as pd
from neural_networks.models.kohonen.kohonen_network import KohonenNetwork
from neural_networks.similarity_functions import euclidean_distance, exponential_similarity
from normalization import Normalization
import json
import numpy as np
from visualization import create_heatmap_for_kohonen_network, create_heatmap_with_country_labels_for_kohonen_network, create_distance_map, visualize_single_variable
import os

np.random.seed(43)

if __name__ == '__main__':
    with open("config.json") as f:
        config = json.load(f)

    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    similarity_functions = {"euclidean_distance": euclidean_distance, "exponential": exponential_similarity}

    df = pd.read_csv("input_data/europe.csv")  
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    x = df[features]
    countries_dict = df['Country'].astype(str).to_dict()

    # kohonen
    kohonen_config = config['kohonen']

    # calculate best k
    k_values = kohonen_config['k'] # tamaño del mapa (#neuronas = k x k)
    selected_similarity_function = similarity_functions[kohonen_config['similarity_function']]
    initialize_random_weights = kohonen_config['initialize_random_weights'] == 'True'


    R_values = kohonen_config['R']
    epoch_values = kohonen_config['epochs']
    learning_rates = kohonen_config['learning_rate'] 
    learning_rate_variation = kohonen_config['learning_rate_variation'] == 'True'
    r_variation = kohonen_config['r_variation'] == 'True'

    # Estandarizo los datos
    standarized_x = Normalization(x).standarize()

    # Cada entrada es un array de las features de un país
    entries = standarized_x.astype(float).to_numpy()

    for k in k_values:
        for R in R_values:
            for epochs in epoch_values:
                for learning_rate in learning_rates: 
                    kohonen_network = KohonenNetwork(entries, len(features), k, selected_similarity_function, initialize_random_weights)
                    entries_per_neuron, epoch = kohonen_network.train(R, epochs, learning_rate, r_variation, learning_rate_variation)

                    # Generamos los gráficos
                    heatmap_data = np.empty((k, k), dtype=int)
                    countries_data = np.empty((k, k), dtype=object)

                    for (i,j), entry in np.ndenumerate(entries_per_neuron):
                        countries = []
                        for index in entry:
                            country = countries_dict[index].upper()[:3]
                            if index == 21:
                                country = "SLK"
                            if index == 22:
                                country = "SLN"
                            if index == 27:
                                country = "UK"
                            countries.append(country)
                        heatmap_data[i, j] = len(entry)
                        countries_data[i, j] = countries
    
                    create_heatmap_for_kohonen_network(heatmap_data,k, R, epochs, initialize_random_weights, learning_rate, learning_rate_variation, r_variation)
                    create_heatmap_with_country_labels_for_kohonen_network(heatmap_data, countries_data, k, R, epochs, initialize_random_weights, learning_rate, learning_rate_variation, r_variation)
                    create_distance_map(kohonen_network.output_layer.neuron_matrix, k, R, epochs, learning_rate, initialize_random_weights, r_variation, learning_rate_variation)
                    
                    visualize_single_variable(kohonen_network.output_layer.neuron_matrix,0)
                    visualize_single_variable(kohonen_network.output_layer.neuron_matrix,1)
                    visualize_single_variable(kohonen_network.output_layer.neuron_matrix,2)
                    visualize_single_variable(kohonen_network.output_layer.neuron_matrix,3)
                    visualize_single_variable(kohonen_network.output_layer.neuron_matrix,4)
                    visualize_single_variable(kohonen_network.output_layer.neuron_matrix,5)
                    visualize_single_variable(kohonen_network.output_layer.neuron_matrix,6)

                    

