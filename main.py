import pandas as pd
from neural_networks.models.kohonen.kohonen_network import KohonenNetwork
from neural_networks.similarity_functions import euclidean_distance
import json

if __name__ == '__main__':
    with open("config.json") as f:
        config = json.load(f)

    similarity_functions = {"euclidean_distance": euclidean_distance}

    df = pd.read_csv("input_data/europe.csv")  
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    x = df[features]

    # kohonen
    kohonen_config = config['kohonen']

    # calculate best k
    k = int(kohonen_config['k'])
    selected_similarity_function = similarity_functions[kohonen_config['similarity_function']]
    initialize_random_weights = bool(kohonen_config['initialize_random_weights'])

    R = float(kohonen_config['R'])
    epochs = int(kohonen_config['epochs'])
    learning_rate = float(kohonen_config['learning_rate'])

    kohonen_network = KohonenNetwork(x, len(features), k, selected_similarity_function, initialize_random_weights)
    #kohonen_network.classify(R, epochs, learning_rate)