import json
from neural_networks.models.oja.oja_network import OjaNetwork
import pandas as pd
from normalization import Normalization
from sklearn.decomposition import PCA
from visualization import plot_pca_comparison
import numpy as np

np.random.seed(43)

if __name__ == '__main__':
     with open("config.json") as f:
        config = json.load(f)

        oja_config = config['oja']
        epochs = oja_config['epochs']
        learning_rate = oja_config['learning_rate']

        df = pd.read_csv("input_data/europe.csv")  
        features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
        x = df[features]

        standarized_x = Normalization(x).standarize()
        entries = standarized_x.astype(float).to_numpy()

        for epoch_value in epochs:
            for learning_rate_value in learning_rate:
                oja_network = OjaNetwork(entries, len(features), learning_rate_value, epoch_value)
                pca = oja_network.classify()
                print(f"PCA (our algorithm): {pca}")

                standarized_x = Normalization(x).standarize()
                entries = standarized_x.astype(float).to_numpy()

                pca_lib = PCA(n_components=1)
                pca_lib.fit(entries)

                first_component = pca_lib.components_[0]  

                print("PCA (library):", first_component)

                print("Difference:", first_component - pca) 

                plot_pca_comparison(features, pca, first_component)
        
