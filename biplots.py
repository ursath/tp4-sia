import pandas as pd
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

if not os.path.exists("graphs"):
    os.makedirs("graphs")

def pca_biplot(score, coeff, labels=None, labels_delta_dict:dict=None, point_labels=None, point_labels_delta_dict:dict=None):
    xs, ys = score[:, 0], score[:, 1]
    n = coeff.shape[0]
    
    plt.figure(figsize=(10, 7))
    plt.scatter(xs, ys, alpha=0.6)

    if point_labels is not None:
        for i, txt in enumerate(point_labels):
            if txt in point_labels_delta_dict:
                plt.text(xs[i]+point_labels_delta_dict[txt][0], ys[i] + point_labels_delta_dict[txt][1], txt, fontsize=9, alpha=0.7)
            else:
                plt.text(xs[i]+0.1, ys[i]+0.1, txt, fontsize=9, alpha=0.7)
    
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0]*2, coeff[i, 1]*2, color='r', alpha=0.5, head_width=0.05)
        if labels is None:
            plt.text(coeff[i, 0]*2.2, coeff[i, 1]*2.2, "Var"+str(i+1), color='g')
        else:
            if labels[i] in labels_delta_dict:
                plt.text(coeff[i, 0] * 2.2 + labels_delta_dict[labels[i]][0], coeff[i, 1] * 2.2 + labels_delta_dict[labels[i]][1], labels[i], color='g')
            else:
                plt.text(coeff[i, 0]*2.2, coeff[i, 1]*2.2, labels[i], color='g')

    plt.xlabel("PC1", fontsize=10)
    plt.ylabel("PC2", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig("graphs/pca_biplot.png")

if __name__ == '__main__':
    df = pd.read_csv("input_data/europe.csv")  

    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    x = df[features]
    # standarize the data 
    x = StandardScaler().fit_transform(x)

    x_scaled_df = pd.DataFrame(x, columns=features)
    x_scaled_df['Country'] = df['Country']

    # Melt the DataFrame for seaborn boxplot
    melted_df = x_scaled_df.melt(id_vars='Country', var_name='Feature', value_name='Valor estandarizado')

    # Plot boxplots
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_df, x='Feature', y='Valor estandarizado', palette='Set2')
    plt.title("Boxplots de las features estandarizadas")
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("graphs/boxplots.png")

    # we need two components for biplot: PC1 and PC2
    pca = PCA(n_components=2) 
    # we reduce dimensionality on x
    principal_components = pca.fit_transform(x)

    point_labels_delta_per_country = {
                                "Switzerland":[-0.3, 0.1], "Luxembourg":[-0.5, 0.1], "Belgium":[-0.6, 0.05], 
                                "Germany": [-0.7, -0.1], "Portugal": [-0.4, -0.2], "Lithuania": [-0.4, -0.2], "Spain": [-0.3, -0.2], "Slovenia": [-0.4, -0.2], "United Kingdom": [-0.7, -0.2], "Sweden": [-0.1, -0.2], "Slovakia": [-0.3, -0.2],
                                "Norway": [0.1, -0.1], "Denmark": [0.1, 0.0],
                                "Poland": [-0.3, 0.1], "Netherlands":[0.1, 0.1],
                                "Bulgaria": [-0.3, 0.1], "Estonia": [-0.3, 0.1], "Latvia": [-0.3, 0.1],
                                "Ireland": [0.1, 0], "Finland": [0.1,0], "Czech Republic": [0.1, 0], "Hungary": [0.1, 0]}
    labels_delta_features = {
                                "Unemployment": [-1.0, 0.0],
                                "Area": [-0.2, -0.2],
                                "Inflation": [-0.6, 0.0]
    }

    pca_biplot(principal_components, pca.components_.T, features, labels_delta_features, df['Country'], point_labels_delta_per_country)
