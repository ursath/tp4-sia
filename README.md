# tp4-sia

Cada modelo tiene su propio main:
hopfield_main
oja_main
kohonen_main

En config.json se definen los parámetros a utilizar en cada modelo
Ejemplo:
```
{
    "kohonen": {
        "k": [ 3 ] ,
        "R": [ 3 ],
        "epochs": [ 3500 ],
        "learning_rate": [ 1 ],
        "initialize_random_weights": "False",
        "similarity_function": "euclidean_distance",
        "learning_rate_variation": "True",
        "r_variation": "True"
    },
    "oja": {
        "epochs": [ 1000 ],
        "learning_rate": [ 0.001 ]
    },
    "hopfield": {
        "patterns": [ "l.txt", "u.txt", "e.txt", "o.txt" ],
        "pattern_for_input": "e.txt",
        "noise_percentage": 0.3
    }
}
```

Para visualizar los gráficos debe crear las siguientes carpetas:
/graphs/kohonen/u_matrix
/graphs/kohonen/heatmap_with_countries

Para visualizar el output de hopfield es necesario correr el main de visualization luego del main