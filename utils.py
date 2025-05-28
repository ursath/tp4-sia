import random

def save_input_pattern(patron_flat, output_dir):

    if len(patron_flat) != 25:
        raise ValueError("El patrón debe tener exactamente 25 elementos.")

    with open(output_dir, "w") as f:
        for i in range(5):
            fila = patron_flat[i*5:(i+1)*5]
            fila_str = " ".join(str(x) for x in fila)
            f.write(fila_str + "\n")

def apply_noise(patterns, noise_percentage, index=None):
    if index is None:
        index = random.randint(0, len(patterns) - 1)

    original = patterns[index]
    noisy = original.copy()

    num_to_flip = int(len(noisy) * noise_percentage)
    indices_to_flip = random.sample(range(len(noisy)), num_to_flip)

    for i in indices_to_flip:
        noisy[i] *= -1  

    return noisy