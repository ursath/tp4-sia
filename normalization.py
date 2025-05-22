import pandas as pd

class Normalization:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def standarize(self):
        """
        Normalize using: X' = (X - mean) / std
        """
        self.data = (self.data - self.data.mean()) / self.data.std()
        return self.data
    
    def unit_length_scaling(self):
        """
        Normalize using: X' = X / ||X||
        """
        self.data = self.data / self.data.abs()
        if self.data.isnull().values.any():
            self.data = self.data.fillna(0)

        return self.data
    
# def main():
#     df = pd.read_csv("input_data/europe.csv")  
#     features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
#     x = df[features]
    
#     normalizer = Normalization(x)
#     normalized_data = normalizer.unit_length_scaling()
#     print(normalized_data)

# if __name__ == "__main__":
#     main()