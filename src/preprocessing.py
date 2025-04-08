import pandas as pd

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path, header=None)

    df.dropna(inplace=True)

    df.to_csv(output_path, index=False)

preprocess("data/adult.data", "data/clean_adult.csv")
