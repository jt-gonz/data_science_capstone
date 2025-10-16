import pandas as pd

def open_csv(file):
    df = pd.read_csv(file)
    return df

def run():
    mappings = open_csv("data/mapping.csv")
    print(mappings.head())

if __name__ == '__main__':
    run()