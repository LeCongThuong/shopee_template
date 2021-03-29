import pandas as pd


def read_csv(data_file):
    if data_file.endswith(".pkl"):
        df = pd.read_pickle(data_file)
    else:
        df = pd.read_csv(data_file)
    return df
