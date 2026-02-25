import pandas as pd

def load_data(file_path):
    """
    Loads wind turbine dataset.
    Returns dataframe.
    """
    df = pd.read_csv(file_path)
    return df