import pandas as pd 
import numpy as np 
import os


def load_data(data_name, split):
    curr_dir = os.getcwd()
    data_path = os.path.join(curr_dir, "Data", data_name)
    dataframe_path = os.path.join(data_path, f"{split}_{data_name.lower()}.csv")
    df = pd.read_csv(dataframe_path)
    
    return df