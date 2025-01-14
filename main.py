import pandas as pd 
import numpy as np 
from utils import load_data
import argparse


def main(args):

    data_name = args.data_name
    split = args.split    

    df = load_data(data_name, split)
    print(len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", "--data_name", type=str, required=True, help="dataset name: IMDb, Blog, Yelp")
    parser.add_argument("--s", "--split", type=str, required=True, help="data split: train, test, validation")

    args = parser.parse_args()
    main(args)