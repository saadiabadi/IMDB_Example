import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def read_data(filename):
    """
    Helper function to read preprocessed data and split it.
    :return: train and test data
    """

    print("-- START READING DATA --")
    # The entire dataset is 50k reviews, we can subsample here for quicker testing.
    data_input = np.array(pd.read_csv(filename))
    print(data_input.shape)

    x_train = data_input[:, 1:-1]
    y_train = data_input[:, -1]
    # _, X, _, y = train_test_split(x_train, y_train, test_size=settings['test_size'])
    return x_train, y_train
