"""
Utility functions.
"""

import pandas as pd


def load_data(filepath):
    """
    Loads data from csv.

    :param filepath: Path for csv file.
    :return: Dataframe of sensory data.
    """
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(filepath)
    df = df[["J1", "J2", "J3", "J4", "J5", "J6"]]
    # Extract the signal values as a numpy array
    energy_consumption_values = df.values[:, 0:]

    return energy_consumption_values
