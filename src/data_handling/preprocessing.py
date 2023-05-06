"""
Functions for processing sensory data.
"""
import numpy as np


def create_signal_windows(robot_energy_consumption, size, overlap):
    """
    Creates windows of desired length and overlap from sensory data dataframe.

    :param robot_energy_consumption: Sensory data dataframe.
    :param size: Size of windows, e.g. 288.
    :param overlap: Number of overlapping samples, e.g. 96 = 288/3.
    :return: Array of data windows.
    """
    n_samples = robot_energy_consumption.shape[0]
    n_windows = (n_samples - size) // (size - overlap) + 1

    # Exclude the last window if it is smaller than the specified size
    if (n_samples - size) % (size - overlap) == 0:
        n_windows -= 1

    robot_energy_consumption_windows = np.zeros((n_windows, size, robot_energy_consumption.shape[1]))
    for i in range(n_windows):
        start = i * (size - overlap)
        end = start + size
        robot_energy_consumption_windows[i] = robot_energy_consumption[start:end]

    return robot_energy_consumption_windows
