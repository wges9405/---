import pandas as pd
import numpy as np
import filters

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z"
]


def _read_csv(filename):    
    return pd.read_csv(filename, delim_whitespace=True, header=None)

def pre_process(cutoff_freq, sensor_freq, analog, data):
    medianFilted = filters.MedianFilter(data)
    lowPassButterworthFilted = filters.LowPassButterworthFilter(cutoff_freq,
                                                                sensor_freq,
                                                                analog,
                                                                medianFilted)
    return lowPassButterworthFilted

def normalization(data):
    normalized = np.empty(shape=data.shape)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if not std[col] == 0:
                normalized[row][col] = (data[row][col] - mean[col]) / std[col]
    return normalized

def load_signals(cutoff_freq, sensor_freq, analog, subset):
    signals_data = []

    for signal in SIGNALS:
        filename = f'./{subset}Data/{signal}_{subset}.txt'

        raw_data = _read_csv(filename)
        filted_data = pre_process(cutoff_freq, sensor_freq, analog, raw_data)
        normalized_data = normalization(filted_data)
        
        signals_data.append(normalized_data)

    # Transpose is used to change the dimensionality of the output,
    # aggregating the signals by combination of sample/timestep.
    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset):
    """
    The objective that we are trying to predict is a integer, from 1 to 6,
    that represents a human activity. We return a binary representation of 
    every sample objective as a 6 bits vector using One Hot Encoding
    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
    """
    filename = f'./{subset}Data/y_{subset}.txt'
    y = _read_csv(filename)[0]

    return pd.get_dummies(y).to_numpy()

def load_data(cutoff_freq, sensor_freq, analog):
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train = load_signals(cutoff_freq, sensor_freq, analog, 'train')
    X_test  = load_signals(cutoff_freq, sensor_freq, analog, 'test')
    y_train = load_y('train')
    y_test  = load_y('test')
    return X_train, X_test, y_train, y_test