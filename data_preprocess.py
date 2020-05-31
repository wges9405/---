import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
import pandas as pd
import numpy as np
import random

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

#-------------------------------------------------
def showplot(data):
    plt.plot(data)
    plt.show()
    
#-------------------------------------------------
def Normalization(data): #input size: 1 axis
    normalized = np.empty(shape=data.shape)
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    for row in range(data.shape[0]):
        if not std[row] == 0:
            for col in range(data.shape[1]):
                normalized[row][col] = (data[row][col] - mean[row]) / std[row]
    return normalized

#-------------------------------------------------
def MedianFilter(data):  #input size: 1 axis
	return signal.medfilt(data, [1,3])

def LowPassButterworthFilter(para, data):  #input size: 1 axis
	b, a = signal.butter(3, 2*para[0]/para[1], btype='lowpass', analog=False, output='ba')
	return signal.filtfilt(b, a, data)

#-------------------------------------------------
def FFT(data):  #input size: 1 axis
    return abs(fft(data))/128

#-------------------------------------------------
def preprocess(para, data):
#     _median = MedianFilter(data)
#     _butterworth = LowPassButterworthFilter(para, _median)
    if para[2]==True:
        _normalize = Normalization(data)
        _fft = FFT(_normalize)
        return _fft
    else:
        _fft = FFT(data)
        return _fft

#------------------------------------------------- 
def shuffle(data, index):
    return data[index]

#-------------------------------------------------
def _read_csv(filename):    
    return pd.read_csv(filename, delim_whitespace=True, header=None)

def load_signals(para, subset):
    tmp_signal = []
    time_data = []
    freq_data = []
    for signal in SIGNALS:
        filename = f'./{subset}Data/{signal}_{subset}.txt'

        raw_data = _read_csv(filename).to_numpy()
        if para[2]==True: raw_data = Normalization(raw_data)
        preprocessed_data = preprocess(para, raw_data)
            
        if 'body_acc' in signal:
            tmp_signal.append(raw_data)
            tmp_signal.append(preprocessed_data)
            
        time_data.append(raw_data)
        freq_data.append(preprocessed_data)
        
    for index in range(3):
        time_data.append(tmp_signal[index*2])
        freq_data.append(tmp_signal[index*2+1])
        
    time_data = np.transpose(time_data, (1, 2, 0))
    freq_data = np.transpose(freq_data, (1, 2, 0))
    
    if para[0]=='T':   return time_data.reshape(time_data.shape[0],time_data.shape[1],time_data.shape[2],1)
    elif para[0]=='F': return freq_data.reshape(freq_data.shape[0],freq_data.shape[1],freq_data.shape[2],1)
    elif para[0]=='B': return np.transpose([time_data, freq_data], (1, 2, 3, 0))
    elif para[0]=='A': return [time_data.reshape(time_data.shape[0],time_data.shape[1],time_data.shape[2],1),
                               freq_data.reshape(freq_data.shape[0],freq_data.shape[1],freq_data.shape[2],1),
                               np.transpose([time_data, freq_data], (1, 2, 3, 0))]

def load_targets(subset):
    filename = f'./{subset}Data/y_{subset}.txt'
    y = _read_csv(filename)[0]
    return pd.get_dummies(y).to_numpy()

#-------------------------------------------------
def load_data(para):
    X_train = shuffle(load_signals(para, 'train'))
    X_test  = load_signals(para, 'test')
    y_train = shuffle(load_targets('train'))
    y_test  = load_targets('test')

    return X_train, X_test, y_train, y_test

#-------------------------------------------------
def load_ALL_data():
    X_train = []
    X_test = []
    y_train = load_targets('train')
    y_test  = load_targets('test')
    
    X_train.append(load_signals(['A', 20, 50, True], 'train'))
    X_train.append(load_signals(['A', 20, 50, False], 'train'))
    
    X_test.append(load_signals(['A', 20, 50, True], 'test'))
    X_test.append(load_signals(['A', 20, 50, False], 'test'))
    
    index = np.arange(y_train.shape[0])
    np.random.shuffle(index)
    
    for data in X_train[0]: data = shuffle(data, index)
    for data in X_train[1]: data = shuffle(data, index)
    y_train = shuffle(y_train, index)

    return X_train, X_test, y_train, y_test