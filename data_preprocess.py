import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

SIGNALS = [
	"body_acc_x"#, "body_acc_y", "body_acc_z",
# 	"body_gyro_x", "body_gyro_y", "body_gyro_z",
# 	"total_acc_x", "total_acc_y", "total_acc_z",
# 	"body_acc_x", "body_acc_y", "body_acc_z"
]

#----------------------------------------------------------------------------------------#
# Debug with matplotlib
#----------------------------------------------------------------------------------------#
def showplot(data):
    plt.subplot(321)
    plt.plot(data[0])
    plt.title('original')
    plt.subplot(322)
    plt.plot(data[1])
    plt.title('median')
    plt.subplot(323)
    plt.plot(data[2])
    plt.plot('butterworth')
    plt.subplot(324)
    plt.plot(data[3])
    plt.plot('fft')
    plt.subplot(325)
    plt.plot(data[4])
    plt.plot('normalize')
    plt.show()


#----------------------------------------------------------------------------------------#
# Filters: Median filter & low-pass butterworth filter
#----------------------------------------------------------------------------------------#
from scipy import signal

def MedianFilter(data):  #input size: 1 axis
	return signal.medfilt(data, [1,3])

def LowPassButterworthFilter(para, data):  #input size: 1 axis
	b, a = signal.butter(3, 2*para[0]/para[1], btype='lowpass', analog=False, output='ba')
	return signal.filtfilt(b, a, data)


#----------------------------------------------------------------------------------------#
# Fast Fourier Transform
#----------------------------------------------------------------------------------------#
from scipy.fftpack import fft

def FFT(data):  #input size: 1 axis
	return abs(fft(data))/128


#----------------------------------------------------------------------------------------#
# Normalization
#----------------------------------------------------------------------------------------#
def Normalization(data): #input size: 1 axis
	normalized = np.empty(shape=data.shape)
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	for row in range(data.shape[0]):
		for col in range(data.shape[1]):
			if not std[col] == 0:
				normalized[row][col] = (data[row][col] - mean[col]) / std[col]
	return normalized


#----------------------------------------------------------------------------------------#
# Summation of preprocesses
#----------------------------------------------------------------------------------------#
def preprocess(para, data):    
    data = data.to_numpy()
    _median = MedianFilter(data)
    _butterworth = LowPassButterworthFilter(para, _median)
    _fft = FFT(_butterworth)
#     _normalize = Normalization(_fft)
#----------------------------------------------------------------------------------------#
# Debug with matplotlib
#----------------------------------------------------------------------------------------#
#     for int in range(5):
#         index = random.randint(0,7352)
#         showplot([data[index], _median[index], _butterworth[index], _fft[index], _normalize[index]])
    return _fft

def _read_csv(filename):    
	return pd.read_csv(filename, delim_whitespace=True, header=None)

def load_signals(para, subset):
	signals_data = []

	for signal in SIGNALS:
		filename = f'./{subset}Data/{signal}_{subset}.txt'

		raw_data = _read_csv(filename)
		preprocessed_data = preprocess(para, raw_data)
        
		signals_data.append(preprocessed_data)
	return np.transpose(signals_data, (1, 2, 0))

def load_targets(subset):
	filename = f'./{subset}Data/y_{subset}.txt'
	y = _read_csv(filename)[0]
	return pd.get_dummies(y).to_numpy()

def load_data(para):
	X_train = load_signals(para, 'train')
	X_test  = load_signals(para, 'test')
	y_train = load_targets('train')
	y_test  = load_targets('test')
	
	return X_train, X_test, y_train, y_test