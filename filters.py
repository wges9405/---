from scipy import signal

def LowPassButterworthFilter(cutoff_freq, sensor_freq, analog, data):
    b, a = signal.butter(3, 2*cutoff_freq/sensor_freq, btype='lowpass', analog=analog, output='ba')
    return signal.filtfilt(b, a, data)

def MedianFilter(data):
    return signal.medfilt(data)