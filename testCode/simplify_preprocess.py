#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir, mkdir
from os.path import isfile, isdir, join, splitext
from scipy import interpolate, signal
import pandas as pd
import numpy as np
import shutil

ACTIVITIES = {
    'walking'   :1,
    'upstairs'  :2,
    'downstairs':3,
    'sitting'   :4,
    'standing'  :5,
    'laying'    :6,
}
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]
INTERPOLATE = ["nearest","zero","slinear","quadratic","cubic"] # interpolate function


FIX = [5.83031, -0.57800, -2.52145] # Arduino gyroscope 校正
GRAVITY = 9.80665                   # Gravitational constant
g = 0.3                             # cutoff frequence of g
cutoff = 20                         # cutoff frequence of output
freq = 50                           # input frequence
NewPath = '/preprocessed'


# In[52]:


def Get_path():
    source = 0
    document = ''

    while(source==0):
        print('Please choose data source:\n1) Smartphone\n2) Arduino')
        temp = int(input())
        if temp in range(1,3): source = temp
        else: print('Wrong!')
    print('\n------------------------------------------\n')
    
    if source==1: document = '../Data_allocation/smartphone'
    else: document = '../Data_allocation/arduino'
    return source, document

def find_csv(source, document):
    if source==1: print('Data comes from Smartphone:')
    else: print('Data comes from Arduino:')
        
    files = listdir(document)
    removelist = list() 
    for f in files:
        fullpath = join(document, f)
        if isfile(fullpath):
            if splitext(f)[1] != '.csv': removelist.append(f)
        else: removelist.append(f)
                
    for f in removelist: files.remove(f)
    for index in range(len(files)): files[index] = document+'/'+files[index]
    
    if files==[]:
        print(f'There has no csv file in \"{document}\"')
        print('\n-------------------------------------------------------\n')
    else:
        for f in files: print(f)
        print('\n------------------------------------------\n')
    return files


# In[53]:


def read_csv(file):
    return pd.read_csv(file, delim_whitespace=True, header=None).to_numpy()

def split_string_to_float(file, source):
    data = []

    if source==1:
        aX = []; aY = []; aZ = []
        for line in file:
            # [ "UID", "a/gx", "a/gy", "a/gz", "timestamp", "abs", "accuracy"]
            #  => [ a/gx, a/gy, a/gz ]
            aX.append( float(line[0].split(',')[1].split("\"")[1]) )
            aY.append( float(line[0].split(',')[2].split("\"")[1]) )
            aZ.append( float(line[0].split(',')[3].split("\"")[1]) )
        return [aX,aY,aZ], len(aX)
    else:
        aX = []; aY = []; aZ = []; gX = []; gY = []; gZ = []

        for line in file:
            # ['counter, ax, ay, az, gx, gy, gz']
            #  => [ ax, ay, az, gx, gy, gz ]
            aX.append( -float(line[0].split(',')[1]) )
            aY.append( float(line[0].split(',')[2]) )
            aZ.append( -float(line[0].split(',')[3]) )
            gX.append( float(line[0].split(',')[4]) )
            gY.append( float(line[0].split(',')[5]) )
            gZ.append( float(line[0].split(',')[6]) )
        return [aX,aY,aZ], len(aX), [gX,gY,gZ], len(gX)

def fix(data, length, source):
    for i in range(3):
        for j in range(length):
            data[i][j] = data[i][j]/GRAVITY
    if source==2:
        for i in range(3):
            for j in range(length):
                data[i+3][j] = data[i+3][j]-FIX[j]
    return data

#     for arduino device: x+向下 / y+向左 / z+向前 (UCI dataset & Arduino裝置)
#     for smartphones in pocket of right leg - sitting & laying: x+向左 / y+向前 / z+向下,
#                                                        others: x+向左 / y+向下 / z+向前
def genXYZ(data, activity):
    if activity in [4,6]:
        return data[2],data[0],data[1]
    else:
        return data[1],data[0],data[2]

def Load_data(source, paths):
    Activity=0
    acce_data = []; acce_length = 0
    gyro_data = []; gyro_length = 0

    if source==1:
        while(Activity==0):
            print("Please choose your activity:\n1) Walking\n2) Upstairs\n3) Downstairs\n4) Sitting\n5) Standing\n6) Laying")
            temp = int(input())
            if temp in range(1,7): Activity = temp
            else: print('Wrong!')
        print('\n------------------------------------------\n')
        
        acce_data, acce_length = split_string_to_float(read_csv(paths[0]), source)
        gyro_data, gyro_length = split_string_to_float(read_csv(paths[1]), source)

        acce_data = fix(acce_data, acce_length, source)

        acce_data = genXYZ(acce_data, Activity)
        gyro_data = genXYZ(gyro_data, Activity)


    else:
        Activity = ACTIVITIES[paths[0].split('/')[3].split('_')[0]]
        acce_data, acce_length, gyro_data, gyro_length = split_string_to_float(read_csv(paths[0]), source)

    print(f'Data length of acceleration:\t{acce_length}')
#     print(acce_data[0][:5])
#     print(acce_data[1][:5])
#     print(acce_data[2][:5])
    print(f'Data length of gyroscope:\t{gyro_length}')
#     print(gyro_data[0][:5])
#     print(gyro_data[1][:5])
#     print(gyro_data[2][:5])
    
    print('\n------------------------------------------\n')
    return Activity, acce_data, acce_length, gyro_data, gyro_length


# In[54]:


def Interpolate(data, old_length, new_length, kind):    
    old_samples=np.linspace(0, old_length, old_length)
    new_samples=np.linspace(0, old_length, new_length)
    
    fx = interpolate.interp1d(old_samples, data[0], kind=kind)
    fy = interpolate.interp1d(old_samples, data[1], kind=kind)
    fz = interpolate.interp1d(old_samples, data[2], kind=kind)
    
    return fx(new_samples), fy(new_samples), fz(new_samples)

def MedianFilter(data):
    return signal.medfilt(data, 3)

def LowPassButterworthFilter(cutoff, freq, data):
    b,a = signal.butter(3, 2*cutoff/freq, btype='lowpass', analog=False, output='ba')
    return signal.filtfilt(b, a, data)

def _split(samples, data):
    after = []
    for index in range(samples):
        after.append( data[index*64:(index+2)*64] )
    return np.transpose(after, (0,1))
    
def Shape(data, length):
    samples = int(length/64)-1
    return _split(samples, data[0]), _split(samples, data[1]), _split(samples, data[2])

def Preprocess(acce_data, acce_length, gyro_data, gyro_length):
    # interpolate
    data_length = acce_length
    if acce_length > gyro_length:
        gyro_data = Interpolate(gyro_data, gyro_length, acce_length, INTERPOLATE[4])
    elif acce_length < gyro_length:
        acce_data = Interpolate(acce_data, acce_length, gyro_length, INTERPOLATE[4])
        data_length = gyro_length
    
    
    # filt
    _MF_acce = [MedianFilter(acce_data[0]), MedianFilter(acce_data[1]), MedianFilter(acce_data[2])]
    _MF_gyro = [MedianFilter(gyro_data[0]), MedianFilter(gyro_data[1]), MedianFilter(gyro_data[2])]

    _LPBWF_acce = [LowPassButterworthFilter(cutoff, freq, _MF_acce[0]),
                   LowPassButterworthFilter(cutoff, freq, _MF_acce[1]),
                   LowPassButterworthFilter(cutoff, freq, _MF_acce[2])]
    _LPBWF_G = [LowPassButterworthFilter(g, freq, _MF_acce[0]),
                LowPassButterworthFilter(g, freq, _MF_acce[1]),
                LowPassButterworthFilter(g, freq, _MF_acce[2])]
    _LPBWF_gyro = [LowPassButterworthFilter(cutoff, freq, _MF_gyro[0]),
                   LowPassButterworthFilter(cutoff, freq, _MF_gyro[1]),
                   LowPassButterworthFilter(cutoff, freq, _MF_gyro[2])]
    
    # 'total_acc': The acceleration signal in standard gravity units 'g'.
    # 'body_acc': The body acceleration signal obtained by subtracting the gravity from the total acceleration. 
    # 'body_gyro': The angular velocity vector measured by the gyroscope for each window sample. The units are radians/second.     
    total_acce_data = _LPBWF_acce
    body_gyro_data = _LPBWF_gyro
    body_acce_data = [_LPBWF_acce[0]-_LPBWF_G[0],
                      _LPBWF_acce[1]-_LPBWF_G[1],
                      _LPBWF_acce[2]-_LPBWF_G[2]]
    
    
    # splite
    samples = int(data_length/64)-1
    print(f'{samples} samples')
    total_acc = Shape(total_acce_data, data_length)
    body_gyro = Shape(body_gyro_data, data_length)
    body_acc = Shape(body_acce_data, data_length)
    
    print('\n------------------------------------------\n')
    return data_length, samples, body_acc, body_gyro, total_acc


# In[55]:


def WriteSignal(source, activity, samples, body_acce, body_gyro, total_acce):
    document = ''
    if source==1: document = '../Data_smartphone'
    else: document = '../Data_arduino'
    print(f'Saving Data in:')
#     Save data
    index = 0
    for signal in SIGNALS:
        index = index+1
        filename = f'{document}/{signal}.txt'
        print(filename)
        
        # w 建&寫 / w+ 建&寫&讀 / r 讀 / r+ 讀&寫 / a 續寫 / a+續寫&讀 / b 二進位模式
        with open(filename, 'a') as f:
        
            if index<=3: target = body_acce[index-1]
            elif index<=6: target = body_gyro[index-4]
            elif index<=9: target = total_acce[index-7]
            
            np.savetxt(f, target, fmt="%.6e")
            # %.6e 科學記號到小數第6位 / %d 整數 / %.2f 浮點數到小數第2位 / %s 字串
    
#     Save target
    y = np.empty(shape=(samples,1))
    y.fill(activity)
    filename = f'{document}/y.txt'
    with open(filename, 'a') as f:
        np.savetxt(f, y, fmt="%d")
    print(filename)
    print('\n------------------------------------------\n')


# In[56]:


def MoveFile(source, activity, files, document):
    length = len( listdir(document) )
    activity = list (ACTIVITIES.keys())[list (ACTIVITIES.values()).index(activity)]
    mkdir(f'{document}/{length}_{activity}')

    if source==1:
        shutil.move(files[0], document+f'/{length}_{activity}')
        shutil.move(files[1], document+f'/{length}_{activity}')
    else:
        shutil.move(files[0], document+f'/{length}_{activity}')
    
    print('Moving Data from:')
    print(files[0])
    print('to:')
    print(f'{document}/{length}_{activity}')
    
    print('\n-------------------------------------------------------\n')


# In[57]:


def main():
    Source, Document = Get_path()
    Files = find_csv(Source, Document)
    
    while Files!=[]:
        Activity, acce_data, acce_length, gyro_data, gyro_length = Load_data(Source, Files)
        data_length, samples, body_acc, body_gyro, total_acc = Preprocess(acce_data, acce_length, gyro_data, gyro_length)
        WriteSignal(Source, Activity, samples, body_acc, body_gyro, total_acc)
        MoveFile(Source, Activity, Files, Document+NewPath)
        
        if Source==1: break
        else: Files = find_csv(Source, Document)


# In[58]:


main()


# In[ ]:




