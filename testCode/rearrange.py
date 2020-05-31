import pandas as pd
import numpy as np

Hz = 50
period = 2.56*2
length = int(period*Hz)

SIGNALS = [
	"body_acc_x", "body_acc_y", "body_acc_z",
	"total_acc_x", "total_acc_y", "total_acc_z",
	"body_gyro_x", "body_gyro_y", "body_gyro_z",
]

body_acc_x = pd.read_csv('../trainData/body_acc_x_train.txt', delim_whitespace=True, header=None).to_numpy()
# body_acc_y = pd.read_csv('../testData/body_acc_y_test.txt', delim_whitespace=True, header=None).to_numpy()
# body_acc_z = pd.read_csv('../testData/body_acc_z_test.txt', delim_whitespace=True, header=None).to_numpy()
# total_acc_x = pd.read_csv('../testData/total_acc_x_test.txt', delim_whitespace=True, header=None).to_numpy()
# total_acc_y = pd.read_csv('../testData/total_acc_y_test.txt', delim_whitespace=True, header=None).to_numpy()
# total_acc_z = pd.read_csv('../testData/total_acc_z_test.txt', delim_whitespace=True, header=None).to_numpy()
# body_gyro_x = pd.read_csv('../testData/body_gyro_x_test.txt', delim_whitespace=True, header=None).to_numpy()
# body_gyro_y = pd.read_csv('../testData/body_gyro_y_test.txt', delim_whitespace=True, header=None).to_numpy()
# body_gyro_z = pd.read_csv('../testData/body_gyro_z_test.txt', delim_whitespace=True, header=None).to_numpy()
y = pd.read_csv('../trainData/y_train.txt', delim_whitespace=True, header=None).to_numpy()
print(y[:10])
num = 0
index = 0
activities = []
cur_target = y[index][0]
target = []
while(index<7352):
    data = []
    list = []
    cur_target = y[index][0]
    target.append(y[index][0])
    while(body_acc_x[index][0]==body_acc_x[index-1][64]):
        data.extend(body_acc_x[index][:64])
        index = index + 1
        if (index>=7352): break
    data.extend(body_acc_x[index-1][64:])
#     print(len(data))
    activities.append(data)
    num = num + 1
print(num)
cur = 0
new_data = []
new_target = []
while(cur<num):
    start = 0;
    total = len(activities[cur])
    while(start+length<total):
        new_data.append(activities[cur][start:start+length])
        new_target.append(target[cur])
        start = start + length
    cur = cur + 1

def writeSignal(signal, name):
    with open('./'+name+'.txt', 'w') as f:
        for line in signal:
            for data in line:
                f.write("%f " %data)
            f.write('\n')
def writeTarget(target, name):
    with open('./'+name+'.txt', 'w') as f:
        for data in new_target:
            f.write("%i\n" %data)

writeSignal(new_data, 'new_body_acc_x_train')
writeTarget(new_target, 'new_y_train')
body_acc_x = pd.read_csv('./new_body_acc_x_train.txt', delim_whitespace=True, header=None).to_numpy()
y = pd.read_csv('./new_y_train.txt', delim_whitespace=True, header=None).to_numpy()
print(body_acc_x.shape)
print(y.shape)