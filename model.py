from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Flatten,LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

def RNN_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(128, 12)))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def CNN_model():
    model = Sequential()

    model.add(Conv2D(filters=48, kernel_size=(13,6), strides=(1,3),  input_shape=(128, 12, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

    model.add(Conv2D(filters=48, kernel_size=(13,3), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

    model.add(Conv2D(filters=48, kernel_size=(12,1), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model