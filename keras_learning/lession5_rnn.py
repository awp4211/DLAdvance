import numpy as np
import os
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 10
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001


(x_train, y_train), (x_test, y_test) = mnist.load_data(os.getcwd() + '/data/mnist.npz')

x_train = x_train.reshape(-1, 28, 28) / 255
x_test = x_test.reshape(-1, 28, 28) / 25
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(SimpleRNN(batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
                    units=CELL_SIZE))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

adam = Adam(LR)
model.compile(optimizer=adam,
              metrics=['accuracy'],
              loss='categorical_crossentropy')

for step in range(4001):
    x_batch = x_train[BATCH_INDEX : BATCH_SIZE + BATCH_INDEX, :, :]
    y_batch = y_train[BATCH_INDEX: BATCH_SIZE + BATCH_INDEX, :]
    cost = model.train_on_batch(x_batch, y_batch)

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX>= x_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
        print '\ntest cost :', cost, ' -- test accuracy :', accuracy


