import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

BATCH_START = 0
BATCH_SIZE = 50
TIME_STEPS = 20
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape(BATCH_SIZE, TIME_STEPS) / (1.*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    plt.show()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


model = Sequential()
model.add(LSTM(batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
               units=CELL_SIZE,
               return_sequences=True,
               stateful=True))
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse')

print 'Training ......'

for step in range(501):
    x_batch, y_batch, xs = get_batch()
    cost = model.train_on_batch(x_batch, y_batch)
    pred = model.predict(x_batch, BATCH_SIZE)
    plt.plot(xs[0, :], y_batch[0].flatten(), 'r', xs[0, :], pred, 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.5)
    if step % 10 == 0:
        print 'train cost:', cost


